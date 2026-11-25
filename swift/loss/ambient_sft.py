# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F

from .base import BaseLoss
from .causal_lm import CustomCrossEntropyLoss


class AmbientSFTLoss(BaseLoss):
    """Ambient SFT (https://huggingface.co/papers/2611.23229).

    On a random fraction (1 - `ambient_clean_probability`) of the training micro-batches, the cross-entropy to the
    labels is replaced by the cross-entropy from the frozen initial model's next-token distribution at temperature
    `ambient_temperature` to the model's distribution at the same temperature, at every supervised position. That
    term is scaled by the ratio of the initial model's entropy at `ambient_temperature` to its entropy at temperature
    1, and reduced like the cross-entropy loss. With LoRA, the initial model is the model with its adapters disabled;
    otherwise a frozen copy is loaded from the model directory.
    """

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.cross_entropy = CustomCrossEntropyLoss(args, trainer)
        # Every rank draws the same coins in the same order, so they agree without communication
        self.generator = torch.Generator().manual_seed(args.seed)
        self.teacher_model = None
        self.use_disable_adapter = None

    def _setup_teacher(self):
        from peft import PeftModel
        trainer = self.trainer
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        self.use_disable_adapter = isinstance(unwrapped_model, PeftModel)
        if self.use_disable_adapter:
            return
        teacher_model = type(unwrapped_model).from_pretrained(
            unwrapped_model.model_dir,
            torch_dtype=unwrapped_model.dtype,
            attn_implementation=unwrapped_model.config._attn_implementation)
        teacher_model.requires_grad_(False)
        if trainer.is_deepspeed_enabled:
            from swift.rlhf_trainers.utils import prepare_deepspeed
            self.teacher_model = prepare_deepspeed(teacher_model, trainer.accelerator)
        elif trainer.is_fsdp_enabled:
            from swift.rlhf_trainers.utils import prepare_fsdp
            self.teacher_model = prepare_fsdp(teacher_model, trainer.accelerator)
        else:
            self.teacher_model = trainer.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.teacher_model.eval()

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, inputs=None, **kwargs):
        trainer = self.trainer
        if not trainer.model.training:
            return self.cross_entropy(outputs, labels, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale)
        if self.use_disable_adapter is None:
            # Set up on the first training micro-batch, before any update, so a copy holds the initial weights
            self._setup_teacher()
        # One coin per micro-batch, so that a step with gradient accumulation mixes both losses
        if torch.rand(1, generator=self.generator).item() < self.args.ambient_clean_probability:
            trainer.custom_metrics['train']['ambient_matching_fraction'].update(torch.tensor(0.))
            return self.cross_entropy(outputs, labels, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale)

        from swift.trainers.utils import disable_gradient_checkpointing
        gc_kwargs = trainer.args.gradient_checkpointing_kwargs
        with torch.no_grad():
            if self.use_disable_adapter:
                unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
                with unwrapped_model.disable_adapter(), disable_gradient_checkpointing(unwrapped_model, gc_kwargs):
                    teacher_logits = unwrapped_model(**inputs).logits
            else:
                teacher_logits = self.teacher_model(**inputs).logits

        # Same alignment as the cross-entropy loss: the logits at position t predict the label at t + 1
        shift_labels = torch.roll(labels, shifts=-1, dims=-1).to(outputs.logits.device)
        loss_mask = shift_labels != -100
        temperature = self.args.ambient_temperature
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
        log_probs = F.log_softmax(outputs.logits.float() / temperature, dim=-1)
        per_token_loss = -(teacher_probs * log_probs).sum(-1) * loss_mask
        with torch.no_grad():
            entropy = -(teacher_probs * teacher_probs.clamp(min=1e-12).log()).sum(-1)
            teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
            base_entropy = -(teacher_log_probs.exp() * teacher_log_probs).sum(-1)
            entropy_scale = (entropy * loss_mask).sum() / (base_entropy * loss_mask).sum().clamp(min=1e-8)
        if loss_scale is not None:
            # Already shifted and flattened by the trainer, as for the cross-entropy loss
            per_token_loss = per_token_loss * loss_scale.view_as(per_token_loss).to(per_token_loss.device)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()

        metrics = trainer.custom_metrics['train']
        metrics['ambient_matching_fraction'].update(torch.tensor(1.))
        metrics['ambient_entropy_scale'].update(entropy_scale.detach())
        return entropy_scale * per_token_loss.sum() / num_items_in_batch
