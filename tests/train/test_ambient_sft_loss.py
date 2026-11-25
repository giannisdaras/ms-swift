# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import tempfile
import unittest

MODEL = 'trl-internal-testing/tiny-Qwen2ForCausalLM-2.5'


class TestAmbientSFTLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.dataset = os.path.join(cls.tmp_dir.name, 'data.jsonl')
        with open(cls.dataset, 'w') as f:
            for i in range(32):
                messages = [{
                    'role': 'user',
                    'content': f'What is {i} plus {i}?'
                }, {
                    'role': 'assistant',
                    'content': f'The answer is {2 * i}.'
                }]
                f.write(json.dumps({'messages': messages}) + '\n')

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def _train(self, **kwargs):
        from swift import SftArguments, sft_main
        params = {
            'model': MODEL,
            'model_type': 'qwen2',
            'template': 'qwen2_5',
            'use_hf': True,
            'dataset': [self.dataset],
            'split_dataset_ratio': 0,
            'tuner_type': 'full',
            'loss_type': 'ambient_sft',
            'max_steps': 4,
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 2,
            'logging_steps': 1,
            'save_strategy': 'no',
            'report_to': [],
            'torch_dtype': 'float32',
            'seed': 0,
            'dataloader_num_workers': 0,
        }
        params.update(kwargs)
        with tempfile.TemporaryDirectory() as output_dir:
            result = sft_main(SftArguments(output_dir=output_dir, **params))
        return [log for log in result['log_history'] if 'loss' in log]

    def test_clean_probability_one_is_cross_entropy(self):
        ambient = self._train(ambient_clean_probability=1.0)
        cross_entropy = self._train(loss_type='cross_entropy')
        for a, b in zip(ambient, cross_entropy, strict=True):
            self.assertAlmostEqual(a['loss'], b['loss'], places=5)
            self.assertAlmostEqual(a['grad_norm'], b['grad_norm'], places=5)

    def test_initial_model_is_the_teacher(self):
        # The model starts at the initial model, where the matching loss is stationary: the first gradient is zero
        for tuner_type in ['full', 'lora']:
            with self.subTest(tuner_type=tuner_type):
                logs = self._train(
                    tuner_type=tuner_type, ambient_clean_probability=0.0, max_steps=1, gradient_accumulation_steps=1)
                self.assertEqual(logs[0]['ambient_matching_fraction'], 1.0)
                self.assertGreater(logs[0]['ambient_entropy_scale'], 1.0)  # a hotter distribution has more entropy
                self.assertLess(logs[0]['grad_norm'], 1e-5)

    def test_coin_per_micro_batch(self):
        logs = self._train(ambient_clean_probability=0.5, max_steps=8)
        fractions = [log['ambient_matching_fraction'] for log in logs]
        self.assertIn(0.5, fractions)  # some steps mix both losses
        self.assertEqual(
            fractions,
            [log['ambient_matching_fraction']
             for log in self._train(ambient_clean_probability=0.5, max_steps=8)])  # same seed, same coins


if __name__ == '__main__':
    unittest.main()
