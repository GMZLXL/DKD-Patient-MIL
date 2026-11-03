"""Artificial features only; official-ratio correction does not replace the old model."""
import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/aggregation_comparison'))
sys.path.insert(0, str(ROOT/'experiments/aggregation_comparison/scripts'))
from src.adapted_models.acmil import ACMILPatient
from src.adapted_models.acmil_official_ratio import ACMILOfficialRatio
from src.adapted_models.dgr_mil import DGRMILPatient
from train_adapted_cv_v2 import dropout_audit


class ModelV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_exact_mask_counts_reproducibility_and_gradients_k1_to_9(self):
        old_counts, new_counts = [], []
        for k in range(1, 10):
            old = ACMILPatient(input_dim=32, hidden_dim=32).train()
            new = ACMILOfficialRatio(input_dim=32, hidden_dim=32).train()
            new.load_state_dict(old.state_dict())
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))
            x = torch.randn(k, 32, requires_grad=True)
            old_counts.append(old(x)['masked_count'])
            torch.manual_seed(123)
            output = new(x)
            torch.manual_seed(123)
            repeat = new(x)
            new_counts.append(output['masked_count'])
            self.assertEqual(output['candidate_count'], k)
            self.assertTrue(torch.equal(output['keep_mask'], repeat['keep_mask']))
            self.assertTrue((output['keep_mask'].sum(-1) == k-output['masked_count']).all())
            self.assertTrue(output['keep_mask'].any(-1).all())
            self.assertTrue((output['attention'][~output['keep_mask']] == 0).all())
            torch.testing.assert_close(output['attention'].sum(-1), torch.ones(5))
            loss, _ = new.objective(output, k % 2)
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(x.grad).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters()))
        self.assertEqual(old_counts, [0, 0, 1, 1, 2, 3, 3, 4, 4])
        self.assertEqual(new_counts, [0, 1, 1, 2, 3, 3, 4, 4, 5])

    def test_worst_case_mask_retains_one_valid_instance(self):
        model = ACMILOfficialRatio(input_dim=32, hidden_dim=32, mask_drop=1.).train()
        for k in range(1, 10):
            output = model(torch.randn(k, 32))
            self.assertEqual(output['masked_count'], k-1)
            self.assertTrue((output['keep_mask'].sum(-1) == 1).all())
            self.assertTrue(torch.isfinite(output['logits']).all())

    def test_padding_eval_equivalence_order_and_all_instance_gradients(self):
        new = ACMILOfficialRatio(input_dim=32, hidden_dim=32)
        old = ACMILPatient(input_dim=32, hidden_dim=32)
        old.load_state_dict(new.state_dict())
        for k in range(1, 10):
            x = torch.randn(k, 32)
            padded = torch.cat([torch.full((2, 32), float('nan')), x, torch.full((3, 32), 1e9)]).requires_grad_()
            mask = torch.tensor([False]*2+[True]*k+[False]*3)
            new.train()
            torch.manual_seed(7)
            train_padded = new(padded, mask)
            torch.manual_seed(7)
            train_plain = new(x)
            torch.testing.assert_close(train_padded['logits'], train_plain['logits'], atol=0, rtol=0)
            train_padded['logits'].sum().backward()
            self.assertTrue(torch.isfinite(padded.grad).all())
            self.assertTrue((padded.grad[~mask] == 0).all())
            new.eval(); old.eval()
            actual = x.clone().requires_grad_()
            output = new(actual)
            self.assertEqual(output['masked_count'], 0)
            output['logits'][1].backward()
            self.assertTrue((actual.grad.abs().sum(-1) > 0).all())
            with torch.no_grad():
                for expected in [new(x)['logits'], new(x.flip(0))['logits'], old(x)['logits'], new(padded, mask)['logits']]:
                    torch.testing.assert_close(output['logits'], expected, atol=2e-5, rtol=2e-5)
        with self.assertRaises(ValueError):
            new(torch.zeros(3, 32), torch.zeros(3, dtype=torch.bool))

    def test_dgr_dropout_audit_and_auxiliary_gradients(self):
        model = DGRMILPatient(input_dim=32, hidden_dim=32).train()
        audit = dropout_audit(model)
        self.assertEqual(audit['projection_dropout_modules'], [])
        self.assertEqual(audit['output_dropout'], {
            'instances.attention.out.1': .1, 'globals.attention.out.1': .1})
        self.assertEqual(audit['cross_attention_probability_dropout'], .1)
        output = model(torch.randn(9, 32))
        full, parts = model.objective(output, 1, epoch=10, update_centers=False)
        gradient = torch.autograd.grad(full-parts['classification'], model.global_codes)[0]
        self.assertTrue(parts['auxiliary_active'])
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
