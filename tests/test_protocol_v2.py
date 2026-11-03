"""Synthetic qualification, compatibility, and inner-only isolation regression."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import pandas as pd
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
COMPARISON = REPO/'experiments/aggregation_comparison'
sys.path.insert(0, str(COMPARISON))
sys.path.insert(0, str(COMPARISON/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_protocol import artificial_bags
from src.patient_bags import split
from src.patient_bags import digest
from src.inner_cache_v2 import load_inner_cache
from src.adapted_models.dgr_mil import DGRMILPatient
from src.protocol_v2 import CheckpointPolicy, inner_split_without_test_labels
import train_adapted_cv as legacy
import train_adapted_cv_v2 as revised


class ForbiddenOuter(dict):
    def __getitem__(self, key):
        if key in {'y', 'x'}:
            raise AssertionError('Outer test labels/features were accessed')
        return super().__getitem__(key)


def args_for(root, epochs=12):
    return argparse.Namespace(run_root=root, method='dgr_fullobj', mode='smoke',
        formal_authorized=False, epochs=epochs, seed=42, patience=2,
        batch_size=16, lr=.001, weight_decay=.0001)


class ProtocolV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_warmup_neither_selects_nor_counts_patience(self):
        policy = CheckpointPolicy(patience=2)
        for epoch in range(1, 10):
            decision = policy.observe(epoch, 1., .01)
            self.assertFalse(decision['checkpoint_eligible'] or decision['save_best'] or decision['stop'])
            self.assertEqual(policy.stale, 0)
            self.assertIsNone(policy.best_epoch)
        decision = policy.observe(10, .6, .8, auxiliary_active=True, auxiliary_optimizer_steps=1)
        self.assertTrue(decision['save_best'])
        self.assertEqual(policy.best_epoch, 10)
        self.assertFalse(policy.observe(11, .6, .8, auxiliary_active=True, auxiliary_optimizer_steps=2)['stop'])
        self.assertTrue(policy.observe(12, .6, .8, auxiliary_active=True, auxiliary_optimizer_steps=3)['stop'])

    def test_eligible_tie_break_and_rejects_missing_auxiliary_evidence(self):
        policy = CheckpointPolicy()
        self.assertTrue(policy.observe(10, .7, .5, auxiliary_active=True, auxiliary_optimizer_steps=1)['save_best'])
        self.assertTrue(policy.observe(11, .7, .4, auxiliary_active=True, auxiliary_optimizer_steps=2)['save_best'])
        self.assertFalse(policy.observe(12, .7, .4, auxiliary_active=True, auxiliary_optimizer_steps=3)['save_best'])
        with self.assertRaises(RuntimeError):
            CheckpointPolicy().observe(10, .7, .4, auxiliary_active=False, auxiliary_optimizer_steps=0)
        with self.assertRaises(RuntimeError):
            policy.validate_checkpoint(dict(epoch=9, auxiliary_active_at_best_checkpoint=True, auxiliary_optimizer_steps=1))
        with self.assertRaises(RuntimeError):
            policy.validate_checkpoint(dict(epoch=11, auxiliary_active_at_best_checkpoint=False, auxiliary_optimizer_steps=0))
        with self.assertRaises(ValueError):
            CheckpointPolicy(checkpoint_eligible_epoch=9)

    def test_inner_split_matches_history_without_outer_labels(self):
        bags = artificial_bags()
        for fold in range(5):
            expected = split(bags, fold, 42)
            guarded = {p: ForbiddenOuter(b) if b['fold']==fold else b for p, b in bags.items()}
            self.assertEqual(inner_split_without_test_labels(guarded, fold, 42), expected)

    def test_smoke_cache_does_not_materialize_outer_labels_or_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            features = np.ones((703, 16), dtype=np.float32)
            for i in range(703):
                pid = i % 205
                held_out = pid % 5 == 0
                rows.append(dict(patient_id=f'artificial_{pid:03d}', fold=pid % 5,
                                 label='DO_NOT_PARSE' if held_out else str(pid % 2), feature_row=i))
                if held_out:
                    features[i] = np.nan  # would fail if test vectors were inspected
            np.save(root/'features.npy', features)
            pd.DataFrame(rows).to_csv(root/'image_index.csv', index=False)
            (root/'metadata.json').write_text(json.dumps(dict(feature_dim=16,
                features_sha256=digest(root/'features.npy'), index_sha256=digest(root/'image_index.csv'))))
            bags, _ = load_inner_cache(root, 0)
            self.assertEqual(len(bags), 205)
            for bag in bags.values():
                if bag['fold'] == 0:
                    self.assertEqual(set(bag), {'fold'})
                else:
                    self.assertTrue(torch.isfinite(bag['x']).all())
            inner_split_without_test_labels(bags, 0, 42)

    def test_actual_qualified_training_roundtrip_and_outer_label_invariance(self):
        bags = artificial_bags()
        s = split(bags, 0, 42)
        changed = {p: ForbiddenOuter(dict(b, y=1-b['y'])) if b['fold']==0 else b for p, b in bags.items()}
        outcomes = []
        for data in [bags, changed]:
            with tempfile.TemporaryDirectory() as root:
                original_predict = revised.predict
                def predict_inner(model, data, ids):
                    self.assertFalse(set(ids) & set(s['test']))
                    return original_predict(model, data, ids)
                with mock.patch.object(revised, 'build_v2', side_effect=lambda method, d: DGRMILPatient(d, hidden_dim=32)), \
                        mock.patch.object(revised, 'predict', side_effect=predict_inner):
                    report = revised.fit_fold_v2(args_for(root), data,
                        {'feature_dim':16, 'features_sha256':'artificial'}, 0)
                self.assertTrue(report['passed'] and report['checkpoint_roundtrip'])
                self.assertEqual(report['outer_test_evaluations'], 0)
                self.assertGreaterEqual(report['best_epoch'], 10)
                self.assertTrue(report['auxiliary_active_at_best_checkpoint'])
                self.assertGreater(report['auxiliary_optimizer_steps_at_best_checkpoint'], 0)
                folder = Path(root)/'dgr_fullobj/fold_0'
                self.assertFalse((folder/'test_patient_predictions.csv').exists())
                evidence = pd.read_csv(folder/'checkpoint_selection.csv')
                self.assertFalse(evidence.loc[evidence.epoch<10, 'save_best'].any())
                self.assertTrue((evidence.loc[evidence.epoch<10, 'eligible_stale']==0).all())
                selected = evidence[evidence.save_best]
                self.assertTrue((selected.epoch>=10).all())
                self.assertTrue(selected.auxiliary_active.all())
                self.assertTrue((selected.auxiliary_optimizer_steps>0).all())
                checkpoint = torch.load(folder/'best_model.pt', weights_only=True)
                t = json.loads((folder/'val_threshold.json').read_text())['threshold']
                outcomes.append((checkpoint, t))
        self.assertEqual(outcomes[0][0]['epoch'], outcomes[1][0]['epoch'])
        self.assertEqual(outcomes[0][1], outcomes[1][1])
        for key in outcomes[0][0]['state_dict']:
            torch.testing.assert_close(outcomes[0][0]['state_dict'][key], outcomes[1][0]['state_dict'][key], atol=0, rtol=0)

    def test_old_dgr_smoke_still_selects_warmup_and_builder_is_unchanged(self):
        self.assertEqual(legacy.build('acmil', 16).__class__.__name__, 'ACMILPatient')
        bags = artificial_bags()
        with tempfile.TemporaryDirectory() as root:
            args = argparse.Namespace(run_root=root, method='dgr_mil', seed=42,
                smoke=True, epochs=3, patience=2, batch_size=16, lr=.001, weight_decay=.0001)
            with mock.patch.object(legacy, 'build', side_effect=lambda method, d: DGRMILPatient(d, hidden_dim=32)):
                report = legacy.fit_fold(args, bags, {'feature_dim':16, 'features_sha256':'artificial'}, 0)
            checkpoint = torch.load(Path(root)/'dgr_mil/fold_0/best_model.pt', weights_only=True)
            self.assertLess(checkpoint['epoch'], 10)
            self.assertEqual(report['outer_test_evaluations'], 0)

    def test_cli_rejects_short_qualification_and_unauthorized_formal_runs(self):
        common = ['--method', 'dgr_fullobj', '--cache', 'ARTIFICIAL', '--run-root', 'ARTIFICIAL']
        for extra in [['--mode','formal'], ['--mode','smoke','--fold','all'], ['--smoke-epochs','2']]:
            with self.assertRaises(SystemExit):
                revised.parse_args(common+extra)
        self.assertEqual(revised.parse_args(common).mode, 'smoke')
        self.assertEqual(revised.parse_args(common).epochs, 10)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                revised.fit_fold_v2(args_for(root, epochs=9), artificial_bags(),
                    {'feature_dim':16, 'features_sha256':'artificial'}, 0)
            self.assertFalse(any(Path(root).iterdir()))


if __name__ == '__main__':
    unittest.main()
