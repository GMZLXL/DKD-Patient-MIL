"""Synthetic-only regression tests for independent controlled-run verification."""
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.verify_controlled_pooling import (evaluate,validation_threshold,checkpoint_selection,
    prediction_frame,training_positive_weight,replay_validation)

class ControlledVerificationTests(unittest.TestCase):
    def test_training_weight_matches_serialized_float32(self):
        import torch
        labels=[0]*75+[1]*56
        expected=float(torch.tensor([75/56],dtype=torch.float32))
        self.assertEqual(training_positive_weight(labels),expected)
        self.assertGreater(abs(expected-75/56),1e-12)
        with self.assertRaises(RuntimeError):training_positive_weight([1,1])

    def test_inner_replay_uses_real_dataset_fields_and_never_opens_outer_images(self):
        import torch
        from PIL import Image
        import experiments.controlled_pooling_adapter_ema.core as core
        torch.set_num_threads(2)
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.projection=torch.nn.Linear(3,1280)
            def forward(self,x):return self.projection(x.mean((-2,-1)))
        with tempfile.TemporaryDirectory() as temporary,patch.object(core,'make_encoder',side_effect=lambda weights:TinyEncoder()):
            folder=Path(temporary);rows=[]
            for i in range(2):
                path=folder/f'artificial_{i}.png'
                Image.fromarray(np.full((24,24,3),40+80*i,dtype=np.uint8)).save(path)
                rows.append(dict(patient_id=f'artificial_{i}',label=i,fold=1,resolved_path=str(path)))
            rows.append(dict(patient_id='artificial_outer',label=1,fold=0,resolved_path='outer-must-not-be-opened.png'))
            index=folder/'artificial_index.csv';pd.DataFrame(rows).to_csv(index,index=False)
            model=core.ControlledModel('C0','unused').eval();criterion=core.SharedEMAPCAR(torch.ones(1))
            criterion(torch.tensor([-.2,.2]),torch.tensor([0.,1.]),torch.randn(2,256),phase='train')
            checkpoint=dict(model=model.state_dict(),criterion=criterion.state_dict())
            replay,bce=replay_validation('C0',checkpoint,dict(pos_weight=1.),
                dict(val=['artificial_0','artificial_1'],test=['artificial_outer']),index,'unused')
            self.assertEqual(replay.patient_id.tolist(),['artificial_0','artificial_1'])
            self.assertEqual(replay.label.tolist(),[0,1])
            self.assertTrue(np.isfinite(replay.probability).all() and np.isfinite(bce))

    def test_known_nine_metrics_and_confusion(self):
        result=evaluate([0,0,1,1],[.1,.8,.7,.9],.5)
        expected=dict(roc_auc=.75,pr_auc=5/6,accuracy=.75,recall=1.,specificity=.5,
            precision=2/3,f1=.8,mcc=1/np.sqrt(3),balanced_accuracy=.75,TP=2,FP=1,TN=1,FN=0)
        for key,value in expected.items():self.assertAlmostEqual(result[key],value)

    def test_validation_f1_tie_rules(self):
        self.assertEqual(validation_threshold([0,0,1,1],[.1,.2,.8,.9]),.5)
        self.assertEqual(validation_threshold([0,1],[.2,.2]),.2)

    def test_pooled_uses_each_patients_fold_threshold(self):
        result=evaluate([0,1],[.4,.45],[.5,.3])
        self.assertEqual(result['accuracy'],1.)
        self.assertEqual(result['FN'],0)
        self.assertEqual(evaluate([0,1],[.4,.45],.5)['FN'],1)

    def test_checkpoint_selection_ties_patience_and_invalid_best(self):
        history=pd.DataFrame(dict(epoch=range(1,53),val_auc=[.9,.9,.9]+[.8]*49,
            val_raw_bce=[.4,.3,.3]+[.2]*49,adapter_gradient_l1=[1.]*52))
        checkpoint=dict(epoch=2,val_auc=.9,val_raw_bce=.3)
        self.assertEqual(checkpoint_selection(history,checkpoint),2)
        with self.assertRaises(RuntimeError):checkpoint_selection(history,dict(checkpoint,epoch=1))
        with self.assertRaises(RuntimeError):checkpoint_selection(history.iloc[:-1],checkpoint)
        extra=pd.concat([history,pd.DataFrame(dict(epoch=[53],val_auc=[.8],val_raw_bce=[.2],adapter_gradient_l1=[1.]))])
        with self.assertRaises(RuntimeError):checkpoint_selection(extra,checkpoint)

    def test_nonfinite_or_invalid_evaluation_is_rejected(self):
        for p in [[np.nan,.8],[-.1,.8],[.1,1.1]]:
            with self.assertRaises(RuntimeError):evaluate([0,1],p,.5)
        with self.assertRaises(RuntimeError):evaluate([1,1],[.1,.8],.5)

    def test_prediction_coverage_labels_and_outer_fold(self):
        index=pd.DataFrame(dict(patient_id=['artificial_a','artificial_b'],label=[0,1],fold=[0,0]))
        frame=index.assign(probability=[.2,.8])
        prediction_frame(frame,index.patient_id,index,0)
        for changed in [frame.iloc[:1],frame.assign(label=[1,0]),frame.assign(fold=[0,1])]:
            with self.assertRaises(RuntimeError):prediction_frame(changed,index.patient_id,index,0)

if __name__=='__main__':unittest.main()
