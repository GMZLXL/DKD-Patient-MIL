"""Synthetic-only regression tests for independent controlled-run verification."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.verify_controlled_pooling import evaluate,validation_threshold,checkpoint_selection,prediction_frame

class ControlledVerificationTests(unittest.TestCase):
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
