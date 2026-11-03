import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.patient_bags import load_cache, split
from src.metrics import metrics, threshold

class ProtocolTests(unittest.TestCase):
    def test_outer_fold_isolation_and_coverage(self):
        bags,_=load_cache()
        seen=[]
        for fold in range(5):
            s=split(bags,fold,42)
            self.assertEqual(s,split(bags,fold,42))
            self.assertFalse(set(s['train'])&set(s['val']))
            self.assertFalse(set(s['train'])&set(s['test']))
            self.assertFalse(set(s['val'])&set(s['test']))
            self.assertEqual(set().union(*map(set,s.values())),set(bags))
            self.assertTrue(all(bags[p]['fold']==fold for p in s['test']))
            seen+=s['test']
        self.assertEqual(len(seen),205)
        self.assertEqual(len(set(seen)),205)

    def test_outer_labels_do_not_influence_inner_split(self):
        bags,_=load_cache()
        s=split(bags,0,42)
        altered={pid:dict(b,y=1-b['y']) if b['fold']==0 else b for pid,b in bags.items()}
        self.assertEqual(s,split(altered,0,42))

    def test_known_confusion_matrix(self):
        m=metrics([0,0,1,1],[.1,.7,.8,.9],.5)
        self.assertEqual(m['roc_auc'],1.)
        self.assertEqual(m['accuracy'],.75)
        self.assertEqual(m['specificity'],.5)
        self.assertEqual(m['recall'],1.)
        self.assertAlmostEqual(m['f1'],.8)
        self.assertAlmostEqual(m['balanced_accuracy'],.75)

    def test_validation_threshold(self):
        y=[0,0,1,1]; p=[.1,.2,.3,.4]
        t=threshold(y,p)
        self.assertAlmostEqual(t,.3)
        self.assertEqual(metrics(y,p,t)['f1'],1.)

if __name__=='__main__': unittest.main()
