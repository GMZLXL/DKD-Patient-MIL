"""Synthetic regression tests for the SAME split/threshold code used in the runs.

The original real-cache test_protocol.py is preserved byte-for-byte under
experiments/aggregation_comparison/tests; no private fixtures are published.
"""
import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
COMPARISON=ROOT/'experiments/aggregation_comparison'
sys.path.insert(0,str(COMPARISON))
from src.patient_bags import split
from src.metrics import metrics,threshold
spec=importlib.util.spec_from_file_location('adapted_trainer',COMPARISON/'scripts/train_adapted_cv.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)

def artificial_bags():
    generator=torch.Generator().manual_seed(123)
    return {f'synthetic_{i:03d}':dict(x=torch.randn(1+i%9,16,generator=generator),y=i%2,fold=i%5) for i in range(205)}

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.classifier=torch.nn.Linear(16,2);self.config={'synthetic':True}
    def forward(self,x):return {'logits':self.classifier(x.mean(0))}
    def objective(self,output,label,epoch=1):
        loss=torch.nn.functional.cross_entropy(output['logits'].reshape(1,2),torch.tensor([label]))
        return loss,{'classification':loss}

class ProtocolTests(unittest.TestCase):
    def test_patient_isolation_coverage_and_same_splits(self):
        bags=artificial_bags();seen=[]
        for fold in range(5):
            s=split(bags,fold,42)
            self.assertEqual(s,split(bags,fold,42))
            self.assertFalse(set(s['train'])&set(s['val'])|set(s['train'])&set(s['test'])|set(s['val'])&set(s['test']))
            self.assertEqual(set().union(*map(set,s.values())),set(bags));seen+=s['test']
        self.assertEqual(len(seen),205);self.assertEqual(len(set(seen)),205)

    def test_known_metrics_and_validation_threshold(self):
        m=metrics([0,0,1,1],[.1,.7,.8,.9],.5)
        self.assertEqual(m['roc_auc'],1.);self.assertEqual(m['accuracy'],.75)
        self.assertEqual(m['specificity'],.5);self.assertEqual(m['recall'],1.)
        self.assertAlmostEqual(m['f1'],.8)
        self.assertEqual(threshold([0,0,1,1],[.1,.2,.3,.4]),.3)

    def test_test_labels_cannot_change_checkpoint_or_threshold(self):
        torch.set_num_threads(2)
        bags=artificial_bags();s=split(bags,0,42)
        changed={p:dict(b,y=1-b['y']) if b['fold']==0 else b for p,b in bags.items()}
        self.assertEqual(s,split(changed,0,42))
        outcomes=[]
        for data in [bags,changed]:
            with tempfile.TemporaryDirectory() as tmp:
                args=argparse.Namespace(run_root=tmp,method='acmil',seed=42,smoke=False,epochs=3,patience=2,batch_size=16,lr=.001,weight_decay=.0001)
                calls=[];original=trainer.predict
                def spy(model,b,ids):
                    calls.append(list(ids));return original(model,b,ids)
                with mock.patch.object(trainer,'build',side_effect=lambda *a,**k:Toy()),mock.patch.object(trainer,'predict',side_effect=spy):
                    trainer.fit_fold(args,data,{'feature_dim':16,'features_sha256':'synthetic'},0)
                self.assertEqual(sum(ids==s['test'] for ids in calls),1)
                self.assertEqual(calls[-1],s['test'])
                folder=Path(tmp)/'acmil/fold_0'
                history=pd.read_csv(folder/'train_history.csv')
                self.assertFalse(any('test' in name for name in history.columns))
                checkpoint=torch.load(folder/'best_model.pt',weights_only=True)
                t=json.loads((folder/'val_threshold.json').read_text())['threshold']
                outcomes.append((checkpoint['epoch'],t,checkpoint['state_dict']))
        self.assertEqual(outcomes[0][:2],outcomes[1][:2])
        for key in outcomes[0][2]:torch.testing.assert_close(outcomes[0][2][key],outcomes[1][2][key],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
