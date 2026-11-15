"""Synthetic verifier tests; intentionally outside the training gate's test list."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import verify_v2_runs as verify

def evidence_fixture():
    rows=[]
    for epoch in range(1,11):
        active=epoch>=10
        rows.append(dict(epoch=epoch,checkpoint_eligible=active,save_best=active,
            eligible_stale=0,stop=False,best_epoch=10 if active else np.nan,
            auxiliary_start_epoch=10,checkpoint_eligible_epoch=10,auxiliary_active=active,
            auxiliary_optimizer_steps=int(active),val_auc=.6 if active else .99,val_ce=.7))
    evidence=pd.DataFrame(rows)
    history=pd.DataFrame(dict(epoch=range(1,11),val_roc_auc=evidence.val_auc,
        val_loss=evidence.val_ce,objective_auxiliary_active=evidence.auxiliary_active.astype(float)))
    checkpoint=dict(epoch=10,val_auc=.6,val_loss=.7,auxiliary_active_at_best_checkpoint=True,auxiliary_optimizer_steps=1)
    return history,evidence,checkpoint

class VerifierTests(unittest.TestCase):
    def test_qualified_checkpoint_beats_higher_warmup_auc_by_eligibility(self):
        h,e,c=evidence_fixture()
        self.assertEqual(verify.validate_selection(h,e,c,'dgr_fullobj'),10)

    def test_rejects_warmup_checkpoint(self):
        h,e,c=evidence_fixture();c.update(epoch=4,val_auc=.99)
        with self.assertRaises(ValueError):verify.validate_selection(h,e,c,'dgr_fullobj')

    def test_rejects_fake_auxiliary_evidence_and_patience(self):
        h,e,c=evidence_fixture();c['auxiliary_optimizer_steps']=0
        with self.assertRaises(ValueError):verify.validate_selection(h,e,c,'dgr_fullobj')
        h,e,c=evidence_fixture();e.loc[2,'eligible_stale']=1
        with self.assertRaises(ValueError):verify.validate_selection(h,e,c,'dgr_fullobj')
        h,e,c=evidence_fixture();h.loc[9,'objective_auxiliary_active']=0
        with self.assertRaises(ValueError):verify.validate_selection(h,e,c,'dgr_fullobj')

    def test_rejects_non_best_eligible_checkpoint(self):
        h,e,c=evidence_fixture()
        h.loc[len(h)]=[11,.8,.5,1.]
        row=e.iloc[-1].copy();row.update(dict(epoch=11,val_auc=.8,val_ce=.5,auxiliary_optimizer_steps=2,best_epoch=11))
        e.loc[len(e)]=row
        with self.assertRaises(ValueError):verify.validate_selection(h,e,c,'dgr_fullobj')

    def test_independent_metrics_and_threshold(self):
        values=verify.metrics_independent([0,0,1,1],[.1,.7,.8,.9],[0,1,1,1])
        self.assertEqual(values['roc_auc'],1.)
        self.assertEqual(values['accuracy'],.75)
        self.assertEqual(values['specificity'],.5)
        self.assertEqual(values['f1'],.8)
        self.assertEqual(verify.threshold_independent([0,0,1,1],np.array([.1,.2,.3,.4])),.3)

    def test_patient_paired_bootstrap_keeps_identical_predictions_identical(self):
        n=205;y=np.arange(n)%2
        frame=pd.DataFrame(dict(patient_id=[f'artificial_{i:03d}' for i in range(n)],label=y,
            fold=np.arange(n)%5,probability=.2+.6*y,prediction=y,threshold=.5))
        frames={method:frame.copy() for method in verify.LABELS}
        stats=verify.metrics_independent(y,frame.probability,y)
        records=[dict(method=m,fold=f,**stats) for m in verify.LABELS for f in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            info=verify.write_statistics(records,frames,out,repeats=20,seed=42)
            paired=pd.read_csv(out/'paired_comparison_bootstrap.csv')
            self.assertTrue((paired[['pooled_delta','ci_low','ci_high']]==0).all().all())
            self.assertEqual(info['unit'],'patient');self.assertEqual(info['valid'],20)
            self.assertEqual(len(pd.read_csv(out/'aggregation_comparison_original_5methods.csv')),5)
            self.assertEqual(len(pd.read_csv(out/'aggregation_comparison_v2_sensitivity.csv')),5)

if __name__=='__main__':unittest.main()
