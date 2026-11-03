"""Verify saved runs without reevaluating any model on outer test patients."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.metrics import metrics,threshold,NAMES

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-runs',type=Path,required=True)
    p.add_argument('--adapted-runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('Refusing to overwrite verification output')
    a.out.mkdir(parents=True)
    records=[];frames=[];summary=[];issues=[]
    for method in ['mean','gated_abmil','phpr','dgr_mil','acmil']:
        root=a.baseline_runs if method in ['mean','gated_abmil','phpr'] else a.adapted_runs
        per_method=[]
        for fold in range(5):
            folder=root/method/f'fold_{fold}'
            required=['config_resolved.yaml','split_patient_ids.json','best_model.pt','train_history.csv','val_threshold.json','test_patient_predictions.csv','test_metrics.json','run.log']
            if not all((folder/f).is_file() for f in required):raise RuntimeError('Incomplete fold: '+str(folder))
            split=json.loads((folder/'split_patient_ids.json').read_text())
            reference=json.loads((a.baseline_runs/'mean'/f'fold_{fold}'/'split_patient_ids.json').read_text())
            assert split==reference,'Methods must use identical splits'
            assert not set(split['train'])&set(split['val'])|set(split['train'])&set(split['test'])|set(split['val'])&set(split['test'])
            prediction=pd.read_csv(folder/'test_patient_predictions.csv',dtype={'patient_id':str})
            assert sorted(prediction.patient_id)==sorted(split['test']) and not prediction.patient_id.duplicated().any()
            truth=pd.read_csv(a.baseline_runs/'mean'/f'fold_{fold}'/'test_patient_predictions.csv',dtype={'patient_id':str})
            pd.testing.assert_frame_equal(prediction[['patient_id','label','fold']].sort_values('patient_id').reset_index(drop=True),truth[['patient_id','label','fold']].sort_values('patient_id').reset_index(drop=True))
            h=pd.read_csv(folder/'train_history.csv');assert not any('test' in c for c in h)
            assert np.isfinite(h.to_numpy()).all()
            checkpoint=torch.load(folder/'best_model.pt',map_location='cpu',weights_only=True)
            best=h.sort_values(['val_roc_auc','val_loss'],ascending=[False,True]).iloc[0]
            assert checkpoint['epoch']==int(best.epoch)
            validation=pd.read_csv(folder/'val_predictions.csv')
            selected=json.loads((folder/'val_threshold.json').read_text())['threshold']
            assert abs(threshold(validation.label,validation.probability)-selected)<1e-7
            assert np.allclose(prediction.threshold,selected)
            assert (prediction.prediction==(prediction.probability>=selected)).all()
            report=json.loads((folder/'test_metrics.json').read_text())
            assert report['outer_test_evaluations']==1
            for k,v in metrics(prediction.label,prediction.probability,selected).items():assert abs(v-report[k])<1e-10
            row={'method':method,'fold':fold,**report}
            if method=='dgr_mil':
                row['auxiliary_active_at_best_checkpoint']=checkpoint['epoch']>=10
                assert (h.loc[h.epoch>=10,'objective_auxiliary_active']==1).all()
                if checkpoint['epoch']<10:issues.append(f'DGR fold {fold}: selected validation-best checkpoint predates auxiliary activation')
            records.append(row);per_method.append(row)
            prediction['method']=method;frames.append(prediction)
        oof=pd.concat([f for f in frames if f.method.iloc[0]==method])
        assert len(oof)==205 and oof.patient_id.nunique()==205
        summary.append(dict(method=method,variant=method+'-adapted' if method in ['dgr_mil','acmil'] else method,**{k+'_mean':float(np.mean([r[k] for r in per_method])) for k in NAMES},**{k+'_std':float(np.std([r[k] for r in per_method],ddof=1)) for k in NAMES}))
    pd.DataFrame(records).to_csv(a.out/'per_fold_metrics.csv',index=False)
    pd.DataFrame(summary).to_csv(a.out/'aggregation_comparison.csv',index=False)
    pd.concat(frames,ignore_index=True).to_csv(a.out/'oof_predictions.csv',index=False)
    result=dict(passed=True,verified_folds=25,oof_rows=1025,patients_per_method=205,
        identical_splits=True,checkpoint_and_threshold_selection_verified=True,
        new_outer_test_model_evaluations=0,limitations=issues)
    (a.out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
