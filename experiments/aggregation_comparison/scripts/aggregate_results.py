"""Separate fold mean/std from pooled OOF; never fill missing runs with numbers."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.aggregators import AVAILABLE,BLOCKED
from src.patient_bags import load_cache
from src.metrics import metrics,NAMES

def main():
    bags,_=load_cache()
    summaries=[]; folds=[]; oofs=[]; pooled=[]
    for method in list(AVAILABLE)+list(BLOCKED):
        paths=sorted((ROOT/'runs'/method).glob('fold_*/test_patient_predictions.csv'))
        frames=[]
        for path in paths:
            df=pd.read_csv(path,dtype={'patient_id':str})
            assert not df.patient_id.duplicated().any()
            for r in df.itertuples(): assert bags[r.patient_id]['y']==r.label and bags[r.patient_id]['fold']==r.fold
            m=json.loads((path.parent/'test_metrics.json').read_text())
            folds.append(dict(method=method,fold=int(df.fold.iloc[0]),**m))
            frames.append(df)
        if frames:
            oof=pd.concat(frames,ignore_index=True)
            assert not oof.patient_id.duplicated().any()
            oof['method']=method; oofs.append(oof)
        complete=len(frames)==5 and len(oof)==205 and set(oof.patient_id)==set(bags) if frames else False
        row=dict(method=method,encoder='Frozen EfficientNet-B0 ImageNet',status='complete' if complete else BLOCKED.get(method,'incomplete'),completed_folds=len(frames))
        if complete:
            vals=[f for f in folds if f['method']==method]
            for k in NAMES:
                row[k+'_mean']=float(np.mean([v[k] for v in vals]))
                row[k+'_std']=float(np.std([v[k] for v in vals],ddof=1))
            # Apply each patient's own inner-validation-derived fold threshold.
            y=oof.label.values; p=oof.probability.values
            pm=metrics(y,p)
            classified=metrics(y,oof.prediction.values,.5)
            for k in NAMES[2:]: pm[k]=classified[k]
            rng=np.random.default_rng(42); auc=[]
            for _ in range(2000):
                idx=rng.integers(0,len(y),len(y))
                if len(np.unique(y[idx]))==2: auc.append(roc_auc_score(y[idx],p[idx]))
            pm.update(auc_ci_low=float(np.quantile(auc,.025)),auc_ci_high=float(np.quantile(auc,.975)),ci_method='patient percentile bootstrap; 2000 replicates; fixed OOF predictions; seed 42',patients=205)
            pooled.append(dict(method=method,**pm))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(ROOT/'reports/aggregation_comparison.csv',index=False)
    pd.DataFrame(folds).to_csv(ROOT/'reports/per_fold_metrics.csv',index=False)
    pd.concat(oofs,ignore_index=True).to_csv(ROOT/'reports/oof_predictions.csv',index=False) if oofs else pd.DataFrame(columns=['method','patient_id','fold','label','probability','prediction','threshold']).to_csv(ROOT/'reports/oof_predictions.csv',index=False)
    pd.DataFrame(pooled).to_csv(ROOT/'reports/pooled_oof_metrics.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__': main()
