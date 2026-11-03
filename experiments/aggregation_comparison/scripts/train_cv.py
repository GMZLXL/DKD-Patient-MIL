"""Nested patient splits; outer test accessed once after validation selection."""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.aggregators import build, AVAILABLE, BLOCKED, PHPR_SOURCE
from src.patient_bags import load_cache, split, digest
from src.metrics import metrics, threshold

def predict(model,bags,ids):
    model.eval()
    with torch.no_grad():
        logits = torch.stack([model(bags[p]['x']) for p in ids])
        y = torch.tensor([bags[p]['y'] for p in ids],dtype=torch.float32)
        loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits,y))
    return y.numpy().astype(int),logits.sigmoid().numpy(),loss

def train_fold(a,bags,meta,fold):
    out = ROOT/'runs'/a.method/f'fold_{fold}'
    if out.exists(): raise RuntimeError('Refusing to overwrite existing fold: '+str(out))
    out.mkdir(parents=True)
    start = time.time()
    def log(s):
        print(s,flush=True)
        with (out/'run.log').open('a',encoding='utf-8') as f: f.write(s+'\n')
    try:
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.use_deterministic_algorithms(True)
        s = split(bags,fold,a.seed)
        (out/'split_patient_ids.json').write_text(json.dumps(s,indent=2))
        cfg = dict(vars(a),outer_fold=fold,device='cpu',torch=str(torch.__version__),cache=meta,
            selection='maximum inner validation ROC-AUC; tie lower validation BCE; patience on this lexicographic criterion',
            threshold='maximum inner validation F1; tie closest to .5 then smaller',optimizer='AdamW',
            hyperparameter_search={'lr_candidates':[a.lr],'selection':'single prespecified setting; no post-test tuning'},
            phpr=dict(num_prototypes_per_class=1,lambda_proto=.1,lambda_std=0.,temperature=.2),
            code_sha256={str(p.relative_to(ROOT)):digest(p) for folder in ['src','scripts'] for p in (ROOT/folder).rglob('*.py')},
            original_phpr_sha256=digest(PHPR_SOURCE))
        (out/'config_resolved.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        model = build(a.method,meta['feature_dim'])
        opt = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
        history = []; best=(-float('inf'),-float('inf')); stale=0
        log(f'method={a.method} fold={fold} cpu threads={a.threads} train/val/test={list(map(len,s.values()))}')
        for epoch in range(1,a.epochs+1):
            model.train(); losses=[]
            ids = list(s['train']); random.shuffle(ids)
            # Accumulate a small number of independent variable-size patient bags.
            for offset in range(0,len(ids),a.batch_size):
                opt.zero_grad()
                batch=ids[offset:offset+a.batch_size]
                logits=torch.stack([model(bags[p]['x']) for p in batch])
                y=torch.tensor([bags[p]['y'] for p in batch],dtype=torch.float32)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward(); opt.step()
                losses.append(float(loss.detach()))
            yt,pt,lt=predict(model,bags,s['train'])
            yv,pv,lv=predict(model,bags,s['val'])
            tm,vm=metrics(yt,pt),metrics(yv,pv)
            row=dict(epoch=epoch,train_loss=lt,val_loss=lv,optimization_loss=float(np.mean(losses)),**{'train_'+k:v for k,v in tm.items()},**{'val_'+k:v for k,v in vm.items()})
            history.append(row)
            pd.DataFrame(history).to_csv(out/'train_history.csv',index=False)
            score=(vm['roc_auc'],-lv)
            if score>best:
                best=score; stale=0
                torch.save({'state_dict':model.state_dict(),'epoch':epoch,'val_auc':vm['roc_auc'],'val_loss':lv},out/'best_model.pt')
            else: stale+=1
            if epoch==1 or epoch%10==0: log(f'epoch={epoch} train_loss={lt:.5f} val_loss={lv:.5f} val_auc={vm["roc_auc"]:.5f}')
            if stale>=a.patience: break
        ckpt=torch.load(out/'best_model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
        yv,pv,_=predict(model,bags,s['val'])
        t=float(threshold(yv,pv))
        (out/'val_threshold.json').write_text(json.dumps({'threshold':t,'selection':'inner validation F1 only','epoch':ckpt['epoch'],'val_f1':metrics(yv,pv,t)['f1']},indent=2))
        pd.DataFrame({'patient_id':s['val'],'label':yv,'probability':pv}).to_csv(out/'val_predictions.csv',index=False)
        # The only outer test prediction call. No epoch loop or selection below.
        y,p,_=predict(model,bags,s['test'])
        pred=pd.DataFrame(dict(patient_id=s['test'],fold=fold,label=y,probability=p,prediction=(p>=t).astype(int),threshold=t))
        pred.to_csv(out/'test_patient_predictions.csv',index=False)
        report=metrics(y,p,t)
        report.update(patients=len(y),positive=int(y.sum()),negative=int((y==0).sum()),parameters=sum(v.numel() for v in model.parameters()),best_epoch=ckpt['epoch'],epochs_run=len(history),seconds=time.time()-start,outer_test_evaluations=1)
        (out/'test_metrics.json').write_text(json.dumps(report,indent=2))
        log('complete '+json.dumps(report))
    except Exception:
        log(traceback.format_exc()); raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=list(AVAILABLE)+list(BLOCKED),required=True)
    p.add_argument('--fold',default='all')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight-decay',type=float,default=.0001)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if a.method in BLOCKED: p.error(BLOCKED[a.method])
    torch.set_num_threads(a.threads)
    bags,meta=load_cache()
    smoke=json.loads((ROOT/'reports/smoke_test.json').read_text())
    assert smoke['passed'] and smoke['cache_sha256']==meta['features_sha256']
    assert smoke['methods'][a.method]['passed']
    for name,sha in smoke['code_sha256'].items(): assert digest(ROOT/name)==sha, 'Code changed since smoke test; rerun smoke_test.py'
    folds=sorted({b['fold'] for b in bags.values()}) if a.fold=='all' else [int(a.fold)]
    for fold in folds: train_fold(a,bags,meta,fold)

if __name__=='__main__': main()
