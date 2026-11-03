"""Nested training for DGR-MIL-adapted and ACMIL-GA-adapted only.

Never invokes an upstream trainer or reruns the completed baselines.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.adapted_models import build
from src.patient_bags import load_cache,split,digest
from src.metrics import metrics,threshold

def code_fingerprint():
    paths=list((ROOT/'src').rglob('*.py'))+[Path(__file__)]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def predict(model,bags,ids):
    model.eval()
    with torch.no_grad():
        logits=torch.stack([model(bags[p]['x'])['logits'] for p in ids])
        y=torch.tensor([bags[p]['y'] for p in ids],dtype=torch.long)
        loss=float(torch.nn.functional.cross_entropy(logits,y))
        p=logits.softmax(-1)[:,1].numpy()
    return y.numpy(),p,loss

def train_epoch(model,bags,ids,opt,epoch,batch_size):
    model.train();ids=list(ids);random.shuffle(ids);total=[];parts={}
    for start in range(0,len(ids),batch_size):
        opt.zero_grad();batch=ids[start:start+batch_size];losses=[]
        for pid in batch:
            output=model(bags[pid]['x'])
            loss,diagnostics=model.objective(output,bags[pid]['y'],epoch=epoch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            losses.append(loss)
            for k,v in diagnostics.items():
                parts.setdefault(k,[]).append(float(v.detach()) if torch.is_tensor(v) else float(v))
        loss=torch.stack(losses).mean();loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
        opt.step();total.append(float(loss.detach()))
    return float(np.mean(total)),{k:float(np.mean(v)) for k,v in parts.items()}

def subset_balanced(bags,ids,n_each):
    return sorted(p for y in [0,1] for p in [pid for pid in ids if bags[pid]['y']==y][:n_each])

def fit_fold(a,bags,meta,fold):
    out=Path(a.run_root)/a.method/f'fold_{fold}'
    if out.exists():raise RuntimeError('Refusing to overwrite '+str(out))
    out.mkdir(parents=True);start=time.time();seed_all(a.seed)
    def log(message):
        print(message,flush=True)
        with (out/'run.log').open('a',encoding='utf-8') as f:f.write(message+'\n')
    s=split(bags,fold,a.seed)
    (out/'split_patient_ids.json').write_text(json.dumps(s,indent=2))
    train=s['train'];val=s['val']
    if a.smoke:train=subset_balanced(bags,train,4);val=subset_balanced(bags,val,2)
    model=build(a.method,meta['feature_dim'])
    cfg=dict(vars(a),fold=fold,device='cpu',torch=str(torch.__version__),model=model.config,
        cache=meta,code_sha256=code_fingerprint(),train_patients=len(train),val_patients=len(val),
        selection='max inner validation AUC, tie lower validation CE',
        threshold='max inner validation F1, tie closest .5 then smaller',
        optimizer='AdamW',search='single prespecified lr=.001; no post-test tuning')
    (out/'config_resolved.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    history=[];best=(-float('inf'),-float('inf'));stale=0
    log(f'{a.method} fold={fold} smoke={a.smoke} train={len(train)} val={len(val)}')
    for epoch in range(1,a.epochs+1):
        objective,components=train_epoch(model,bags,train,opt,epoch,a.batch_size)
        yt,pt,lt=predict(model,bags,train);yv,pv,lv=predict(model,bags,val)
        tm,vm=metrics(yt,pt),metrics(yv,pv)
        row=dict(epoch=epoch,train_loss=lt,val_loss=lv,optimization_loss=objective,
            **{'objective_'+k:v for k,v in components.items()},
            **{'train_'+k:v for k,v in tm.items()},**{'val_'+k:v for k,v in vm.items()})
        history.append(row);pd.DataFrame(history).to_csv(out/'train_history.csv',index=False)
        score=(vm['roc_auc'],-lv)
        if score>best:
            best=score;stale=0
            torch.save({'state_dict':model.state_dict(),'epoch':epoch,'val_auc':vm['roc_auc'],'val_loss':lv},out/'best_model.pt')
        else:stale+=1
        if epoch==1 or epoch%10==0:log(f'epoch={epoch} val_auc={vm["roc_auc"]:.5f} objective={objective:.5f}')
        if stale>=a.patience:break
    checkpoint=torch.load(out/'best_model.pt',weights_only=True,map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    yv,pv,_=predict(model,bags,val);t=float(threshold(yv,pv))
    (out/'val_threshold.json').write_text(json.dumps(dict(threshold=t,selection='inner validation F1 only',epoch=checkpoint['epoch']),indent=2))
    pd.DataFrame(dict(patient_id=val,label=yv,probability=pv)).to_csv(out/'val_predictions.csv',index=False)
    # Roundtrip is checked on validation patients; no extra outer test forward.
    reloaded=build(a.method,meta['feature_dim']);reloaded.load_state_dict(checkpoint['state_dict'])
    _,again,_=predict(reloaded,bags,val)
    np.testing.assert_array_equal(pv,again)
    if a.smoke:
        report=dict(passed=True,method=a.method,checkpoint_roundtrip=True,outer_test_evaluations=0,
            code_sha256=code_fingerprint(),cache_sha256=meta['features_sha256'],seed=a.seed,
            models_tested=True,seconds=time.time()-start)
        (out/'smoke_passed.json').write_text(json.dumps(report,indent=2))
        log('smoke passed; outer test not evaluated')
        return report
    # The only outer test prediction call, after checkpoint AND threshold selection.
    y,p,_=predict(model,bags,s['test'])
    pred=pd.DataFrame(dict(patient_id=s['test'],fold=fold,label=y,probability=p,prediction=(p>=t).astype(int),threshold=t))
    pred.to_csv(out/'test_patient_predictions.csv',index=False)
    report=metrics(y,p,t)
    report.update(patients=len(y),positive=int(y.sum()),negative=int((y==0).sum()),
        parameters=sum(p.numel() for p in model.parameters()),best_epoch=checkpoint['epoch'],
        epochs_run=len(history),seconds=time.time()-start,outer_test_evaluations=1,
        method_variant=a.method+'-adapted')
    (out/'test_metrics.json').write_text(json.dumps(report,indent=2));log('complete '+json.dumps(report))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['dgr_mil','acmil'],required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--run-root',required=True)
    p.add_argument('--fold',default='all')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--gate',help='Single-fold smoke_passed.json required for formal training')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight-decay',type=float,default=.0001)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    bags,meta=load_cache(a.cache)
    if a.smoke:
        # Tests are part of the executable gate, never a user-supplied pass flag.
        import subprocess
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT.parents[1]/'tests'),'-v'],cwd=ROOT.parents[1],check=True)
        if a.fold=='all':p.error('Smoke test requires exactly one fold')
        a.epochs=2
    else:
        if not a.gate:p.error('--gate is required for formal training')
        gate=json.loads(Path(a.gate).read_text())
        if not (gate['passed'] and gate['method']==a.method and gate['code_sha256']==code_fingerprint() and gate['cache_sha256']==meta['features_sha256'] and gate['seed']==a.seed):p.error('Smoke gate is stale or mismatched')
    folds=sorted({b['fold'] for b in bags.values()}) if a.fold=='all' else [int(a.fold)]
    for fold in folds:
        try:fit_fold(a,bags,meta,fold)
        except Exception:
            import traceback
            failure=Path(a.run_root)/a.method/f'fold_{fold}'/'failure.log'
            if failure.parent.is_dir():failure.write_text(traceback.format_exc())
            raise

if __name__=='__main__':main()
