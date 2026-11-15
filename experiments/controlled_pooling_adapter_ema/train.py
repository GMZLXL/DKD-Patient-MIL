"""One shared nested image-training entry. No formal execution without new authorization."""
import argparse
import json
import random
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
import sklearn
from torch.utils.data import DataLoader

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.controlled_pooling_adapter_ema.core import (
    legacy,FIXED,GROUPS,ControlledModel,SharedEMAPCAR,make_optimizer,make_scheduler,counts,sha,validate_reference)
sys.path.insert(0,str(REPO/'experiments/aggregation_comparison'))
from src.protocol_v2 import inner_split_without_test_labels
from src.metrics import metrics,threshold

def fingerprint():
    paths=list(Path(__file__).parent.rglob('*.py'))
    paths.extend(REPO/name for name in [
        'scripts_qc/train_effb0_conv_adapter_phpr_ema_pcar_auc.py',
        'experiments/aggregation_comparison/src/protocol_v2.py',
        'experiments/aggregation_comparison/src/metrics.py'])
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(paths)}

def seed_all():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

def read_items(index):
    df=pd.read_csv(index,dtype={'patient_id':str})
    if len(df)!=703 or df.patient_id.nunique()!=205 or set(df.fold)!=set(range(5)) or set(df.label)!={0,1}:
        raise ValueError('Wrong patient/image/fold counts')
    items={}
    for pid,g in df.groupby('patient_id',sort=True):
        if g.label.nunique()!=1 or g.fold.nunique()!=1 or not 1<=len(g)<=9:raise ValueError('Inconsistent bag')
        paths=g.resolved_path.tolist()
        for path,expected in zip(paths,g.sha256):
            if not Path(path).is_file() or sha(path)!=expected:raise ValueError('Image changed or missing')
        items[pid]=dict(patient_id=pid,label=int(g.label.iloc[0]),y=int(g.label.iloc[0]),
            fold=int(g.fold.iloc[0]),paths=paths,num_images=len(g))
    return items

def loader(items,ids,train,workers=0):
    return DataLoader(legacy.PatientWholeImageDataset([items[p] for p in ids],384,train),
        batch_size=4,shuffle=train,num_workers=workers,collate_fn=legacy.patient_collate_fn,
        generator=torch.Generator().manual_seed(42))

def epoch(model,criterion,batches,device,phase,optimizer=None,scaler=None):
    training=phase=='train';model.train(training);criterion.train(training)
    labels=[];raw=[];calibrated=[];ids=[];losses=[];pairs=0;adapter_norms=[];raw_bce_sum=0.
    before={n:b.detach().clone() for n,b in criterion.named_buffers()} if not training else None
    with torch.set_grad_enabled(training):
        for batch in batches:
            x=batch['images'].to(device);index=batch['patient_indices'].to(device);y=batch['labels'].to(device)
            if training:optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda' and FIXED['amp']):
                logits,z,_=model(x,index,len(y))
                loss=criterion(logits,y,z,phase=phase)
                adjusted=criterion.calibrate_logits(logits,z)
            raw_bce_sum+=float(torch.nn.functional.binary_cross_entropy_with_logits(
                logits.detach().float(),y,pos_weight=criterion.bce.pos_weight,reduction='sum'))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite shared objective')
            if training:
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient')
                norm=sum(float(p.grad.abs().sum()) for p in model.encoder.adapters.parameters() if p.grad is not None)
                adapter_norms.append(norm)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(optimizer);scaler.update()
            positives=int((y>=.5).sum());pairs+=positives*(len(y)-positives)
            labels.extend(y.detach().cpu().numpy().astype(int).tolist())
            raw.extend(logits.detach().sigmoid().float().cpu().tolist())
            calibrated.extend(adjusted.detach().sigmoid().float().cpu().tolist())
            ids.extend(batch['patient_ids']);losses.append(float(loss.detach()))
    if before is not None:
        if not all(torch.equal(value,dict(criterion.named_buffers())[name]) for name,value in before.items()):
            raise RuntimeError('EMA mutated during validation/test')
    return dict(patient_ids=ids,y=np.array(labels),p=np.array(raw),calibrated=np.array(calibrated),
        loss=float(np.mean(losses)),raw_bce=raw_bce_sum/len(labels),pairs=pairs,
        adapter_gradient_l1=max(adapter_norms,default=0.))

def restore(path,model,criterion):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model']);criterion.load_state_dict(saved['criterion'])
    return saved

def run_fold(args,items,group,fold):
    out=Path(args.run_root)/group/f'fold_{fold}'
    if out.exists():raise RuntimeError('Refusing existing output fold')
    out.mkdir(parents=True);seed_all();started=time.time()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    split=inner_split_without_test_labels(items,fold,42)
    train_ids,val_ids=split['train'],split['val']
    if args.mode=='smoke':
        train_ids=sorted(p for label in [0,1] for p in [i for i in train_ids if items[i]['y']==label][:2])
        val_ids=sorted(p for label in [0,1] for p in [i for i in val_ids if items[i]['y']==label][:1])
    (out/'split_patient_ids.json').write_text(json.dumps(split,indent=2))
    (out/'smoke_used_patient_ids.json').write_text(json.dumps(dict(train=train_ids,val=val_ids),indent=2))
    model=ControlledModel(group,args.weights).to(device)
    ys=[items[p]['label'] for p in train_ids]
    pos_weight=torch.tensor([ys.count(0)/ys.count(1)],device=device)
    criterion=SharedEMAPCAR(pos_weight).to(device)
    optimizer=make_optimizer(model);scheduler=make_scheduler(optimizer)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and FIXED['amp'])
    train_loader=loader(items,train_ids,True,args.workers);val_loader=loader(items,val_ids,False,args.workers)
    config=dict(historical_reference=vars(args),locked=FIXED,group=group,device=str(device),
        effective_amp=device.type=='cuda' and FIXED['amp'],primary_score='raw sigmoid',
        secondary_score='EMA-calibrated sigmoid only; not used for checkpoint/threshold',
        code_sha256=fingerprint(),weights_sha256=sha(args.weights),index_sha256=sha(args.index),
        environment=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,
            numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        parameters=counts(model),pos_weight=float(pos_weight))
    (out/'config_resolved.json').write_text(json.dumps(config,indent=2))
    best=(-float('inf'),-float('inf'));stale=0;history=[];max_grad=0.
    epochs=2 if args.mode=='smoke' else 100
    for step in range(1,epochs+1):
        train=epoch(model,criterion,train_loader,device,'train',optimizer,scaler)
        val=epoch(model,criterion,val_loader,device,'val')
        vm=metrics(val['y'],val['p'])
        # Tie criterion is RAW validation weighted BCE, not calibrated ranking loss.
        raw_bce=val['raw_bce']
        score=(vm['roc_auc'],-raw_bce)
        scheduler.step(vm['roc_auc']);max_grad=max(max_grad,train['adapter_gradient_l1'])
        history.append(dict(epoch=step,train_loss=train['loss'],val_loss=val['loss'],
            val_raw_bce=raw_bce,val_auc=vm['roc_auc'],training_pairs=train['pairs'],
            adapter_gradient_l1=train['adapter_gradient_l1']))
        if score>best:
            best=score;stale=0
            torch.save(dict(model=model.state_dict(),criterion=criterion.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=step,val_auc=vm['roc_auc'],
                val_raw_bce=raw_bce),out/'best_model.pt')
        else:stale+=1
        pd.DataFrame(history).to_csv(out/'train_history.csv',index=False)
        print(f'{group} fold={fold} {args.mode} epoch={step} adapter_gradient_l1={train["adapter_gradient_l1"]:.6g}',flush=True)
        if stale>=50:break
    if max_grad<=0:raise RuntimeError('No Adapter gradient')
    restore(out/'best_model.pt',model,criterion)
    val=epoch(model,criterion,val_loader,device,'val')
    t=float(threshold(val['y'],val['p']))
    (out/'val_threshold.json').write_text(json.dumps(dict(threshold=t,source='inner val raw F1'),indent=2))
    pd.DataFrame(dict(patient_id=val['patient_ids'],label=val['y'],probability=val['p'],
        calibrated_secondary=val['calibrated'])).to_csv(out/'val_predictions.csv',index=False)
    # A fresh model + fresh empty EMA buffer must reproduce both score versions.
    restored=ControlledModel(group,args.weights).to(device);restored_loss=SharedEMAPCAR(pos_weight).to(device)
    restore(out/'best_model.pt',restored,restored_loss)
    repeated=epoch(restored,restored_loss,val_loader,device,'val')
    np.testing.assert_array_equal(val['p'],repeated['p'])
    np.testing.assert_array_equal(val['calibrated'],repeated['calibrated'])
    report=dict(passed=True,group=group,mode=args.mode,outer_test_evaluations=0,
        code_sha256=fingerprint(),weights_sha256=sha(args.weights),reference_sha256=sha(args.reference),
        index_sha256=sha(args.index),adapter_gradient_l1=max_grad,model_and_ema_roundtrip=True,
        device=str(device),effective_amp=config['effective_amp'],seconds=time.time()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        parameters=counts(model),best_epoch=restore(out/'best_model.pt',model,criterion)['epoch'])
    if args.mode=='formal':
        test=epoch(model,criterion,loader(items,split['test'],False,args.workers),device,'test')
        pd.DataFrame(dict(patient_id=test['patient_ids'],fold=fold,label=test['y'],probability=test['p'],
            prediction=(test['p']>=t).astype(int),threshold=t,
            calibrated_secondary=test['calibrated'])).to_csv(out/'test_patient_predictions.csv',index=False)
        pred=test['p']>=t;y=test['y']==1
        report.update(metrics(test['y'],test['p'],t),outer_test_evaluations=1,
            TP=int((pred&y).sum()),FP=int((pred&~y).sum()),TN=int((~pred&~y).sum()),FN=int((~pred&y).sum()))
    (out/('smoke_passed.json' if args.mode=='smoke' else 'test_metrics.json')).write_text(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=[*GROUPS,'all'],required=True)
    p.add_argument('--index',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--run-root',required=True)
    p.add_argument('--mode',choices=['smoke','formal'],default='smoke')
    p.add_argument('--fold',choices=['0','1','2','3','4','all'],default='0')
    p.add_argument('--formal-authorized',action='store_true');p.add_argument('--gate-root')
    p.add_argument('--workers',type=int);p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if args.mode=='formal' and (not args.formal_authorized or not args.gate_root):
        p.error('Formal CV requires a new explicit authorization and core-group smoke gates')
    if args.mode=='smoke' and args.fold=='all':p.error('Smoke uses one outer fold only')
    if args.workers is None:args.workers=0 if args.mode=='smoke' else 4
    if args.mode=='formal' and args.workers!=4:p.error('Formal workers must match historical configuration: 4')
    validate_reference(json.loads(Path(args.reference).read_text()))
    torch.set_num_threads(args.threads)
    if args.mode=='smoke':
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent/'tests'),'-v'],cwd=REPO,check=True)
    items=read_items(args.index)
    groups=list(GROUPS) if args.group=='all' else [args.group]
    if args.mode=='formal':
        for group in groups:
            gate=json.loads((Path(args.gate_root)/group/'fold_0/smoke_passed.json').read_text())
            if not (gate['passed'] and gate['outer_test_evaluations']==0
                    and gate['group']==group and gate['mode']=='smoke'
                    and gate['code_sha256']==fingerprint() and gate['weights_sha256']==sha(args.weights)
                    and gate['reference_sha256']==sha(args.reference) and gate['index_sha256']==sha(args.index)):
                raise RuntimeError('Missing or stale inner-only smoke gate')
    folds=range(5) if args.fold=='all' else [int(args.fold)]
    for group in groups:
        for fold in folds:run_fold(args,items,group,fold)

if __name__=='__main__':main()
