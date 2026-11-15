"""Read-only verification of controlled pooling runs; never infer on outer test.

The training implementation and gate fingerprint are deliberately not modified.
All outputs, including aggregate metrics, must remain outside the Git worktree.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics as sk

REPO=Path(__file__).resolve().parents[1]
NAMES=['roc_auc','pr_auc','accuracy','recall','specificity','precision','f1','mcc','balanced_accuracy']

def require(condition,message):
    if not condition:raise RuntimeError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def csv(path):return pd.read_csv(path,dtype={'patient_id':str},float_precision='round_trip')

def training_positive_weight(labels):
    y=np.asarray(labels)
    require(set(y)=={0,1},'Inner training weight requires both classes')
    # The trainer constructs torch.tensor([negative/positive]) in float32, then
    # writes float(tensor) to JSON. Compare to that representation, not float64.
    return float(np.float32((y==0).sum()/(y==1).sum()))

def validation_threshold(y,p):
    y=np.asarray(y);p=np.asarray(p)
    choices=np.unique(np.r_[0.,.5,1.,p])
    return float(max(choices,key=lambda t:(sk.f1_score(y,p>=t,zero_division=0),-abs(t-.5),-t)))

def evaluate(y,p,thresholds):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);t=np.asarray(thresholds,dtype=float)
    require(set(y)=={0,1},'Evaluation requires two classes')
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probabilities')
    require(np.isfinite(t).all(),'Invalid threshold')
    pred=p>=t
    tn,fp,fn,tp=sk.confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=float(sk.roc_auc_score(y,p)),pr_auc=float(sk.average_precision_score(y,p)),
        accuracy=float(sk.accuracy_score(y,pred)),recall=float(sk.recall_score(y,pred,zero_division=0)),
        specificity=float(tn/(tn+fp)),precision=float(sk.precision_score(y,pred,zero_division=0)),
        f1=float(sk.f1_score(y,pred,zero_division=0)),mcc=float(sk.matthews_corrcoef(y,pred)),
        balanced_accuracy=float(sk.balanced_accuracy_score(y,pred)),
        TP=int(tp),FP=int(fp),TN=int(tn),FN=int(fn))

def checkpoint_selection(history,checkpoint):
    require(history.epoch.tolist()==list(range(1,len(history)+1)),'Epoch sequence differs')
    require(1<=len(history)<=100,'Epoch budget differs')
    require(np.isfinite(history[['val_auc','val_raw_bce','adapter_gradient_l1']].values).all(),'Nonfinite history')
    require((history.adapter_gradient_l1>0).all(),'Adapter gradient missing')
    best=(-np.inf,-np.inf);best_epoch=None;stale=0
    for row in history.itertuples():
        require(stale<50,'Training continued after patience exhausted')
        score=(row.val_auc,-row.val_raw_bce)
        if score>best:best=score;best_epoch=row.epoch;stale=0
        else:stale+=1
    require(len(history)==100 or stale==50,'Training stopped without exhausting budget/patience')
    require(checkpoint['epoch']==best_epoch,'Best epoch differs from inner selection')
    require(abs(checkpoint['val_auc']-best[0])<1e-12,'Checkpoint validation AUC differs')
    require(abs(checkpoint['val_raw_bce']+best[1])<1e-12,'Checkpoint validation BCE differs')
    return best_epoch

def prediction_frame(frame,ids,index,fold=None):
    require(frame.patient_id.is_unique and set(frame.patient_id)==set(ids),'Prediction coverage differs')
    expected=index.set_index('patient_id').loc[frame.patient_id]
    require(np.array_equal(frame.label.values,expected.label.values),'Saved labels differ from index')
    require(np.isfinite(frame.probability).all() and frame.probability.between(0,1).all(),'Invalid saved scores')
    if fold is not None:
        require((expected.fold.values==fold).all() and (frame.fold==fold).all(),'Outer fold differs')

def replay_validation(group,checkpoint,config,split,index_path,weights):
    # Only inner validation images are constructed. There is no outer inference path.
    import torch
    sys.path.insert(0,str(REPO))
    from experiments.controlled_pooling_adapter_ema.core import ControlledModel,SharedEMAPCAR,legacy
    from torch.utils.data import DataLoader
    index=csv(index_path)
    items=[]
    for pid in split['val']:
        rows=index[index.patient_id==pid]
        items.append(dict(patient_id=pid,label=int(rows.label.iloc[0]),fold=int(rows.fold.iloc[0]),
            paths=rows.resolved_path.tolist()))
    require(not (set(split['val'])&set(split['test'])),'Outer images in validation replay')
    model=ControlledModel(group,weights).eval()
    criterion=SharedEMAPCAR(torch.tensor([config['pos_weight']])).eval()
    model.load_state_dict(checkpoint['model']);criterion.load_state_dict(checkpoint['criterion'])
    saved={key:value.clone() for key,value in criterion.state_dict().items()}
    loader=DataLoader(legacy.PatientWholeImageDataset(items,384,False),batch_size=4,
        num_workers=0,shuffle=False,collate_fn=legacy.patient_collate_fn)
    records=[];bce=0.
    with torch.inference_mode():
        for batch in loader:
            logits,z,_=model(batch['images'],batch['patient_indices'],len(batch['labels']))
            calibrated=criterion.calibrate_logits(logits,z)
            bce+=float(torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(),batch['labels'],pos_weight=criterion.bce.pos_weight,reduction='sum'))
            for pid,y,p,c in zip(batch['patient_ids'],batch['labels'].tolist(),logits.sigmoid().tolist(),calibrated.sigmoid().tolist()):
                records.append(dict(patient_id=pid,label=int(y),probability=p,calibrated_secondary=c))
    require(all(torch.equal(value,criterion.state_dict()[key]) for key,value in saved.items()),'Validation replay mutated EMA')
    return pd.DataFrame(records),bce/len(items)

def verify(args):
    import torch
    torch.set_num_threads(2)
    root=Path(args.run_root).resolve();out=Path(args.output).resolve()
    require(REPO!=out and REPO not in out.parents,'Private verification output must be outside repository')
    require(not out.exists(),'Verification output already exists')
    manifest=load(args.run_manifest)
    require(root==Path(manifest['formal_root']).resolve(),'Formal root differs from authorized manifest')
    for path,expected in manifest['code_sha256'].items():require(digest(REPO/path)==expected,'Training source changed')
    for path,expected in manifest['protected_sha256'].items():require(digest(path)==expected,'Protected historical file changed')
    for name in ['reference','weights','index']:
        require(digest(manifest[name])==manifest[name+'_sha256'],'Authorized input changed: '+name)
    index=csv(manifest['index'])
    patient_index=index[['patient_id','label','fold']].drop_duplicates()
    require(len(patient_index)==205 and patient_index.patient_id.is_unique,'Index patient count differs')
    rows=[];all_oof=[];checks=[];before={}
    for group in ['C0','C1','C2']:
        group_oof=[]
        for fold in range(5):
            folder=root/group/f'fold_{fold}'
            required=['config_resolved.json','split_patient_ids.json','best_model.pt','train_history.csv',
                'val_predictions.csv','val_threshold.json','test_patient_predictions.csv','test_metrics.json']
            for name in required:
                path=folder/name;require(path.is_file(),'Incomplete formal fold: '+group+'/'+str(fold))
                before[str(path)]=digest(path)
            config=load(folder/'config_resolved.json');split=load(folder/'split_patient_ids.json')
            report=load(folder/'test_metrics.json');saved_t=load(folder/'val_threshold.json')
            h=csv(folder/'train_history.csv');val=csv(folder/'val_predictions.csv');test=csv(folder/'test_patient_predictions.csv')
            require(config['git_sha']==manifest['base_git'],'Training Git base differs')
            require(config['locked']==manifest['locked'],'Fixed configuration differs')
            require(config['code_sha256']==manifest['code_sha256']==report['code_sha256'],'Training fingerprint differs')
            require(config['device']=='cpu' and not config['effective_amp'],'Execution environment differs')
            require(config['historical_reference']['mode']=='formal' and config['historical_reference']['workers']==4,'Execution mode/workers differ')
            require(report['passed'] and report['mode']=='formal' and report['group']==group
                and report['outer_test_evaluations']==1 and report['model_and_ema_roundtrip'],'Formal evidence incomplete')
            require([len(split[key]) for key in ['train','val','test']]==[131,33,41],'Split sizes differ')
            sets=[set(split[key]) for key in ['train','val','test']]
            require(not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'Patient leakage')
            require(set.union(*sets)==set(patient_index.patient_id),'Split patient coverage differs')
            require(sets[2]==set(patient_index[patient_index.fold==fold].patient_id),'Outer split differs')
            # Independently reproduce the predeclared inner split from development labels.
            from sklearn.model_selection import train_test_split
            dev=sorted(sets[0]|sets[1]);labels=patient_index.set_index('patient_id').loc[dev].label.tolist()
            tr,va=train_test_split(dev,test_size=.2,random_state=42,stratify=labels)
            require(set(tr)==sets[0] and set(va)==sets[1],'Inner stratified split differs')
            train_labels=patient_index.set_index('patient_id').loc[split['train']].label.values
            pos_weight=training_positive_weight(train_labels)
            require(abs(config['pos_weight']-pos_weight)<1e-12,'BCE weight used non-training labels')
            checkpoint=torch.load(folder/'best_model.pt',map_location='cpu',weights_only=True)
            best_epoch=checkpoint_selection(h,checkpoint)
            require(best_epoch==report['best_epoch'],'Best epoch report differs')
            require(bool(checkpoint['criterion']['ema_initialized']) and tuple(checkpoint['criterion']['ema_prototypes'].shape)==(2,256),'EMA checkpoint missing')
            require(abs(float(checkpoint['criterion']['bce.pos_weight'])-pos_weight)<1e-6,'Checkpoint BCE weight differs')
            prediction_frame(val,split['val'],patient_index);prediction_frame(test,split['test'],patient_index,fold)
            threshold=validation_threshold(val.label,val.probability)
            require(saved_t['source']=='inner val raw F1' and abs(saved_t['threshold']-threshold)<1e-12,'Threshold is not inner F1 optimum')
            require(np.allclose(test.threshold,threshold,rtol=0,atol=1e-12),'Test threshold changed')
            require(np.array_equal(test.prediction.values,(test.probability.values>=threshold).astype(int)),'Saved test decision differs')
            replay,bce=replay_validation(group,checkpoint,config,split,manifest['index'],manifest['weights'])
            expected=val.set_index('patient_id').sort_index();got=replay.set_index('patient_id').sort_index()
            require(np.array_equal(expected.label,got.label),'Replayed validation labels differ')
            require(np.allclose(expected[['probability','calibrated_secondary']],got[['probability','calibrated_secondary']],atol=1e-7,rtol=0),'Checkpoint fails validation replay')
            require(abs(sk.roc_auc_score(got.label,got.probability)-checkpoint['val_auc'])<1e-12,'Replayed checkpoint AUC differs')
            require(abs(bce-checkpoint['val_raw_bce'])<1e-6,'Replayed checkpoint BCE differs')
            result=evaluate(test.label,test.probability,threshold)
            for key,value in result.items():require(abs(report[key]-value)<1e-12,'Saved test metric differs: '+key)
            rows.append(dict(group=group,fold=fold,best_epoch=best_epoch,epochs_run=len(h),threshold=threshold,**result))
            group_oof.append(test.assign(group=group));checks.append(dict(group=group,fold=fold,passed=True,inner_val_replayed=True,outer_test_model_replays=0))
            print(f'Verified {group} fold {fold}: checkpoint, validation threshold, nine metrics; outer replay=0',flush=True)
        pooled=pd.concat(group_oof,ignore_index=True)
        require(len(pooled)==205 and pooled.patient_id.is_unique,'OOF duplicates/missing patients')
        all_oof.append(pooled)
    for path,expected in before.items():require(digest(path)==expected,'Verification mutated a formal artifact')
    per_fold=pd.DataFrame(rows);oof=pd.concat(all_oof,ignore_index=True)
    reference=oof[oof.group=='C0'].set_index('patient_id')[['label','fold']].sort_index()
    summaries=[];pooled_rows=[]
    for group in ['C0','C1','C2']:
        current=oof[oof.group==group]
        require(reference.equals(current.set_index('patient_id')[['label','fold']].sort_index()),'Cross-group labels/folds differ')
        folds=per_fold[per_fold.group==group]
        row=dict(group=group)
        for name in NAMES:
            row[name+'_mean']=float(folds[name].mean());row[name+'_std']=float(folds[name].std(ddof=1))
        summaries.append(row)
        pooled_rows.append(dict(group=group,patients=205,**evaluate(current.label,current.probability,current.threshold)))
    out.mkdir(parents=True)
    per_fold.to_csv(out/'controlled_pooling_per_fold.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'controlled_pooling_final.csv',index=False)
    pd.DataFrame(pooled_rows).to_csv(out/'controlled_pooling_pooled_oof.csv',index=False)
    oof.to_csv(out/'controlled_pooling_oof_private.csv',index=False)
    (out/'verification.json').write_text(json.dumps(dict(passed=True,formal_folds=15,groups=3,
        patients_per_group=205,inner_val_replays=15,outer_test_model_replays=0,checks=checks,
        source_sha256=digest(__file__),protected_files_unchanged=len(manifest['protected_sha256'])),indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    return per_fold,pd.DataFrame(summaries),pd.DataFrame(pooled_rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True);parser.add_argument('--run-manifest',required=True)
    parser.add_argument('--output',required=True)
    verify(parser.parse_args())

if __name__=='__main__':main()
