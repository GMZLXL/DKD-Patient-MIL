"""Read-only v2 qualification/OOF verification and patient-paired statistics.

Only validation patients may be replayed through a model. Outer test metrics are
recomputed from saved predictions; no outer test model forward is performed.
All output paths must be local and outside the source repository.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_adapted_cv_v2 as training

NAMES=['roc_auc','pr_auc','accuracy','recall','specificity','precision','f1','mcc','balanced_accuracy']
LABELS={'mean':'Mean','gated_abmil':'Gated-ABMIL','phpr':'PHPR-p1',
 'dgr_mil':'DGR-MIL-adapted (warmup-selected)',
 'acmil':'ACMIL-GA-adapted (conservative-mask)',
 'dgr_fullobj':'DGR-MIL-adapted-fullobj-v2',
 'acmil_official_ratio':'ACMIL-GA-adapted-v2-official-mask'}
REQUIRED=['config_resolved.yaml','split_patient_ids.json','train_history.csv',
 'checkpoint_selection.csv','best_model.pt','val_predictions.csv','val_threshold.json',
 'test_patient_predictions.csv','test_metrics.json','run.log']

def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def close(a,b,message,atol=1e-10):
    require(np.isfinite(float(a)) and np.isfinite(float(b)) and abs(float(a)-float(b))<=atol,message)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def metrics_independent(y,p,pred):
    y=np.asarray(y,dtype=int);p=np.asarray(p);pred=np.asarray(pred,dtype=int)
    tp=int(((y==1)&(pred==1)).sum());tn=int(((y==0)&(pred==0)).sum())
    fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum())
    recall=tp/(tp+fn) if tp+fn else 0.;spec=tn/(tn+fp) if tn+fp else 0.
    precision=tp/(tp+fp) if tp+fp else 0.
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    denominator=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
       accuracy=(tp+tn)/len(y),recall=recall,specificity=spec,precision=precision,f1=f1,
       mcc=(tp*tn-fp*fn)/denominator if denominator else 0.,balanced_accuracy=(recall+spec)/2)

def threshold_independent(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p)
    candidates=sorted({0.,.5,1.,*map(float,p)})
    def rank(t):
        prediction=p>=t;tp=int(((y==1)&prediction).sum())
        fp=int(((y==0)&prediction).sum());fn=int(((y==1)&~prediction).sum())
        denominator=2*tp+fp+fn
        return (2*tp/denominator if denominator else 0.,-abs(t-.5),-t)
    return max(candidates,key=rank)

def validate_selection(history,evidence,checkpoint,method,patience=20):
    require(len(history)==len(evidence)>0,'History/evidence length mismatch')
    require(history.epoch.tolist()==list(range(1,len(history)+1)),'Missing/repeated epochs')
    require(history.epoch.tolist()==evidence.epoch.tolist(),'Evidence epoch mismatch')
    require(not any('test' in c for c in history.columns),'Outer test metric in history')
    require(np.isfinite(history.select_dtypes(include='number').to_numpy()).all(),'Nonfinite history')
    start=10 if method=='dgr_fullobj' else 1
    expected_eligible=evidence.epoch>=start
    require(np.array_equal(evidence.checkpoint_eligible,expected_eligible),'Wrong eligibility')
    best=(-math.inf,-math.inf);best_epoch=None;stale=0
    for i,row in evidence.iterrows():
        epoch=int(row.epoch)
        close(row.val_auc,history.iloc[i].val_roc_auc,'History AUC mismatch')
        close(row.val_ce,history.iloc[i].val_loss,'History CE mismatch')
        save=False
        if epoch>=start:
            score=(float(row.val_auc),-float(row.val_ce))
            if score>best:best=score;best_epoch=epoch;stale=0;save=True
            else:stale+=1
        require(bool(row.save_best)==save,'Incorrect checkpoint save decision')
        require(int(row.eligible_stale)==stale,'Warmup/patience counting error')
        require(bool(row.stop)==(epoch>=start and stale>=patience),'Incorrect early stopping evidence')
        if i<len(evidence)-1:require(not bool(row.stop),'Training continued after early stopping')
        if method=='dgr_fullobj':
            active=epoch>=10
            require(bool(row.auxiliary_active)==active,'Missing/early auxiliary objective')
            close(history.iloc[i].objective_auxiliary_active,float(active),'Objective activation mismatch')
            require((int(row.auxiliary_optimizer_steps)>0)==active,'Missing actual auxiliary updates')
            require(int(row.auxiliary_start_epoch)==10 and int(row.checkpoint_eligible_epoch)==10,'DGR protocol mismatch')
    require(best_epoch==int(checkpoint['epoch']),'Checkpoint not validation-best eligible epoch')
    close(checkpoint['val_auc'],best[0],'Checkpoint AUC differs from evidence')
    close(checkpoint['val_loss'],-best[1],'Checkpoint CE differs from evidence')
    if method=='dgr_fullobj':
        require(checkpoint['epoch']>=10 and checkpoint['auxiliary_active_at_best_checkpoint'] is True,'Warmup checkpoint is invalid for v2')
        require(checkpoint['auxiliary_optimizer_steps']>0,'Checkpoint lacks auxiliary update evidence')
        selected=evidence[evidence.epoch==best_epoch].iloc[0]
        require(int(selected.auxiliary_optimizer_steps)==checkpoint['auxiliary_optimizer_steps'],'Checkpoint update count mismatch')
    return best_epoch

def verify_fold(method,fold,folder,baseline,bags,preflight):
    for name in REQUIRED:require((folder/name).is_file(),'Missing '+str(folder/name))
    config=yaml.safe_load((folder/'config_resolved.yaml').read_text())
    require(config['method']==method and config['mode']=='formal' and config['formal_authorized'],'Wrong method/mode')
    require(config['variant']==LABELS[method],'Wrong variant')
    for key,value in dict(seed=42,epochs=100,patience=20,lr=.001,weight_decay=.0001,batch_size=16).items():
        require(config[key]==value,'Changed fixed budget: '+key)
    require(config['code_sha256']==preflight['training_code_sha256'],'Training source/test fingerprint mismatch')
    require(config['cache']['features_sha256']==preflight['metadata']['features_sha256'],'Wrong cache')
    require(config['cache']['index_sha256']==preflight['metadata']['index_sha256'],'Wrong image index')
    splits=read_json(folder/'split_patient_ids.json')
    require(splits==read_json(baseline/f'fold_{fold}/split_patient_ids.json'),'Different baseline patient split')
    require(splits==training.inner_split_without_test_labels(bags,fold,42),'Different current patient split')
    require(not(set(splits['train'])&set(splits['val']) or set(splits['train'])&set(splits['test']) or set(splits['val'])&set(splits['test'])),'Patient leakage')
    checkpoint=torch.load(folder/'best_model.pt',map_location='cpu',weights_only=True)
    history=pd.read_csv(folder/'train_history.csv');evidence=pd.read_csv(folder/'checkpoint_selection.csv')
    best_epoch=validate_selection(history,evidence,checkpoint,method)
    updates_per_epoch=math.ceil(len(splits['train'])/16)
    if method=='dgr_fullobj':
        np.testing.assert_array_equal(evidence.auxiliary_optimizer_steps,np.maximum(evidence.epoch.to_numpy()-9,0)*updates_per_epoch)
        require(config['checkpoint_eligible_epoch']==10 and config['auxiliary_start_epoch']==10,'Wrong eligibility config')
        require(config['warmup_counts_toward_patience'] is False,'Warmup patience enabled')
    model=training.build_v2(method,preflight['metadata']['feature_dim'])
    require(config['model']==model.config,'Unexpected architecture/masking/dropout configuration')
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    parameters=sum(p.numel() for p in model.parameters())
    val=pd.read_csv(folder/'val_predictions.csv',dtype={'patient_id':str})
    require(val.patient_id.tolist()==splits['val'] and not val.patient_id.duplicated().any(),'Invalid validation patients')
    require(val.label.tolist()==[bags[p]['y'] for p in splits['val']],'Validation label mismatch')
    # Only inner validation inference is replayed, never outer test inference.
    yv,pv,validation_ce=training.predict(model,bags,splits['val'])
    np.testing.assert_allclose(val.probability.to_numpy(),pv,rtol=0,atol=1e-7)
    close(roc_auc_score(yv,pv),checkpoint['val_auc'],'Validation AUC replay mismatch')
    close(validation_ce,checkpoint['val_loss'],'Validation CE replay mismatch',atol=2e-6)
    reconstructed_ce=float(-np.log(np.where(yv==1,pv.astype(float),1-pv.astype(float))).mean())
    close(reconstructed_ce,checkpoint['val_loss'],'Saved probability CE mismatch',atol=2e-6)
    chosen=read_json(folder/'val_threshold.json')
    threshold=threshold_independent(val.label.to_numpy(),val.probability.to_numpy(dtype=np.float32))
    close(threshold,chosen['threshold'],'Validation threshold mismatch',atol=1e-7)
    require(chosen['best_epoch']==best_epoch,'Threshold from wrong checkpoint')
    prediction=pd.read_csv(folder/'test_patient_predictions.csv',dtype={'patient_id':str})
    require(prediction.patient_id.tolist()==splits['test'] and not prediction.patient_id.duplicated().any(),'Invalid test patients')
    require(prediction.label.tolist()==[bags[p]['y'] for p in splits['test']],'Test label mismatch')
    require((prediction.fold==fold).all(),'Wrong fold IDs')
    np.testing.assert_allclose(prediction.threshold,chosen['threshold'],rtol=0,atol=1e-12)
    probs=prediction.probability.to_numpy(dtype=np.float32)
    require(np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all(),'Invalid probability')
    np.testing.assert_array_equal(prediction.prediction,(probs>=np.float32(chosen['threshold'])).astype(int))
    measured=metrics_independent(prediction.label,probs,prediction.prediction)
    report=read_json(folder/'test_metrics.json')
    require(report['outer_test_evaluations']==1 and report['mode']=='formal','Incorrect test evaluation record')
    require(report['method']==method and report['variant']==LABELS[method],'Incorrect report variant')
    require(report['best_epoch']==best_epoch and report['epochs_run']==len(history),'Report selection mismatch')
    for name,value in measured.items():close(value,report[name],'Test metric mismatch: '+name)
    if method=='dgr_fullobj':
        require(report['auxiliary_active_at_best_checkpoint'] is True and report['auxiliary_optimizer_steps_at_best_checkpoint']==checkpoint['auxiliary_optimizer_steps'],'Auxiliary report mismatch')
    return dict(method=method,variant=LABELS[method],fold=fold,parameters=parameters,
                best_epoch=best_epoch,epochs_run=len(history),**measured),prediction

def legacy_records(preflight):
    records=[];frames={}
    for method in ['mean','gated_abmil','phpr','dgr_mil','acmil']:
        root=Path(preflight['baseline_root']) if method in ['mean','gated_abmil','phpr'] else Path(preflight['legacy_root'])
        pieces=[]
        for fold in range(5):
            folder=root/method/f'fold_{fold}'
            frame=pd.read_csv(folder/'test_patient_predictions.csv',dtype={'patient_id':str})
            stats=metrics_independent(frame.label,frame.probability.to_numpy(dtype=np.float32),frame.prediction)
            saved=read_json(folder/'test_metrics.json')
            for k,v in stats.items():close(v,saved[k],'Legacy metric mismatch: '+method+'/'+k)
            records.append(dict(method=method,variant=LABELS[method],fold=fold,
                parameters=saved['parameters'],best_epoch=saved['best_epoch'],epochs_run=saved['epochs_run'],**stats))
            pieces.append(frame)
        frames[method]=pd.concat(pieces,ignore_index=True).sort_values('patient_id').reset_index(drop=True)
    return records,frames

def write_statistics(records,frames,out,repeats=2000,seed=42):
    summaries=[]
    for method in LABELS:
        subset=[r for r in records if r['method']==method]
        summaries.append(dict(method=method,variant=LABELS[method],
            **{k+'_mean':float(np.mean([r[k] for r in subset])) for k in NAMES},
            **{k+'_std':float(np.std([r[k] for r in subset],ddof=1)) for k in NAMES}))
    table=pd.DataFrame(summaries)
    original=['mean','gated_abmil','phpr','dgr_mil','acmil']
    sensitivity=['mean','gated_abmil','phpr','dgr_fullobj','acmil_official_ratio']
    table[table.method.isin(original)].to_csv(out/'aggregation_comparison_original_5methods.csv',index=False)
    table[table.method.isin(sensitivity)].assign(analysis_type='post-hoc exploratory sensitivity analysis').to_csv(out/'aggregation_comparison_v2_sensitivity.csv',index=False)
    reference=frames['mean'][['patient_id','label','fold']]
    for method,frame in frames.items():
        require(len(frame)==205 and frame.patient_id.nunique()==205,'Incomplete OOF '+method)
        pd.testing.assert_frame_equal(frame[['patient_id','label','fold']],reference)
    y=reference.label.to_numpy();rng=np.random.default_rng(seed)
    sampled=[];skipped=0
    for _ in range(repeats):
        indices=rng.integers(0,len(y),len(y))
        if len(np.unique(y[indices]))<2:skipped+=1
        else:sampled.append(indices)
    points={};distribution={};pooled=[]
    for method,frame in frames.items():
        p=frame.probability.to_numpy(dtype=np.float32);pred=frame.prediction.to_numpy()
        point=metrics_independent(y,p,pred);points[method]=point
        boot=np.array([[v for v in metrics_independent(y[i],p[i],pred[i]).values()] for i in sampled])
        distribution[method]=boot
        row=dict(method=method,variant=LABELS[method],patients=205,**point)
        for j,k in enumerate(NAMES):
            row[k+'_ci_low'],row[k+'_ci_high']=map(float,np.quantile(boot[:,j],[.025,.975]))
        pooled.append(row)
    pd.DataFrame(pooled).to_csv(out/'pooled_oof_metrics.csv',index=False)
    pairs=[('dgr_fullobj','dgr_mil'),('acmil_official_ratio','acmil')]
    pairs += [(new,base) for new in ['dgr_fullobj','acmil_official_ratio'] for base in ['mean','gated_abmil','phpr']]
    contrasts=[]
    for new,old in pairs:
        differences=distribution[new]-distribution[old]
        for j,k in enumerate(NAMES):
            low,high=np.quantile(differences[:,j],[.025,.975])
            contrasts.append(dict(method=new,reference=old,metric=k,pooled_delta=points[new][k]-points[old][k],
                ci_low=float(low),ci_high=float(high),unit='paired patients',seed=seed,
                attempted=repeats,valid=len(sampled),skipped_single_class=skipped))
    pd.DataFrame(contrasts).to_csv(out/'paired_comparison_bootstrap.csv',index=False)
    return dict(unit='patient',method='percentile bootstrap, fixed OOF predictions and fold thresholds',
        seed=seed,attempted=repeats,valid=len(sampled),skipped_single_class=skipped,
        includes_retraining_uncertainty=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--bootstrap-repeats',type=int,default=2000)
    args=parser.parse_args();preflight=read_json(args.preflight)
    out=Path(preflight['report_root']).resolve()
    require(not out.is_relative_to(REPO.resolve()),'Private reports must be outside Git repository')
    require(not out.exists(),'Refusing to overwrite reports');out.mkdir(parents=True)
    torch.set_num_threads(2)
    bags,metadata=training.load_cache(preflight['cache'])
    require(metadata==preflight['metadata'],'Cache metadata changed')
    require(training.code_fingerprint()==preflight['training_code_sha256'],'Training source/test changed')
    project=REPO.parents[1]
    for name,value in preflight['historical_sha256'].items():require(sha(project/name)==value,'Historical artifact changed: '+name)
    for name,value in preflight['old_source_sha256'].items():require(sha(REPO/name)==value,'Historical source changed: '+name)
    records,frames=legacy_records(preflight);new_predictions=[]
    for method in ['dgr_fullobj','acmil_official_ratio']:
        parts=[]
        for fold in range(5):
            row,frame=verify_fold(method,fold,Path(preflight['formal_root'])/method/f'fold_{fold}',
                                  Path(preflight['baseline_root'])/'mean',bags,preflight)
            records.append(row);parts.append(frame)
        frames[method]=pd.concat(parts,ignore_index=True).sort_values('patient_id').reset_index(drop=True)
        new_predictions.append(frames[method].assign(method=method,variant=LABELS[method]))
    pd.DataFrame([r for r in records if r['method'] in ['dgr_fullobj','acmil_official_ratio']]).to_csv(out/'v2_per_fold_metrics.csv',index=False)
    pd.DataFrame(records).to_csv(out/'all_per_fold_metrics.csv',index=False)
    pd.concat(new_predictions,ignore_index=True).to_csv(out/'v2_oof_predictions.csv',index=False)
    bootstrap=write_statistics(records,frames,out,args.bootstrap_repeats)
    result=dict(passed=True,new_folds=10,new_oof_rows=410,patients_per_v2_method=205,
        legacy_folds=25,total_folds=35,historical_artifacts_unchanged=len(preflight['historical_sha256']),
        historical_source_files_unchanged=len(preflight['old_source_sha256']),splits_identical=True,
        checkpoints_eligible=True,thresholds_verified=True,nine_metrics_recomputed=True,
        validation_checkpoint_inference_replays=10,verification_outer_test_model_evaluations=0,bootstrap=bootstrap)
    (out/'v2_verification.json').write_text(json.dumps(result,indent=2))
    manifest=dict(preflight,verified_folds=records,bootstrap=bootstrap,verification=result,
        verifier_sha256=sha(Path(__file__)),environment=dict(torch=str(torch.__version__),python=sys.version),
        run_artifacts_sha256={str(p):sha(p) for p in Path(preflight['formal_root']).rglob('*') if p.is_file()})
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
