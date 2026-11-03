"""Versioned, post-hoc sensitivity protocols. Default action is INNER-ONLY smoke.

Formal CV is never launched by smoke and requires a separate explicit operator
authorization flag plus a current smoke gate. Historical trainers are unchanged.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_adapted_cv import predict, seed_all, train_epoch, subset_balanced
from src.adapted_models.acmil_official_ratio import ACMILOfficialRatio
from src.adapted_models.dgr_mil import DGRMILPatient
from src.patient_bags import digest, load_cache
from src.inner_cache_v2 import load_inner_cache
from src.metrics import metrics, threshold
from src.protocol_v2 import METHODS, inner_split_without_test_labels, policy_for


def dropout_audit(model):
    return dict(
        projection_dropout_modules=[n for n, m in model.projection.named_modules()
                                    if isinstance(m, (nn.Dropout, nn.Dropout1d))],
        output_dropout={n: float(m.p) for n, m in model.named_modules()
                        if isinstance(m, (nn.Dropout, nn.Dropout1d))},
        cross_attention_probability_dropout=float(model.cross.dropout))


def build_v2(method, d):
    if method == 'acmil_official_ratio':
        return ACMILOfficialRatio(input_dim=d)
    if method == 'dgr_fullobj':
        model = DGRMILPatient(input_dim=d)
        model.config = dict(model.config, variant=METHODS[method],
                            dropout_actual=dropout_audit(model))
        return model
    raise ValueError('Unknown versioned method')


def code_fingerprint():
    files = list((ROOT/'src').rglob('*.py'))
    files += [Path(__file__), Path(__file__).with_name('train_adapted_cv.py')]
    files += list((REPO/'tests').glob('*.py'))
    return {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(files)}


def fit_fold_v2(args, bags, metadata, fold):
    """No outer test features or labels are used unless mode explicitly is formal."""
    if args.mode not in {'smoke', 'formal'}:
        raise ValueError('Unknown run mode')
    if args.mode == 'formal' and not args.formal_authorized:
        raise RuntimeError('Formal training requires separate explicit authorization')
    policy = policy_for(args.method, args.patience)
    if args.epochs < policy.checkpoint_eligible_epoch:
        raise ValueError('Run ends before checkpoint eligibility; no qualification smoke possible')
    out = Path(args.run_root)/args.method/f'fold_{fold}'
    if out.exists():
        raise RuntimeError('Refusing to overwrite an existing run directory')
    out.mkdir(parents=True)
    started = time.time()
    def log(message):
        print(message, flush=True)
        with (out/'run.log').open('a', encoding='utf-8') as f:
            f.write(message+'\n')
    seed_all(args.seed)
    splits = inner_split_without_test_labels(bags, fold, args.seed)
    (out/'split_patient_ids.json').write_text(json.dumps(splits, indent=2))
    train_ids, val_ids = splits['train'], splits['val']
    if args.mode == 'smoke':
        train_ids = subset_balanced(bags, train_ids, 4)
        val_ids = subset_balanced(bags, val_ids, 2)
    model = build_v2(args.method, metadata['feature_dim'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resolved = dict(vars(args), fold=fold, variant=METHODS[args.method],
        analysis_type='post-hoc exploratory sensitivity analysis',
        auxiliary_start_epoch=policy.auxiliary_start_epoch,
        checkpoint_eligible_epoch=policy.checkpoint_eligible_epoch,
        warmup_counts_toward_patience=False, optimizer='AdamW', device='cpu',
        torch=str(torch.__version__), model=model.config, cache=metadata,
        selection='eligible inner val AUC maximum, tie inner val CE minimum',
        threshold='inner val F1 maximum, tie closest .5 then smaller',
        code_sha256=code_fingerprint())
    (out/'config_resolved.yaml').write_text(yaml.safe_dump(resolved, allow_unicode=True), encoding='utf-8')
    log(f'{METHODS[args.method]} mode={args.mode} fold={fold} '
        f'auxiliary_start_epoch={policy.auxiliary_start_epoch} '
        f'checkpoint_eligible_epoch={policy.checkpoint_eligible_epoch}')
    history, evidence, auxiliary_steps = [], [], 0
    for epoch in range(1, args.epochs+1):
        objective, components = train_epoch(model, bags, train_ids, optimizer, epoch, args.batch_size)
        auxiliary_active = components.get('auxiliary_active', 0.) == 1.
        if args.method == 'dgr_fullobj':
            if auxiliary_active != (epoch >= policy.auxiliary_start_epoch):
                raise RuntimeError('Model objective and eligibility protocol disagree')
            if auxiliary_active:
                auxiliary_steps += math.ceil(len(train_ids)/args.batch_size)
        y_val, p_val, val_ce = predict(model, bags, val_ids)
        val_metrics = metrics(y_val, p_val)
        decision = policy.observe(epoch, val_metrics['roc_auc'], val_ce,
                                  auxiliary_active=auxiliary_active,
                                  auxiliary_optimizer_steps=auxiliary_steps)
        history.append(dict(epoch=epoch, optimization_loss=objective, val_loss=val_ce,
            **{'objective_'+k: v for k, v in components.items()},
            **{'val_'+k: v for k, v in val_metrics.items()}))
        evidence.append(dict(decision, val_auc=val_metrics['roc_auc'], val_ce=val_ce))
        pd.DataFrame(history).to_csv(out/'train_history.csv', index=False)
        pd.DataFrame(evidence).to_csv(out/'checkpoint_selection.csv', index=False)
        if decision['save_best']:
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch,
                val_auc=val_metrics['roc_auc'], val_loss=val_ce,
                auxiliary_start_epoch=policy.auxiliary_start_epoch,
                checkpoint_eligible_epoch=policy.checkpoint_eligible_epoch,
                auxiliary_active_at_best_checkpoint=auxiliary_active,
                auxiliary_optimizer_steps=auxiliary_steps), out/'best_model.pt')
        log(f'epoch={epoch} eligible={decision["checkpoint_eligible"]} '
            f'eligible_stale={policy.stale} auxiliary_updates={auxiliary_steps} '
            f'save_best={decision["save_best"]}')
        if decision['stop']:
            break
    if policy.best_epoch is None:
        raise RuntimeError('No qualified checkpoint was trained')
    checkpoint = torch.load(out/'best_model.pt', map_location='cpu', weights_only=True)
    policy.validate_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    y_val, p_val, _ = predict(model, bags, val_ids)
    selected_threshold = float(threshold(y_val, p_val))
    (out/'val_threshold.json').write_text(json.dumps(dict(
        threshold=selected_threshold, selection='inner validation F1 only',
        best_epoch=checkpoint['epoch']), indent=2))
    pd.DataFrame(dict(patient_id=val_ids, label=y_val, probability=p_val)).to_csv(out/'val_predictions.csv', index=False)
    reloaded = build_v2(args.method, metadata['feature_dim'])
    reloaded.load_state_dict(checkpoint['state_dict'])
    _, restored_probabilities, _ = predict(reloaded, bags, val_ids)
    np.testing.assert_array_equal(p_val, restored_probabilities)
    report = dict(passed=True, method=args.method, variant=METHODS[args.method], mode=args.mode,
        analysis_type='post-hoc exploratory sensitivity analysis',
        auxiliary_start_epoch=policy.auxiliary_start_epoch,
        checkpoint_eligible_epoch=policy.checkpoint_eligible_epoch,
        best_epoch=checkpoint['epoch'],
        auxiliary_active_at_best_checkpoint=checkpoint['auxiliary_active_at_best_checkpoint'],
        auxiliary_optimizer_steps_at_best_checkpoint=checkpoint['auxiliary_optimizer_steps'],
        epochs_run=len(history), checkpoint_roundtrip=True, outer_test_evaluations=0,
        code_sha256=code_fingerprint(), cache_sha256=metadata['features_sha256'],
        seed=args.seed, seconds=time.time()-started)
    if args.mode == 'smoke':
        (out/'smoke_passed.json').write_text(json.dumps(report, indent=2))
        log('inner-only smoke passed; outer_test_evaluations=0')
        return report
    # Future, separately authorized formal execution only. No loop or selection below.
    y_test, p_test, _ = predict(model, bags, splits['test'])
    pd.DataFrame(dict(patient_id=splits['test'], fold=fold, label=y_test,
        probability=p_test, prediction=(p_test>=selected_threshold).astype(int),
        threshold=selected_threshold)).to_csv(out/'test_patient_predictions.csv', index=False)
    report.update(metrics(y_test, p_test, selected_threshold), outer_test_evaluations=1)
    (out/'test_metrics.json').write_text(json.dumps(report, indent=2))
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=list(METHODS), required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--run-root', required=True)
    p.add_argument('--mode', choices=['smoke', 'formal'], default='smoke')
    p.add_argument('--fold', default='0')
    p.add_argument('--smoke-epochs', type=int)
    p.add_argument('--formal-authorized', action='store_true',
                   help='Operator acknowledgment of separate user authorization; never implied by smoke')
    p.add_argument('--gate', help='Current v2 smoke_passed.json required for formal execution')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args(argv)
    if args.mode == 'smoke':
        if args.fold == 'all':
            p.error('Smoke requires exactly one outer fold')
        args.epochs = args.smoke_epochs if args.smoke_epochs is not None else (10 if args.method=='dgr_fullobj' else 2)
        if args.epochs < (10 if args.method=='dgr_fullobj' else 1):
            p.error('DGR qualification smoke requires at least 10 epochs')
    else:
        if not args.formal_authorized or not args.gate:
            p.error('Formal CV requires separate authorization AND a current smoke gate')
        if args.smoke_epochs is not None:
            p.error('--smoke-epochs is only valid in smoke mode')
        args.epochs = 100
    args.seed, args.patience, args.lr, args.weight_decay, args.batch_size = 42, 20, .001, .0001, 16
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=REPO, check=True)
    if args.mode == 'smoke':
        bags, metadata = load_inner_cache(args.cache, int(args.fold))
    else:
        bags, metadata = load_cache(args.cache)
    if args.mode == 'formal':
        gate = json.loads(Path(args.gate).read_text())
        if not (gate['passed'] and gate['mode']=='smoke' and gate['method']==args.method
                and gate['code_sha256']==code_fingerprint()
                and gate['cache_sha256']==metadata['features_sha256'] and gate['seed']==42):
            raise RuntimeError('Mismatched or stale v2 smoke gate')
        if args.method == 'dgr_fullobj' and not (
                gate['best_epoch']>=10 and gate['auxiliary_active_at_best_checkpoint']
                and gate['auxiliary_optimizer_steps_at_best_checkpoint']>0):
            raise RuntimeError('DGR smoke has not verified full-objective checkpoint eligibility')
    folds = sorted({bag['fold'] for bag in bags.values()}) if args.fold=='all' else [int(args.fold)]
    for fold in folds:
        fit_fold_v2(args, bags, metadata, fold)


if __name__ == '__main__':
    main()
