"""CPU execution variant of the unchanged controlled-pooling trainer.

Default mode remains inner-only smoke. No formal CV without new authorization
and a gate bound to this backend's sources and its synthetic tests.
"""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.controlled_pooling_adapter_ema import train as baseline
from experiments.controlled_pooling_adapter_ema.core import sha
from experiments.controlled_pooling_cpu.loaders import LoaderSession,check_environment

BASE_FINGERPRINT=baseline.fingerprint
BASE_RUN_FOLD=baseline.run_fold

def fingerprint():
    result=BASE_FINGERPRINT()
    result.update({path.relative_to(REPO).as_posix():sha(path) for path in sorted(Path(__file__).parent.rglob('*.py'))})
    return result

def run_fold(args,items,group,fold):
    check_environment()
    if args.workers!=4 or args.threads!=2:
        raise ValueError('Validated CPU backend requires train workers=4 and model threads=2')
    args.cpu_execution=dict(version=1,train_workers=4,validation_workers=0,test_workers=0,
        persistent_train_workers=True,epoch_worker_reseeding=True,preserve_sampler_generator=True,
        model_threads=2,torch_version='2.7.0',baseline='602419721d6983a701c59db5e7e753bddddd9d8b')
    session=LoaderSession()
    try:
        with patch.object(baseline,'loader',session.make):
            report=BASE_RUN_FOLD(args,items,group,fold)
    finally:session.close()
    report['cpu_execution']=args.cpu_execution
    folder=Path(args.run_root)/group/f'fold_{fold}'
    target=folder/('smoke_passed.json' if args.mode=='smoke' else 'test_metrics.json')
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    check_environment()
    if not any(arg=='--workers' or arg.startswith('--workers=') for arg in arguments):arguments+=['--workers','4']
    is_formal=('--mode' in arguments and arguments[arguments.index('--mode')+1]=='formal') or '--mode=formal' in arguments
    if not is_formal:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent/'tests'),'-v'],cwd=REPO,check=True)
    # Scoped injection: baseline model, epoch, selection and threshold code stays unchanged.
    with patch.object(baseline,'fingerprint',fingerprint),patch.object(baseline,'run_fold',run_fold),patch.object(sys,'argv',[sys.argv[0]]+arguments):
        baseline.main()

if __name__=='__main__':main()
