"""Synthetic-only four-worker equivalence and backend isolation tests."""
import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from experiments.controlled_pooling_cpu.loaders import PersistentTrainLoader
from experiments.controlled_pooling_cpu import train as cpu
from experiments.controlled_pooling_adapter_ema.core import legacy

class RandomDataset(Dataset):
    def __len__(self):return 33
    def __getitem__(self,index):
        return dict(index=index,features=torch.rand(1+index%9,4).sum(0),
            python_value=random.random(),numpy_value=np.random.random(),initial_seed=torch.initial_seed())

class DeterministicDataset(Dataset):
    def __len__(self):return 33
    def __getitem__(self,index):return torch.arange(12).float()+index

def reset():random.seed(42);np.random.seed(42);torch.manual_seed(42)

def equal(left,right):
    if isinstance(left,torch.Tensor):return torch.equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(equal(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left==right

def collect(loader,epochs):
    return [[(copy.deepcopy(batch),torch.nn.functional.dropout(torch.ones(8),p=.5)) for batch in loader] for epoch in range(epochs)]

def trajectory(persistent):
    reset();dataset=RandomDataset();generator=torch.Generator().manual_seed(42)
    model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(8,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    loader=PersistentTrainLoader(dataset) if persistent else DataLoader(dataset,batch_size=4,shuffle=True,
        num_workers=4,generator=generator)
    records=[];pids=[]
    try:
        for epoch in range(3):
            current=[]
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                output=model(batch['features']);loss=output.square().mean();loss.backward();optimizer.step()
                current.append((copy.deepcopy(batch),output.detach().clone(),float(loss)))
            records.append((current,copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict())))
            if persistent:pids.append([worker.pid for worker in loader.loader._iterator._workers])
        end=(torch.get_rng_state().clone(),(loader.generator if persistent else generator).get_state().clone())
        if persistent:
            if not pids[0]==pids[1]==pids[2]:raise AssertionError('Training worker pool restarted')
    finally:
        if persistent:
            workers=list(loader.loader._iterator._workers);loader.close();loader.close()
            if any(worker.is_alive() for worker in workers):raise AssertionError('Worker not shut down')
    return records,end

class CpuLoaderTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_four_worker_order_rng_and_parameter_trajectory(self):
        self.assertTrue(equal(trajectory(False),trajectory(True)))

    def test_validation_four_workers_vs_zero(self):
        outputs=[]
        for workers in [4,0]:
            reset();loader=DataLoader(DeterministicDataset(),batch_size=4,shuffle=False,
                num_workers=workers,generator=torch.Generator().manual_seed(42))
            outputs.append((collect(loader,2),torch.get_rng_state().clone(),random.getstate(),np.random.get_state()))
        self.assertTrue(equal(outputs[0][:3],outputs[1][:3]))
        self.assertTrue(np.array_equal(outputs[0][3][1],outputs[1][3][1]))

    def test_original_transforms_and_k_one_to_nine_with_four_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            items=[]
            for index in range(9):
                paths=[]
                for image_index in range(index+1):
                    path=Path(temporary)/f'artificial_{index}_{image_index}.png'
                    pixels=np.random.default_rng(index*10+image_index).integers(0,256,(24,24,3),dtype=np.uint8)
                    Image.fromarray(pixels).save(path);paths.append(str(path))
                items.append(dict(patient_id=f'artificial_{index}',label=index%2,fold=0,num_images=index+1,paths=paths))
            def dataset():return legacy.PatientWholeImageDataset(items,32,True)
            reset();original=DataLoader(dataset(),batch_size=4,shuffle=True,num_workers=4,
                collate_fn=legacy.patient_collate_fn,generator=torch.Generator().manual_seed(42))
            expected=collect(original,2)
            reset();candidate=PersistentTrainLoader(dataset(),collate_fn=legacy.patient_collate_fn)
            try:self.assertTrue(equal(expected,collect(candidate,2)))
            finally:candidate.close()

    def test_fold_loader_cleanup_and_baseline_injection_restored_on_error(self):
        from argparse import Namespace
        args=Namespace(workers=4,threads=2)
        original_loader=cpu.baseline.loader
        with patch.object(cpu,'BASE_RUN_FOLD',side_effect=RuntimeError('synthetic failure')),patch.object(cpu,'LoaderSession') as session:
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):cpu.run_fold(args,{},'C0',0)
            session.return_value.close.assert_called_once()
        self.assertIs(cpu.baseline.loader,original_loader)
        self.assertIn('experiments/controlled_pooling_cpu/loaders.py',cpu.fingerprint())

    def test_formal_without_new_authorization_and_invalid_worker_settings_rejected(self):
        from argparse import Namespace
        with self.assertRaises(ValueError):cpu.run_fold(Namespace(workers=0,threads=2),{},'C0',0)
        with self.assertRaises(ValueError):cpu.run_fold(Namespace(workers=4,threads=4),{},'C0',0)
        with self.assertRaises(SystemExit):cpu.main(['--group','C0','--mode','formal','--index','unused',
            '--reference','unused','--weights','unused','--run-root','unused'])

if __name__=='__main__':unittest.main()
