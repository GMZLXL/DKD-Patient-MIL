"""Persistent training workers with the original per-epoch RNG trajectory.

Pinned to the audited PyTorch 2.7.0 map-style loader implementation. The shared
loader generator feeds both worker base seeds and RandomSampler. A plain change
to persistent_workers would alter that consumption and the sampling trajectory.
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,RandomSampler,get_worker_info
from torch.utils.data._utils.worker import _generate_state

def check_environment():
    if str(torch.__version__).split('+')[0]!='2.7.0' or torch.cuda.is_available():
        raise RuntimeError('CPU loader backend requires revalidation outside PyTorch 2.7.0 CPU')

class SeededDataset(Dataset):
    def __init__(self,dataset):self.dataset=dataset;self.token=None
    def __len__(self):return len(self.dataset)
    def __getitem__(self,payload):
        epoch,base_seed,index=payload
        token=(epoch,base_seed)
        if token!=self.token:
            worker=get_worker_info()
            if worker is None:raise RuntimeError('Seeded training dataset requires worker isolation')
            random.seed(base_seed+worker.id);torch.manual_seed(base_seed+worker.id)
            np.random.seed(_generate_state(base_seed,worker.id))
            self.token=token
        return self.dataset[index]

class SeededSampler:
    def __init__(self,sampler):self.sampler=sampler;self.epoch=0;self.base_seed=0
    def __len__(self):return len(self.sampler)
    def __iter__(self):
        for index in self.sampler:yield self.epoch,self.base_seed,index

class PersistentTrainLoader:
    def __init__(self,dataset,batch_size=4,num_workers=4,collate_fn=None):
        check_environment()
        if num_workers<1:raise ValueError('Persistent training requires worker processes')
        self.generator=torch.Generator().manual_seed(42)
        self.sampler=SeededSampler(RandomSampler(dataset,generator=self.generator))
        self.loader=DataLoader(SeededDataset(dataset),batch_size=batch_size,
            sampler=self.sampler,num_workers=num_workers,collate_fn=collate_fn,
            generator=self.generator,persistent_workers=True)
        self.started=False;self.closed=False
    def __len__(self):return len(self.loader)
    def __iter__(self):
        if self.closed:raise RuntimeError('Loader is closed')
        self.sampler.epoch+=1
        if self.started:
            base=int(torch.empty((),dtype=torch.int64).random_(generator=self.generator))
        else:
            preview=torch.Generator();preview.set_state(self.generator.get_state())
            base=int(torch.empty((),dtype=torch.int64).random_(generator=preview))
        self.sampler.base_seed=base
        iterator=iter(self.loader);self.started=True
        return iterator
    def close(self):
        if self.closed:return
        iterator=getattr(self.loader,'_iterator',None)
        if iterator is not None:iterator._shutdown_workers()
        self.closed=True

class LoaderSession:
    """One fold owns one training pool; validation/test never create another pool."""
    def __init__(self):self.active=[]
    def make(self,items,ids,train,workers=4):
        from experiments.controlled_pooling_adapter_ema.core import legacy
        dataset=legacy.PatientWholeImageDataset([items[pid] for pid in ids],384,train)
        if train:
            if workers!=4:raise ValueError('Validated CPU execution uses four training workers')
            loader=PersistentTrainLoader(dataset,num_workers=workers,collate_fn=legacy.patient_collate_fn)
            self.active.append(loader);return loader
        return DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0,
            collate_fn=legacy.patient_collate_fn,generator=torch.Generator().manual_seed(42))
    def close(self):
        for loader in self.active:loader.close()
        self.active.clear()
