import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_cache(cache=None):
    cache = Path(cache or ROOT / 'feature_cache/efficientnet_b0_imagenet')
    meta = json.loads((cache/'metadata.json').read_text())
    assert digest(cache/'features.npy') == meta['features_sha256']
    assert digest(cache/'image_index.csv') == meta['index_sha256']
    df = pd.read_csv(cache/'image_index.csv', dtype={'patient_id':str})
    x = np.load(cache/'features.npy', allow_pickle=False)
    assert x.shape == (703,meta['feature_dim']) and np.isfinite(x).all()
    assert np.array_equal(df.feature_row.values,np.arange(len(df)))
    bags = {}
    for pid, g in df.groupby('patient_id',sort=True):
        assert g.label.nunique() == g.fold.nunique() == 1 and 1 <= len(g) <= 9
        bags[pid] = {'x':torch.from_numpy(x[g.feature_row.values].copy()), 'y':int(g.label.iloc[0]), 'fold':int(g.fold.iloc[0])}
    assert len(bags) == 205
    return bags,meta

def split(bags, fold, seed):
    dev = sorted(pid for pid,b in bags.items() if b['fold'] != fold)
    test = sorted(set(bags)-set(dev))
    train,val = train_test_split(dev,test_size=.2,random_state=seed,stratify=[bags[p]['y'] for p in dev])
    result = {'train':sorted(train),'val':sorted(val),'test':test}
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert set(train)|set(val)|set(test) == set(bags)
    for ids in result.values(): assert {bags[p]['y'] for p in ids} == {0,1}
    return result
