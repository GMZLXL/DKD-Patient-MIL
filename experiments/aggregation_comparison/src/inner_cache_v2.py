"""Read the existing cache for inner-only smoke; do not construct outer test bags."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .patient_bags import digest


def load_inner_cache(cache, fold):
    cache = Path(cache)
    metadata = json.loads((cache/'metadata.json').read_text())
    if digest(cache/'features.npy') != metadata['features_sha256']:
        raise ValueError('Feature cache checksum mismatch')
    if digest(cache/'image_index.csv') != metadata['index_sha256']:
        raise ValueError('Feature index checksum mismatch')
    index = pd.read_csv(cache/'image_index.csv', dtype={'patient_id': str})
    # Memory map avoids materializing test vectors; hashing above checks bytes only.
    features = np.load(cache/'features.npy', allow_pickle=False, mmap_mode='r')
    if features.shape != (703, metadata['feature_dim']) or len(index) != 703:
        raise ValueError('Unexpected cache dimensions')
    if not np.array_equal(index.feature_row.values, np.arange(len(index))):
        raise ValueError('Feature index alignment mismatch')
    bags = {}
    for pid, group in index.groupby('patient_id', sort=True):
        if group.fold.nunique() != 1 or not 1 <= len(group) <= 9:
            raise ValueError('Invalid bag fold or size')
        bag_fold = int(group.fold.iloc[0])
        if bag_fold == fold:
            bags[pid] = {'fold': bag_fold}  # no test labels, tensors, or model input
            continue
        labels = pd.to_numeric(group.label, errors='raise')
        if labels.nunique() != 1 or labels.iloc[0] not in {0, 1}:
            raise ValueError('Invalid development labels')
        values = np.array(features[group.feature_row.values], copy=True)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite development features')
        bags[pid] = dict(fold=bag_fold, y=int(labels.iloc[0]), x=torch.from_numpy(values))
    if len(bags) != 205:
        raise ValueError('Expected 205 patients')
    return bags, metadata
