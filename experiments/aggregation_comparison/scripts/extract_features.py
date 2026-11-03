"""Deterministic frozen ImageNet features at existing 384 resolution (no new crop)."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from audit_dataset import ROOT, sha256

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--index', type=Path, default=ROOT / 'reports/resolved_image_index.csv')
    p.add_argument('--out', type=Path, default=ROOT / 'feature_cache/efficientnet_b0_imagenet')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    assert json.loads((ROOT / 'reports/data_audit.json').read_text())['passed'], 'Audit must pass'
    if a.out.exists() and any(a.out.iterdir()): raise RuntimeError('Refusing to overwrite existing cache')
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    digest = sha256(a.weights)
    # torchvision published ImageNet checkpoint filename includes this digest prefix.
    assert digest.startswith('7f5810bc'), 'Unverified ImageNet EfficientNet B0 weights'
    model = efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(a.weights, map_location='cpu', weights_only=True), strict=True)
    model.classifier = torch.nn.Identity()
    model.requires_grad_(False).eval()
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
    df = pd.read_csv(a.index, dtype={'patient_id': str})
    feats = []
    start = time.time()
    with torch.inference_mode():
        for offset in range(0, len(df), a.batch_size):
            batch = []
            for row in df.iloc[offset:offset+a.batch_size].itertuples():
                assert sha256(row.resolved_path) == row.sha256, 'Image changed after audit'
                with Image.open(row.resolved_path) as im:
                    assert im.size == (384,384)
                    batch.append(tfm(im.convert('RGB')))
            feats.append(model(torch.stack(batch)).numpy())
            print(f'features {min(offset+a.batch_size,len(df))}/{len(df)} elapsed={time.time()-start:.1f}s', flush=True)
        # Independent repeat of one image checks reproducibility within numeric tolerance.
        with Image.open(df.iloc[0].resolved_path) as im:
            repeat = model(tfm(im.convert('RGB')).unsqueeze(0)).numpy()[0]
    x = np.concatenate(feats)
    assert x.shape == (703,1280) and np.isfinite(x).all()
    np.testing.assert_allclose(x[0], repeat, atol=2e-5, rtol=2e-5)
    np.save(a.out / 'features.npy', x)
    df.to_csv(a.out / 'image_index.csv', index=False)
    meta = dict(encoder='torchvision.efficientnet_b0', source_url=EfficientNet_B0_Weights.IMAGENET1K_V1.url, weights=str(a.weights.resolve()), weights_sha256=digest, preprocessing='Existing 384x384 crop-only; RGB; ToTensor; ImageNet mean/std; no resize/crop/augmentation. Matches existing project eval at 384; differs from torchvision default 224 center crop.', feature_dim=x.shape[1], images=len(df), patients=df.patient_id.nunique(), device='cpu', torch=torch.__version__, extraction_seconds=time.time()-start, extracted_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), source_index_sha256=sha256(a.index), features_sha256=sha256(a.out/'features.npy'), index_sha256=sha256(a.out/'image_index.csv'), repeat_max_abs=float(abs(x[0]-repeat).max()))
    (a.out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))

if __name__ == '__main__': main()
