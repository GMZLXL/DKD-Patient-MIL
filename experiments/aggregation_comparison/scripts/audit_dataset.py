"""Read-only audit; writes a separate resolved index, never changes source CSVs."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def audit(metadata, labels, out):
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metadata, dtype={'patient_id': str})
    errors, warnings, checks = [], [], []
    required = ['patient_id', 'label', 'fold', 'original_path', 'processed_path']
    missing = sorted(set(required) - set(df.columns))
    if missing:
        result = {'passed': False, 'errors': ['Missing columns: ' + str(missing)]}
        (out / 'data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    if df[required].isna().any().any(): errors.append('Null required values')
    grouped = df.groupby('patient_id', sort=True)
    if (grouped.label.nunique() != 1).any(): errors.append('Conflicting patient labels')
    if (grouped.fold.nunique() != 1).any(): errors.append('Patient crosses outer folds')
    if set(df.label.unique()) != {0, 1}: errors.append('Labels must be exactly 0/1')
    if df.fold.nunique() != 5: errors.append('Expected five folds')
    sizes = grouped.size()
    if len(df) != 703 or len(sizes) != 205: errors.append('Expected 703 images and 205 patients')
    if not sizes.between(1, 9).all(): errors.append('Bag size outside 1..9')
    if Path(labels).is_file():
        try:
            lab = pd.read_csv(labels, dtype={'patient_id': str}, encoding='utf-8-sig')
        except UnicodeDecodeError:
            lab = pd.read_csv(labels, dtype={'patient_id': str}, encoding='gb18030')
        if lab.patient_id.duplicated().any(): errors.append('Duplicate label patient IDs')
        merged = df[['patient_id', 'label']].drop_duplicates().merge(lab[['patient_id', 'label']], on='patient_id', how='outer', suffixes=('', '_source'), indicator=True)
        if not ((merged['_merge'] == 'both') & (merged.label == merged.label_source)).all(): errors.append('Label source mismatch')
    else:
        errors.append('Missing independent labels CSV: ' + str(labels))
    records = []
    for row in df.to_dict('records'):
        raw = str(row['processed_path']).replace('\\', '/')
        p = Path(raw)
        if not p.is_file():
            marker = '/DKD_Fundus_MIL/'
            if marker in raw:
                p = PROJECT / raw.split(marker, 1)[1]
        record = dict(row, resolved_path=str(p.resolve()), path_remapped=str(p) != raw)
        try:
            with Image.open(p) as im:
                im.load()
                record.update(width=im.width, height=im.height, mode=im.mode,
                              pixel_sha256=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
            record['sha256'] = sha256(p)
            if (record['width'], record['height']) != (384, 384): errors.append('Non-384 image: ' + str(p))
        except Exception as exc:
            errors.append('Unreadable: ' + str(p))
            checks.append(dict(path=str(p), error=str(exc)))
        records.append(record)
    index = pd.DataFrame(records).sort_values(['patient_id', 'resolved_path']).reset_index(drop=True)
    for col in ['resolved_path', 'original_path', 'sha256', 'pixel_sha256']:
        if col in index:
            dup = index[index[col].duplicated(keep=False)]
            if len(dup):
                errors.append('Duplicate image/source: ' + col)
                dup.to_csv(out / ('duplicates_' + col + '.csv'), index=False)
    index.insert(0, 'feature_row', range(len(index)))
    index.to_csv(out / 'resolved_image_index.csv', index=False)
    patients = df[['patient_id', 'label', 'fold']].drop_duplicates()
    folds = []
    for fold, g in patients.groupby('fold'):
        folds.append(dict(fold=int(fold), patients=len(g), images=int((df.fold == fold).sum()), positive=int(g.label.sum()), negative=int((g.label == 0).sum())))
    pd.DataFrame(folds).to_csv(out / 'fold_audit.csv', index=False)
    result = dict(passed=not errors, metadata=str(Path(metadata).resolve()), metadata_sha256=sha256(metadata), labels=str(Path(labels).resolve()), labels_sha256=sha256(labels) if Path(labels).is_file() else None, patients=len(sizes), images=len(df), fold_ids=sorted(map(int, df.fold.unique())), bag_size_counts={str(k): int(v) for k,v in sizes.value_counts().sort_index().items()}, remapped_paths=int(index.path_remapped.sum()), errors=errors, warnings=warnings, unreadable=checks)
    (out / 'data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', type=Path, default=PROJECT / 'data_qc/whole_image_metadata_384_crop_only.csv')
    p.add_argument('--labels', type=Path, default=PROJECT / 'data_qc/205_labels.csv')
    p.add_argument('--out', type=Path, default=ROOT / 'reports')
    a = p.parse_args()
    result = audit(a.metadata, a.labels, a.out)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
