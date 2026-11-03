#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_auto(path):
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法读取CSV编码: {path}")


class ImageDataset(Dataset):
    def __init__(self, df, image_size=224):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row["processed_path"])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "patient_id": str(row["patient_id"]),
            "label": int(row["label"]),
            "fold": int(row["fold"]),
            "path": path,
        }


def build_retfound_encoder(retfound_root, weight_path, device):
    retfound_root = str(Path(retfound_root).resolve())
    if retfound_root not in sys.path:
        sys.path.insert(0, retfound_root)

    import models_vit
    from util.pos_embed import interpolate_pos_embed

    # 使用CLS特征。global_pool=False可加载MAE预训练的norm参数，
    # 避免冻结特征提取时使用随机初始化的fc_norm。
    model = models_vit.RETFound_mae(
        img_size=224,
        num_classes=2,
        drop_path_rate=0.0,
        global_pool=False,
    )

    checkpoint = torch.load(
    weight_path,
    map_location="cpu",
    weights_only=False,
    )
    checkpoint_model = copy.deepcopy(checkpoint["model"])
    model_state = model.state_dict()

    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model:
            if (
                key not in model_state
                or checkpoint_model[key].shape != model_state[key].shape
            ):
                del checkpoint_model[key]

    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)

    allowed_missing = {"head.weight", "head.bias"}
    unexpected_encoder = [
        key for key in msg.unexpected_keys
        if not (
            key == "mask_token"
            or key.startswith("decoder_")
        )
    ]
    unexpected_missing = [
        key for key in msg.missing_keys
        if key not in allowed_missing
    ]

    if unexpected_missing:
        raise RuntimeError(f"存在异常Missing keys: {unexpected_missing}")
    if unexpected_encoder:
        raise RuntimeError(f"存在异常Unexpected keys: {unexpected_encoder}")

    feature_dim = int(model.head.in_features)
    model.head = nn.Identity()

    for parameter in model.parameters():
        parameter.requires_grad = False

    model = model.to(device)
    model.eval()

    print("===== RETFound encoder loaded =====")
    print(f"RETFound root: {retfound_root}")
    print(f"Weight path: {weight_path}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected MAE decoder keys: {len(msg.unexpected_keys)}")

    return model, feature_dim


def extract_image_features(
    df,
    image_size,
    batch_size,
    num_workers,
    device,
    retfound_root,
    weight_path,
):
    dataset = ImageDataset(df, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model, feature_dim = build_retfound_encoder(
        retfound_root=retfound_root,
        weight_path=weight_path,
        device=device,
    )

    all_records = []
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting RETFound image features"):
            images = batch["image"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                feats = model.forward_features(images)

            if feats.ndim != 2 or feats.shape[1] != feature_dim:
                raise RuntimeError(
                    f"RETFound特征形状异常: {tuple(feats.shape)}, "
                    f"期望 [B, {feature_dim}]"
                )

            feats = feats.float().cpu().numpy()

            for i in range(len(feats)):
                all_records.append({
                    "patient_id": batch["patient_id"][i],
                    "label": int(batch["label"][i]),
                    "fold": int(batch["fold"][i]),
                    "processed_path": batch["path"][i],
                    "feature": feats[i],
                })

    return all_records, feature_dim


def build_patient_features(records):
    patient_dict = {}

    for record in records:
        pid = str(record["patient_id"])
        if pid not in patient_dict:
            patient_dict[pid] = {
                "patient_id": pid,
                "label": int(record["label"]),
                "fold": int(record["fold"]),
                "features": [],
            }

        patient_dict[pid]["features"].append(record["feature"])

    rows = []
    for pid, item in patient_dict.items():
        image_features = np.stack(item["features"], axis=0)
        patient_feature = image_features.mean(axis=0)

        rows.append({
            "patient_id": pid,
            "label": item["label"],
            "fold": item["fold"],
            "num_images": len(item["features"]),
            "feature": patient_feature,
        })

    return rows


class PatientFeatureDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = torch.tensor(row["feature"], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y, row["patient_id"]


class LinearPatientClassifier(nn.Module):
    def __init__(self, in_dim=1024, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x):
        return self.classifier(x).squeeze(1)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {}

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")

    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = float("nan")

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    out["specificity"] = float(tn / (tn + fp + 1e-8))
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    out["tp"] = int(tp)

    return out


def train_one_fold(
    fold,
    patient_rows,
    feature_dim,
    out_dir,
    epochs,
    patience,
    batch_size,
    lr,
    weight_decay,
    device,
):
    train_rows = [r for r in patient_rows if int(r["fold"]) != fold]
    val_rows = [r for r in patient_rows if int(r["fold"]) == fold]

    train_loader = DataLoader(
        PatientFeatureDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PatientFeatureDataset(val_rows),
        batch_size=batch_size,
        shuffle=False,
    )

    model = LinearPatientClassifier(
        in_dim=feature_dim,
        dropout=0.1,
    ).to(device)

    train_labels = np.array([r["label"] for r in train_rows])
    num_pos = max(int((train_labels == 1).sum()), 1)
    num_neg = max(int((train_labels == 0).sum()), 1)
    pos_weight = torch.tensor(
        [num_neg / num_pos],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_auc = -1.0
    best_state = None
    bad_epochs = 0

    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        y_true, y_prob = [], []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                logits = model(x)
                prob = torch.sigmoid(logits).cpu().numpy()

                y_prob.extend(prob.tolist())
                y_true.extend(y.numpy().astype(int).tolist())

        metrics = compute_metrics(y_true, y_prob)
        auc = metrics["roc_auc"]

        print(
            f"Fold {fold} | Epoch {epoch:03d} | "
            f"loss={np.mean(train_losses):.4f} | "
            f"auc={metrics['roc_auc']:.4f} | "
            f"pr_auc={metrics['pr_auc']:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"spec={metrics['specificity']:.4f}"
        )

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Fold {fold}: early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true, y_prob, pids = [], [], []

    with torch.no_grad():
        for x, y, pid in val_loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()

            y_prob.extend(prob.tolist())
            y_true.extend(y.numpy().astype(int).tolist())
            pids.extend(list(pid))

    metrics = compute_metrics(y_true, y_prob)

    pred_df = pd.DataFrame({
        "patient_id": pids,
        "label": y_true,
        "prob": y_prob,
        "pred": (np.asarray(y_prob) >= 0.5).astype(int),
    })
    pred_df.to_csv(
        fold_dir / "patient_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    torch.save(model.state_dict(), fold_dir / "best_classifier.pt")
    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--retfound-root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        required=True,
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patient-batch-size", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.image_size != 224:
        raise ValueError(
            "当前RETFound MAE权重对应224×224输入，请使用 --image-size 224"
        )

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.retfound_root).exists():
        raise FileNotFoundError(
            f"RETFound项目目录不存在: {args.retfound_root}"
        )
    if not Path(args.weight_path).exists():
        raise FileNotFoundError(
            f"RETFound权重不存在: {args.weight_path}"
        )

    df = read_csv_auto(args.metadata)

    required = ["patient_id", "label", "fold", "processed_path"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"metadata缺少必要列: {column}")

    for path in df["processed_path"].astype(str):
        if not Path(path).exists():
            raise FileNotFoundError(f"图像不存在: {path}")

    print("===== Metadata summary =====")
    print(f"Images: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print("Patient labels:")
    print(
        df.drop_duplicates("patient_id")["label"]
        .value_counts()
        .sort_index()
    )
    print("Patients per fold:")
    print(
        df.drop_duplicates("patient_id")["fold"]
        .value_counts()
        .sort_index()
    )

    feature_cache = out_dir / "image_features_retfound_mae_frozen.pt"
    feature_dim = 1024

    if feature_cache.exists():
        print(f"Loading cached features: {feature_cache}")
        cache = torch.load(feature_cache, weights_only=False)

        if isinstance(cache, dict) and "records" in cache:
            records = cache["records"]
            feature_dim = int(cache.get("feature_dim", 1024))
        else:
            records = cache
    else:
        records, feature_dim = extract_image_features(
            df=df,
            image_size=args.image_size,
            batch_size=args.extract_batch_size,
            num_workers=args.num_workers,
            device=device,
            retfound_root=args.retfound_root,
            weight_path=args.weight_path,
        )

        torch.save(
            {
                "records": records,
                "feature_dim": feature_dim,
                "model": "RETFound_mae",
                "pooling": "CLS token",
                "image_size": args.image_size,
            },
            feature_cache,
        )
        print(f"Saved image feature cache to: {feature_cache}")

    patient_rows = build_patient_features(records)

    fold_metrics = []
    folds = sorted(df["fold"].unique().tolist())

    for fold in folds:
        print(f"\n========== Fold {fold} ==========")
        metrics = train_one_fold(
            fold=int(fold),
            patient_rows=patient_rows,
            feature_dim=feature_dim,
            out_dir=out_dir,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.patient_batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        metrics["fold"] = int(fold)
        fold_metrics.append(metrics)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(
        out_dir / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {}
    metric_names = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "f1",
    ]

    for name in metric_names:
        summary[name] = {
            "mean": float(metrics_df[name].mean()),
            "std": float(metrics_df[name].std()),
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Five-fold summary =====")
    print(metrics_df)
    print("\nMean ± std:")
    for name, value in summary.items():
        print(
            f"{name}: {value['mean']:.4f} ± {value['std']:.4f}"
        )

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
