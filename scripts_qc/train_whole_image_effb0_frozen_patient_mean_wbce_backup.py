#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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
    def __init__(self, df, image_size=384):
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
        path = row["processed_path"]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "patient_id": str(row["patient_id"]),
            "label": int(row["label"]),
            "fold": int(row["fold"]),
            "path": path,
        }


def extract_image_features(df, image_size, batch_size, device):
    dataset = ImageDataset(df, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()

    all_records = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting EfficientNet-B0 image features"):
            images = batch["image"].to(device)
            feats = model(images)
            feats = feats.detach().cpu().numpy()

            for i in range(len(feats)):
                all_records.append({
                    "patient_id": batch["patient_id"][i],
                    "label": int(batch["label"][i]),
                    "fold": int(batch["fold"][i]),
                    "processed_path": batch["path"][i],
                    "feature": feats[i],
                })

    return all_records


def build_patient_features(records):
    patient_dict = {}

    for r in records:
        pid = str(r["patient_id"])
        if pid not in patient_dict:
            patient_dict[pid] = {
                "patient_id": pid,
                "label": int(r["label"]),
                "fold": int(r["fold"]),
                "features": [],
                "paths": [],
            }

        patient_dict[pid]["features"].append(r["feature"])
        patient_dict[pid]["paths"].append(r["processed_path"])

    rows = []
    for pid, item in patient_dict.items():
        feats = np.stack(item["features"], axis=0)
        patient_feat = feats.mean(axis=0)

        rows.append({
            "patient_id": pid,
            "label": item["label"],
            "fold": item["fold"],
            "num_images": len(item["features"]),
            "feature": patient_feat,
        })

    return rows


class PatientFeatureDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = torch.tensor(r["feature"], dtype=torch.float32)
        y = torch.tensor(r["label"], dtype=torch.float32)
        return x, y, r["patient_id"]


class LinearPatientClassifier(nn.Module):
    def __init__(self, in_dim=1280, dropout=0.1):
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

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["specificity"] = float(tn / (tn + fp + 1e-8))
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    out["tp"] = int(tp)

    return out


def train_one_fold(
    fold,
    patient_rows,
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

    train_dataset = PatientFeatureDataset(train_rows)
    val_dataset = PatientFeatureDataset(val_rows)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LinearPatientClassifier(in_dim=1280, dropout=0.1).to(device)

    train_labels = np.array([r["label"] for r in train_rows])
    num_pos = max((train_labels == 1).sum(), 1)
    num_neg = max((train_labels == 0).sum(), 1)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_auc = -1
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
        y_true, y_prob, pids = [], [], []

        with torch.no_grad():
            for x, y, pid in val_loader:
                x = x.to(device)
                logits = model(x)
                prob = torch.sigmoid(logits).detach().cpu().numpy()

                y_prob.extend(prob.tolist())
                y_true.extend(y.numpy().astype(int).tolist())
                pids.extend(list(pid))

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
            best_state = model.state_dict()
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
            prob = torch.sigmoid(logits).detach().cpu().numpy()

            y_prob.extend(prob.tolist())
            y_true.extend(y.numpy().astype(int).tolist())
            pids.extend(list(pid))

    metrics = compute_metrics(y_true, y_prob)

    pred_df = pd.DataFrame({
        "patient_id": pids,
        "label": y_true,
        "prob": y_prob,
        "pred": (np.array(y_prob) >= 0.5).astype(int),
    })
    pred_df.to_csv(fold_dir / "patient_predictions.csv", index=False, encoding="utf-8-sig")

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    torch.save(model.state_dict(), fold_dir / "best_classifier.pt")

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--patient-batch-size", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_auto(args.metadata)

    required = ["patient_id", "label", "fold", "processed_path"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"metadata 缺少必要列: {c}")

    for p in df["processed_path"].astype(str).tolist():
        if not Path(p).exists():
            raise FileNotFoundError(f"图像不存在: {p}")

    print("===== Metadata summary =====")
    print(f"Images: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print("Patient labels:")
    print(df.drop_duplicates("patient_id")["label"].value_counts().sort_index())
    print("Patients per fold:")
    print(df.drop_duplicates("patient_id")["fold"].value_counts().sort_index())

    feature_cache = out_dir / "image_features_effb0_frozen.pt"

    if feature_cache.exists():
        print(f"Loading cached features: {feature_cache}")
        records = torch.load(feature_cache, weights_only=False)
    else:
        records = extract_image_features(
            df=df,
            image_size=args.image_size,
            batch_size=args.extract_batch_size,
            device=device,
        )
        torch.save(records, feature_cache)
        print(f"Saved image feature cache to: {feature_cache}")

    patient_rows = build_patient_features(records)

    fold_metrics = []
    folds = sorted(df["fold"].unique().tolist())

    for fold in folds:
        print(f"\n========== Fold {fold} ==========")
        metrics = train_one_fold(
            fold=int(fold),
            patient_rows=patient_rows,
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
    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {}
    for col in ["roc_auc", "pr_auc", "accuracy", "recall", "specificity", "precision", "f1"]:
        summary[col] = {
            "mean": float(metrics_df[col].mean()),
            "std": float(metrics_df[col].std()),
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Five-fold summary =====")
    print(metrics_df)
    print("\nMean ± std:")
    for k, v in summary.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()