#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_csv_auto(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot read CSV with common encodings: {path}")


def safe_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp + 1e-8)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = (recall + specificity) / 2.0
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "roc_auc": safe_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "accuracy": float(acc),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


class PatientWholeImageDataset(Dataset):
    def __init__(self, patient_items: List[Dict[str, Any]], image_size: int = 384, train: bool = True):
        self.patient_items = patient_items

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8, fill=0),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.03, hue=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.patient_items)

    def __getitem__(self, idx):
        item = self.patient_items[idx]
        images = []
        for p in item["paths"]:
            img = Image.open(p).convert("RGB")
            images.append(self.transform(img))
        images = torch.stack(images, dim=0)
        return {
            "patient_id": item["patient_id"],
            "label": torch.tensor(item["label"], dtype=torch.float32),
            "fold": int(item["fold"]),
            "images": images,
            "num_images": len(item["paths"]),
        }


def patient_collate_fn(batch):
    all_images = []
    patient_indices = []
    labels = []
    patient_ids = []
    num_images = []

    for patient_idx, item in enumerate(batch):
        imgs = item["images"]
        n = imgs.shape[0]
        all_images.append(imgs)
        patient_indices.extend([patient_idx] * n)
        labels.append(item["label"])
        patient_ids.append(item["patient_id"])
        num_images.append(item["num_images"])

    return {
        "images": torch.cat(all_images, dim=0),
        "patient_indices": torch.tensor(patient_indices, dtype=torch.long),
        "labels": torch.stack(labels, dim=0),
        "patient_ids": patient_ids,
        "num_images": num_images,
    }


def build_patient_items(df: pd.DataFrame) -> List[Dict[str, Any]]:
    items = []
    for pid, g in df.groupby("patient_id"):
        labels = g["label"].unique()
        folds = g["fold"].unique()

        if len(labels) != 1:
            raise ValueError(f"Patient {pid} has multiple labels: {labels}")
        if len(folds) != 1:
            raise ValueError(f"Patient {pid} appears in multiple folds: {folds}")

        paths = g["processed_path"].astype(str).tolist()
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Image not found: {p}")

        items.append({
            "patient_id": str(pid),
            "label": int(labels[0]),
            "fold": int(folds[0]),
            "paths": paths,
            "num_images": len(paths),
        })
    return items


class ConvAdapter(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, alpha: float = 0.1, dropout: float = 0.1):
        super().__init__()
        bottleneck = max(channels // reduction, 8)
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            bottleneck,
            bottleneck,
            kernel_size=3,
            padding=1,
            groups=bottleneck,
            bias=False,
        )
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
        self.alpha = float(alpha)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        residual = self.down(x)
        residual = self.act(residual)
        residual = self.dwconv(residual)
        residual = self.act(residual)
        residual = self.dropout(residual)
        residual = self.up(residual)
        return x + self.alpha * residual


class EfficientNetB0ConvAdapterEncoder(nn.Module):
    def __init__(
        self,
        adapter_indices: str = "7,8",
        adapter_reduction: int = 8,
        adapter_alpha: float = 0.1,
        adapter_dropout: float = 0.1,
        freeze_bn: bool = True,
    ):
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        backbone.classifier = nn.Identity()

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_dim = 1280
        self.freeze_bn_flag = freeze_bn

        channel_map = {
            0: 32,
            1: 16,
            2: 24,
            3: 40,
            4: 80,
            5: 112,
            6: 192,
            7: 320,
            8: 1280,
        }

        self.adapter_indices = [int(x) for x in adapter_indices.split(",") if x.strip() != ""]
        for idx in self.adapter_indices:
            if idx not in channel_map:
                raise ValueError(f"Unsupported adapter index: {idx}. Choose from 0 to 8.")

        for p in self.features.parameters():
            p.requires_grad = False

        self.adapters = nn.ModuleDict()
        for idx in self.adapter_indices:
            self.adapters[str(idx)] = ConvAdapter(
                channels=channel_map[idx],
                reduction=adapter_reduction,
                alpha=adapter_alpha,
                dropout=adapter_dropout,
            )

    def freeze_batchnorm_layers(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn_flag:
            self.freeze_batchnorm_layers()
        return self

    def forward(self, x):
        for idx, block in enumerate(self.features):
            x = block(x)
            key = str(idx)
            if key in self.adapters:
                x = self.adapters[key](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class PatientPoolingHead(nn.Module):
    """Patient-level pooling head.

    Supported pooling modes:
      - mean: stable mean pooling over all image embeddings of one patient.
      - gated_attention: gated attention pooling over image embeddings.
      - phpr: Prototype-guided Heterogeneity-aware Patient Risk pooling.

    PHPR keeps patient mean as the main branch and adds two residual corrections:
      1) a heterogeneity residual from patient-level standard deviation;
      2) a prototype residual from learnable low/high risk prototypes.
    The residual projections are zero-initialized so PHPR starts close to mean pooling.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.1,
        patient_pooling: str = "mean",
        num_prototypes_per_class: int = 2,
        lambda_proto: float = 0.25,
        lambda_std: float = 0.25,
        proto_temperature: float = 0.2,
    ):
        super().__init__()
        self.patient_pooling = patient_pooling
        self.num_prototypes_per_class = int(num_prototypes_per_class)
        self.lambda_proto = float(lambda_proto)
        self.lambda_std = float(lambda_std)
        self.proto_temperature = float(proto_temperature)

        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if patient_pooling == "gated_attention":
            self.att_v = nn.Linear(hidden_dim, attention_dim)
            self.att_u = nn.Linear(hidden_dim, attention_dim)
            self.att_w = nn.Linear(attention_dim, 1)
        else:
            self.att_v = None
            self.att_u = None
            self.att_w = None

        if patient_pooling == "phpr":
            if self.num_prototypes_per_class < 1:
                raise ValueError("num_prototypes_per_class must be >= 1 for PHPR.")
            if self.proto_temperature <= 0:
                raise ValueError("proto_temperature must be > 0 for PHPR.")

            num_prototypes = 2 * self.num_prototypes_per_class
            self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
            nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

            self.std_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proto_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
            self._zero_init_residual_projection(self.std_proj)
            self._zero_init_residual_projection(self.proto_proj)
        elif patient_pooling in ["mean", "gated_attention"]:
            self.prototypes = None
            self.std_proj = None
            self.proto_proj = None
            self.fusion_norm = None
        else:
            raise ValueError(f"Unsupported patient_pooling: {patient_pooling}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _zero_init_residual_projection(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _phpr_pool(self, h_i: torch.Tensor):
        z_mean = h_i.mean(dim=0)
        if h_i.shape[0] > 1:
            z_std = h_i.std(dim=0, unbiased=False)
        else:
            z_std = torch.zeros_like(z_mean)

        query = nn.functional.normalize(z_mean, dim=0)
        proto_bank = nn.functional.normalize(self.prototypes, dim=1)
        proto_scores = torch.matmul(proto_bank, query) / self.proto_temperature
        proto_weights = torch.softmax(proto_scores, dim=0)
        z_proto = torch.sum(self.prototypes * proto_weights.unsqueeze(-1), dim=0)

        z_patient = self.fusion_norm(
            z_mean
            + self.lambda_std * self.std_proj(z_std)
            + self.lambda_proto * self.proto_proj(z_proto)
        )
        return z_patient, proto_weights

    def forward(self, image_features, patient_indices, num_patients: int):
        h = self.proj(image_features.float())

        if self.patient_pooling == "gated_attention":
            raw_scores = self.att_w(
                torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))
            ).squeeze(-1)
        else:
            raw_scores = None

        logits = []
        patient_features = []
        attention_weights = []

        for pid in range(num_patients):
            mask = patient_indices == pid
            h_i = h[mask]

            if self.patient_pooling == "mean":
                z_i = h_i.mean(dim=0)
                aux_i = torch.ones(h_i.shape[0], device=h_i.device) / max(h_i.shape[0], 1)
            elif self.patient_pooling == "gated_attention":
                s_i = raw_scores[mask]
                aux_i = torch.softmax(s_i, dim=0)
                z_i = torch.sum(h_i * aux_i.unsqueeze(-1), dim=0)
            elif self.patient_pooling == "phpr":
                z_i, aux_i = self._phpr_pool(h_i)
            else:
                raise RuntimeError(f"Unexpected patient_pooling: {self.patient_pooling}")

            logit_i = self.classifier(z_i).view(())
            logits.append(logit_i)
            patient_features.append(z_i)
            attention_weights.append(aux_i.detach().cpu())

        return torch.stack(logits, dim=0), torch.stack(patient_features, dim=0), attention_weights

class WholeImageEffB0ConvAdapterPatientPooling(nn.Module):
    def __init__(
        self,
        adapter_indices: str = "7,8",
        adapter_reduction: int = 8,
        adapter_alpha: float = 0.1,
        adapter_dropout: float = 0.1,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        head_dropout: float = 0.1,
        patient_pooling: str = "mean",
        freeze_bn: bool = True,
        num_prototypes_per_class: int = 2,
        lambda_proto: float = 0.25,
        lambda_std: float = 0.25,
        proto_temperature: float = 0.2,
    ):
        super().__init__()
        self.encoder = EfficientNetB0ConvAdapterEncoder(
            adapter_indices=adapter_indices,
            adapter_reduction=adapter_reduction,
            adapter_alpha=adapter_alpha,
            adapter_dropout=adapter_dropout,
            freeze_bn=freeze_bn,
        )
        self.patient_head = PatientPoolingHead(
            input_dim=1280,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=head_dropout,
            patient_pooling=patient_pooling,
            num_prototypes_per_class=num_prototypes_per_class,
            lambda_proto=lambda_proto,
            lambda_std=lambda_std,
            proto_temperature=proto_temperature,
        )

    def forward(self, images, patient_indices, num_patients: int):
        image_features = self.encoder(images)
        return self.patient_head(image_features, patient_indices, num_patients)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_optimizer(model, adapter_lr, head_lr, weight_decay):
    adapter_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "adapters" in name:
            adapter_params.append(p)
        else:
            head_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": adapter_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]
    )


def run_one_epoch(model, loader, criterion, optimizer, device, scaler=None, train: bool = True, grad_clip: float = 5.0):
    model.train() if train else model.eval()

    losses = []
    y_true, y_prob, patient_ids = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
            images = batch["images"].to(device, non_blocking=True)
            patient_indices = batch["patient_indices"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            num_patients = labels.shape[0]

            if train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, _ = model(images, patient_indices, num_patients)
                loss = criterion(logits, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            prob = torch.sigmoid(logits).detach().cpu().numpy()

            y_prob.extend(prob.tolist())
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            patient_ids.extend(batch["patient_ids"])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics["loss"] = float(np.mean(losses)) if len(losses) > 0 else float("nan")
    return metrics, y_true, y_prob, patient_ids


def train_one_fold(fold, patient_items, args, device, out_dir):
    train_items = [x for x in patient_items if int(x["fold"]) != int(fold)]
    val_items = [x for x in patient_items if int(x["fold"]) == int(fold)]

    train_loader = DataLoader(
        PatientWholeImageDataset(train_items, args.image_size, train=True),
        batch_size=args.patient_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=patient_collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        PatientWholeImageDataset(val_items, args.image_size, train=False),
        batch_size=args.patient_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=patient_collate_fn,
        drop_last=False,
    )

    model = WholeImageEffB0ConvAdapterPatientPooling(
        adapter_indices=args.adapter_indices,
        adapter_reduction=args.adapter_reduction,
        adapter_alpha=args.adapter_alpha,
        adapter_dropout=args.adapter_dropout,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        head_dropout=args.head_dropout,
        patient_pooling=args.patient_pooling,
        freeze_bn=not args.unfreeze_bn,
        num_prototypes_per_class=args.num_prototypes_per_class,
        lambda_proto=args.lambda_proto,
        lambda_std=args.lambda_std,
        proto_temperature=args.proto_temperature,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    train_labels = np.array([x["label"] for x in train_items])
    num_pos = max(int((train_labels == 1).sum()), 1)
    num_neg = max(int((train_labels == 0).sum()), 1)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = make_optimizer(model, args.adapter_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_info = {
        "fold": int(fold),
        "train_patients": len(train_items),
        "val_patients": len(val_items),
        "train_label_counts": {
            "0": int((train_labels == 0).sum()),
            "1": int((train_labels == 1).sum()),
        },
        "pos_weight": float(pos_weight.item()),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_ratio": float(trainable_params / total_params),
        "patient_pooling": args.patient_pooling,
        "num_prototypes_per_class": args.num_prototypes_per_class,
        "lambda_proto": args.lambda_proto,
        "lambda_std": args.lambda_std,
        "proto_temperature": args.proto_temperature,
        "adapter_indices": args.adapter_indices,
        "adapter_reduction": args.adapter_reduction,
        "adapter_alpha": args.adapter_alpha,
        "adapter_dropout": args.adapter_dropout,
        "freeze_bn": not args.unfreeze_bn,
    }

    with open(fold_dir / "fold_info.json", "w", encoding="utf-8") as f:
        json.dump(fold_info, f, indent=2, ensure_ascii=False)

    print(f"\n========== Fold {fold} ==========")
    print(json.dumps(fold_info, indent=2, ensure_ascii=False))

    best_auc = -1.0
    best_epoch = -1
    best_metrics = None
    best_preds = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _, _ = run_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, train=True, grad_clip=args.grad_clip
        )

        val_metrics, y_true, y_prob, pids = run_one_epoch(
            model, val_loader, criterion, optimizer, device, scaler=None, train=False, grad_clip=args.grad_clip
        )

        val_auc = val_metrics["roc_auc"]
        scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)
        current_lrs = [group["lr"] for group in optimizer.param_groups]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_acc": val_metrics["accuracy"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_precision": val_metrics["precision"],
            "val_f1": val_metrics["f1"],
            "adapter_lr": current_lrs[0],
            "head_lr": current_lrs[1],
        }
        history.append(row)

        print(
            f"Fold {fold} | Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_spec={val_metrics['specificity']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"lr=({current_lrs[0]:.2e},{current_lrs[1]:.2e})"
        )

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_preds = pd.DataFrame({
                "patient_id": pids,
                "label": y_true,
                "prob": y_prob,
                "pred": (np.array(y_prob) >= 0.5).astype(int),
            })

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": best_metrics,
                    "args": vars(args),
                },
                fold_dir / "best_model.pt",
            )

            best_preds.to_csv(fold_dir / "best_patient_predictions.csv", index=False, encoding="utf-8-sig")
            bad_epochs = 0
        else:
            bad_epochs += 1

        pd.DataFrame(history).to_csv(fold_dir / "training_log.csv", index=False, encoding="utf-8-sig")

        if bad_epochs >= args.patience:
            print(f"Fold {fold}: early stopping at epoch {epoch}, best_epoch={best_epoch}")
            break

    if best_metrics is None:
        best_metrics = val_metrics

    best_metrics = dict(best_metrics)
    best_metrics["fold"] = int(fold)
    best_metrics["best_epoch"] = int(best_epoch)
    best_metrics["n_train_patients"] = len(train_items)
    best_metrics["n_val_patients"] = len(val_items)

    with open(fold_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)

    return best_metrics, best_preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--patient-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--patient-pooling", type=str, default="mean", choices=["mean", "gated_attention", "phpr"])
    parser.add_argument("--num-prototypes-per-class", type=int, default=2)
    parser.add_argument("--lambda-proto", type=float, default=0.25)
    parser.add_argument("--lambda-std", type=float, default=0.25)
    parser.add_argument("--proto-temperature", type=float, default=0.2)

    parser.add_argument("--adapter-indices", type=str, default="7,8")
    parser.add_argument("--adapter-reduction", type=int, default=8)
    parser.add_argument("--adapter-alpha", type=float, default=0.1)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--head-dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--unfreeze-bn", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=str, default="all")

    args = parser.parse_args()
    args.amp = not args.no_amp

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_auto(args.metadata)

    required_cols = ["patient_id", "label", "fold", "processed_path"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Metadata is missing required column: {c}")

    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["fold"] = df["fold"].astype(int)

    patient_items = build_patient_items(df)

    all_folds = sorted(df["fold"].unique().tolist())
    if args.folds != "all":
        folds = [int(x) for x in args.folds.split(",")]
    else:
        folds = all_folds

    print("===== Dataset summary =====")
    print(f"Images: {len(df)}")
    print(f"Patients: {len(patient_items)}")
    print("Patient labels:")
    print(df.drop_duplicates("patient_id")["label"].value_counts().sort_index())
    print("Patients per fold:")
    print(df.drop_duplicates("patient_id")["fold"].value_counts().sort_index())
    print("Images per fold:")
    print(df["fold"].value_counts().sort_index())
    print("Images per patient:")
    print(df.groupby("patient_id").size().describe())
    print(f"Patient pooling: {args.patient_pooling}")
    if args.patient_pooling == "phpr":
        print(
            "PHPR config: "
            f"prototypes/class={args.num_prototypes_per_class}, "
            f"lambda_proto={args.lambda_proto}, "
            f"lambda_std={args.lambda_std}, "
            f"temperature={args.proto_temperature}"
        )
    print(f"Running folds: {folds}")

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    all_metrics = []
    all_preds = []

    for fold in folds:
        metrics, preds = train_one_fold(
            fold=int(fold),
            patient_items=patient_items,
            args=args,
            device=device,
            out_dir=out_dir,
        )
        all_metrics.append(metrics)
        if preds is not None:
            preds = preds.copy()
            preds["fold"] = int(fold)
            all_preds.append(preds)

    metrics_df = pd.DataFrame(all_metrics).sort_values("fold")
    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    if len(all_preds) > 0:
        pooled_df = pd.concat(all_preds, ignore_index=True)
        pooled_df.to_csv(out_dir / "pooled_oof_predictions.csv", index=False, encoding="utf-8-sig")
        pooled_metrics = compute_metrics(pooled_df["label"].values, pooled_df["prob"].values, threshold=0.5)
    else:
        pooled_metrics = None

    summary = {}
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "recall",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "mcc",
    ]

    for col in metric_cols:
        summary[col] = {
            "mean": float(metrics_df[col].mean()),
            "std": float(metrics_df[col].std()),
        }

    summary["pooled_out_of_fold"] = pooled_metrics
    summary["configuration"] = vars(args)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Five-fold results =====")
    print(metrics_df)

    print("\n===== Summary =====")
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"{k}: {v['mean']:.4f} +/- {v['std']:.4f}")

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
