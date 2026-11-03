#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def find_col(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"找不到 {name} 列。当前列名为: {list(df.columns)}")


def build_basename_index(processed_dir: Path):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in processed_dir.rglob("*") if p.suffix.lower() in image_exts]

    basename_to_path = {}
    duplicate = {}

    for p in files:
        name = p.name
        if name in basename_to_path:
            duplicate.setdefault(name, []).append(str(p))
        else:
            basename_to_path[name] = str(p)

    if duplicate:
        print(f"警告：处理后目录中存在 {len(duplicate)} 个重复文件名。脚本将使用第一次出现的路径。")

    return basename_to_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, required=True)
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    labels_csv = Path(args.labels_csv)
    processed_dir = Path(args.processed_dir)
    out_csv = Path(args.out_csv)

    def read_csv_auto(path):
        for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"无法自动识别CSV编码: {path}")

    meta = read_csv_auto(metadata_csv)
    labels = read_csv_auto(labels_csv)

    print("metadata columns:", list(meta.columns))
    print("labels columns:", list(labels.columns))

    patient_col_meta = find_col(
        meta,
        ["patient_id", "patient", "pid", "PatientID", "id"],
        "metadata中的patient_id",
    )

    image_col_meta = find_col(
        meta,
        ["image_path", "img_path", "path", "filename", "file_name", "image", "image_id"],
        "metadata中的图像路径/文件名",
    )

    fold_col = find_col(
        meta,
        ["fold", "Fold", "cv_fold", "split_fold"],
        "fold",
    )

    patient_col_label = find_col(
        labels,
        ["patient_id", "patient", "pid", "PatientID", "id"],
        "labels中的patient_id",
    )

    label_col = find_col(
        labels,
        ["label", "Label", "target", "DKD", "dkd", "y"],
        "label",
    )

    meta = meta.copy()
    labels = labels.copy()

    meta["patient_id"] = meta[patient_col_meta].astype(str)
    labels["patient_id"] = labels[patient_col_label].astype(str)

    labels = labels[["patient_id", label_col]].rename(columns={label_col: "label_from_labels"})
    labels = labels.drop_duplicates(subset=["patient_id"])

    df = meta.merge(labels, on="patient_id", how="left")

    # 优先使用 205_labels.csv 中的标签；如果没有，再使用 metadata.csv 自带的 label
    if "label_from_labels" in df.columns and not df["label_from_labels"].isna().all():
        df["label"] = df["label_from_labels"]
    elif "label" in df.columns:
        df["label"] = df["label"]
    else:
        raise ValueError("没有找到可用的 label 列。")

    if df["label"].isna().any():
        missing = df[df["label"].isna()]["patient_id"].unique()
        raise ValueError(f"有 {len(missing)} 个患者没有匹配到标签，例如: {missing[:10]}")

    basename_to_path = build_basename_index(processed_dir)

    original_paths = []
    processed_paths = []
    missing_images = []

    for x in df[image_col_meta].astype(str).tolist():
        original_paths.append(x)
        name = Path(x).name

        if name in basename_to_path:
            processed_paths.append(basename_to_path[name])
        else:
            processed_paths.append("")
            missing_images.append(name)

    df["original_path"] = original_paths
    df["processed_path"] = processed_paths

    if missing_images:
        print(f"错误：有 {len(missing_images)} 张图像没有在裁剪后目录中找到。")
        print("前20个缺失文件名：")
        print(missing_images[:20])
        raise ValueError("processed_path 匹配失败，请检查裁剪后图像目录和 metadata 中文件名是否一致。")

    out_df = pd.DataFrame({
        "patient_id": df["patient_id"].astype(str),
        "label": df["label"].astype(int),
        "fold": df[fold_col].astype(int),
        "original_path": df["original_path"].astype(str),
        "processed_path": df["processed_path"].astype(str),
    })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n===== Whole-image metadata summary =====")
    print(f"Total images: {len(out_df)}")
    print(f"Total patients: {out_df['patient_id'].nunique()}")
    print(f"Labels:")
    print(out_df.drop_duplicates("patient_id")["label"].value_counts().sort_index())
    print("\nImages per fold:")
    print(out_df["fold"].value_counts().sort_index())
    print("\nPatients per fold:")
    print(out_df.drop_duplicates("patient_id")["fold"].value_counts().sort_index())
    print("\nPatient labels per fold:")
    print(pd.crosstab(
        out_df.drop_duplicates("patient_id")["fold"],
        out_df.drop_duplicates("patient_id")["label"]
    ))

    print(f"\nSaved to: {out_csv}")


if __name__ == "__main__":
    main()