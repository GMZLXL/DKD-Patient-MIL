#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whole fundus image preprocessing and quality scoring.

功能：
1. 自动裁剪眼底有效视野，减少黑边干扰
2. padding 成正方形
3. resize 到指定尺寸
4. 可选 CLAHE 亮度增强
5. 计算图像质量指标：
   - valid_ratio：有效视野占比
   - sharpness：清晰度
   - brightness：亮度
   - contrast：对比度
   - glare_ratio：亮斑比例
   - dark_ratio：过暗比例
   - quality_score：综合质量分数
6. 保存处理后图像和 image_quality_report.csv

推荐用途：
整图 EfficientNet-B0 baseline / fine-tuning 前的数据预处理。
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def find_image_files(image_dir: Path) -> List[Path]:
    files = []
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def read_metadata_images(
    metadata_csv: Path,
    image_dir: Path,
    image_col: Optional[str],
) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)

    if image_col is None:
        candidates = [
            "image_path",
            "img_path",
            "path",
            "filename",
            "file_name",
            "image",
            "image_id",
        ]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise ValueError(
                f"没有找到图像路径列。请用 --image-col 指定。当前CSV列为: {list(df.columns)}"
            )
        image_col = found

    if image_col not in df.columns:
        raise ValueError(f"指定的 image_col={image_col} 不在CSV中。")

    paths = []
    for x in df[image_col].astype(str).tolist():
        p = Path(x)
        if not p.is_absolute():
            p = image_dir / p
        paths.append(p)

    df = df.copy()
    df["_image_path_resolved"] = [str(p) for p in paths]
    df["_image_col_used"] = image_col
    return df


def largest_contour_mask(gray: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    根据非黑区域检测眼底圆形视野。
    """
    mask = (gray > threshold).astype(np.uint8) * 255

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return mask

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=-1)
    return clean


def crop_fundus_fov(
    img_bgr: np.ndarray,
    black_threshold: int = 10,
    margin_ratio: float = 0.03,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    裁剪眼底有效区域。
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = largest_contour_mask(gray, threshold=black_threshold)

    valid_area = int((mask > 0).sum())
    total_area = int(h * w)
    original_valid_ratio = valid_area / max(total_area, 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0 or valid_area < 100:
        metrics = {
            "original_valid_ratio": original_valid_ratio,
            "crop_success": 0.0,
            "crop_x": 0,
            "crop_y": 0,
            "crop_w": w,
            "crop_h": h,
        }
        return img_bgr, metrics

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    margin = int(max(bw, bh) * margin_ratio)

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)

    crop = img_bgr[y1:y2, x1:x2].copy()

    metrics = {
        "original_valid_ratio": original_valid_ratio,
        "crop_success": 1.0,
        "crop_x": x1,
        "crop_y": y1,
        "crop_w": x2 - x1,
        "crop_h": y2 - y1,
    }
    return crop, metrics


def pad_to_square(img_bgr: np.ndarray, pad_value: int = 0) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = max(h, w)

    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left

    padded = cv2.copyMakeBorder(
        img_bgr,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return padded


def apply_clahe_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    在 LAB 空间只增强 L 通道，避免过度改变颜色。
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


def compute_quality_metrics(img_bgr: np.ndarray, black_threshold: int = 10) -> Dict[str, float]:
    """
    计算图像质量指标。
    注意：这里不是医学诊断，只是图像质量估计。
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mask = largest_contour_mask(gray, threshold=black_threshold)
    valid = mask > 0

    valid_ratio = float(valid.sum()) / float(h * w + 1e-6)

    if valid.sum() < 100:
        return {
            "valid_ratio": valid_ratio,
            "sharpness": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "glare_ratio": 1.0,
            "dark_ratio": 1.0,
            "quality_score": 0.0,
        }

    valid_gray = gray[valid]
    brightness = float(valid_gray.mean())
    contrast = float(valid_gray.std())

    # 清晰度：Laplacian 方差
    # 为避免黑边影响，先用中位数填充非有效区域
    gray_filled = gray.copy()
    med = int(np.median(valid_gray))
    gray_filled[~valid] = med
    sharpness = float(cv2.Laplacian(gray_filled, cv2.CV_64F).var())

    # 亮斑比例：非常高亮的像素
    b, g, r = cv2.split(img_bgr)
    glare_mask = (r > 240) & (g > 240) & (b > 220) & valid
    glare_ratio = float(glare_mask.sum()) / float(valid.sum() + 1e-6)

    # 过暗比例
    dark_mask = (gray < 25) & valid
    dark_ratio = float(dark_mask.sum()) / float(valid.sum() + 1e-6)

    # 下面是经验归一化，后续可以根据你的数据分布微调
    valid_score = clamp01((valid_ratio - 0.45) / 0.35)

    # Laplacian 方差不同数据差异较大，用 log 缓和
    sharpness_score = clamp01(np.log1p(sharpness) / np.log1p(1200.0))

    # 眼底图像平均亮度一般不应太暗或太亮
    brightness_norm = brightness / 255.0
    brightness_score = clamp01(1.0 - abs(brightness_norm - 0.45) / 0.45)

    # 对比度太低说明发雾/模糊
    contrast_score = clamp01((contrast - 18.0) / 45.0)

    # 亮斑比例超过 3% 明显惩罚
    glare_score = 1.0 - clamp01(glare_ratio / 0.03)

    # 过暗区域比例超过 40% 明显惩罚
    dark_score = 1.0 - clamp01(dark_ratio / 0.40)

    quality_score = (
        0.25 * sharpness_score
        + 0.20 * contrast_score
        + 0.20 * glare_score
        + 0.15 * brightness_score
        + 0.10 * valid_score
        + 0.10 * dark_score
    )

    return {
        "valid_ratio": valid_ratio,
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "glare_ratio": glare_ratio,
        "dark_ratio": dark_ratio,
        "valid_score": valid_score,
        "sharpness_score": sharpness_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "glare_score": glare_score,
        "dark_score": dark_score,
        "quality_score": float(quality_score),
    }


def safe_relative_path(img_path: Path, image_dir: Path) -> Path:
    try:
        return img_path.resolve().relative_to(image_dir.resolve())
    except Exception:
        return Path(img_path.name)


def process_one_image(
    img_path: Path,
    image_dir: Path,
    out_image_dir: Path,
    resize: int,
    apply_clahe: bool,
    black_threshold: int,
    quality_threshold: float,
    clahe_clip: float,
) -> Dict[str, object]:

    record: Dict[str, object] = {
        "original_path": str(img_path),
        "processed_path": "",
        "read_success": 0,
        "keep": 0,
        "quality_group": "read_failed",
    }

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return record

    record["read_success"] = 1
    orig_h, orig_w = img.shape[:2]
    record["orig_w"] = orig_w
    record["orig_h"] = orig_h

    # 1. 裁剪眼底视野
    crop, crop_metrics = crop_fundus_fov(
        img,
        black_threshold=black_threshold,
        margin_ratio=0.03,
    )
    record.update(crop_metrics)

    # 2. padding 成正方形
    padded = pad_to_square(crop, pad_value=0)

    # 3. 可选 CLAHE
    if apply_clahe:
        padded = apply_clahe_bgr(padded, clip_limit=clahe_clip, tile_grid_size=8)

    # 4. resize
    resized = cv2.resize(padded, (resize, resize), interpolation=cv2.INTER_AREA)

    # 5. 计算质量分数
    quality = compute_quality_metrics(resized, black_threshold=black_threshold)
    record.update(quality)

    quality_score = float(quality["quality_score"])

    if quality_score >= 0.70:
        quality_group = "high"
    elif quality_score >= 0.40:
        quality_group = "medium"
    elif quality_score >= quality_threshold:
        quality_group = "low"
    else:
        quality_group = "reject"

    keep = int(quality_score >= quality_threshold)

    record["keep"] = keep
    record["quality_group"] = quality_group

    # 6. 保存处理后的图像
    rel_path = safe_relative_path(img_path, image_dir)
    out_path = out_image_dir / rel_path
    ensure_dir(out_path.parent)

    cv2.imwrite(str(out_path), resized)
    record["processed_path"] = str(out_path)

    return record


def summarize_report(df: pd.DataFrame, out_dir: Path, label_col: Optional[str] = None) -> None:
    summary_lines = []

    summary_lines.append(f"Total images: {len(df)}")
    summary_lines.append(f"Read success: {int(df['read_success'].sum())}")

    if "keep" in df.columns:
        summary_lines.append(f"Keep images: {int(df['keep'].sum())}")
        summary_lines.append(f"Reject images: {int((df['keep'] == 0).sum())}")

    if "quality_score" in df.columns:
        summary_lines.append("")
        summary_lines.append("Quality score statistics:")
        summary_lines.append(str(df["quality_score"].describe()))

    if "quality_group" in df.columns:
        summary_lines.append("")
        summary_lines.append("Quality group counts:")
        summary_lines.append(str(df["quality_group"].value_counts()))

    if label_col is not None and label_col in df.columns:
        summary_lines.append("")
        summary_lines.append(f"Quality group by label: {label_col}")
        summary_lines.append(str(pd.crosstab(df[label_col], df["quality_group"])))

        summary_lines.append("")
        summary_lines.append(f"Keep by label: {label_col}")
        summary_lines.append(str(pd.crosstab(df[label_col], df["keep"])))

    text = "\n".join(summary_lines)
    print("\n" + text + "\n")

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", type=str, required=True, help="原始眼底图像目录")
    parser.add_argument("--out-dir", type=str, required=True, help="输出目录")

    parser.add_argument("--metadata-csv", type=str, default=None, help="可选，包含图像路径、患者ID、标签的CSV")
    parser.add_argument("--image-col", type=str, default=None, help="CSV中的图像路径列名")
    parser.add_argument("--patient-col", type=str, default="patient_id", help="CSV中的患者ID列名")
    parser.add_argument("--label-col", type=str, default="label", help="CSV中的标签列名")

    parser.add_argument("--resize", type=int, default=384, help="输出图像尺寸，推荐 384")
    parser.add_argument("--black-threshold", type=int, default=10, help="黑边阈值")
    parser.add_argument("--quality-threshold", type=float, default=0.30, help="低于该质量分数标记为reject")
    parser.add_argument("--apply-clahe", action="store_true", help="是否启用轻度CLAHE")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clipLimit")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_image_dir = out_dir / "images"

    ensure_dir(out_dir)
    ensure_dir(out_image_dir)

    if args.metadata_csv is not None:
        metadata_csv = Path(args.metadata_csv)
        meta_df = read_metadata_images(metadata_csv, image_dir, args.image_col)
        image_paths = [Path(p) for p in meta_df["_image_path_resolved"].tolist()]
    else:
        meta_df = None
        image_paths = find_image_files(image_dir)

    print(f"Found {len(image_paths)} images.")

    records = []

    for idx, img_path in enumerate(tqdm(image_paths, desc="Preprocessing fundus images")):
        rec = process_one_image(
            img_path=img_path,
            image_dir=image_dir,
            out_image_dir=out_image_dir,
            resize=args.resize,
            apply_clahe=args.apply_clahe,
            black_threshold=args.black_threshold,
            quality_threshold=args.quality_threshold,
            clahe_clip=args.clahe_clip,
        )
        rec["_order"] = idx
        records.append(rec)

    report_df = pd.DataFrame(records)

    if meta_df is not None:
        meta_df = meta_df.reset_index(drop=True)
        report_df = report_df.reset_index(drop=True)

        # 合并原始metadata，方便后续按patient和label统计
        merged = pd.concat([meta_df, report_df], axis=1)
    else:
        merged = report_df

    report_path = out_dir / "image_quality_report.csv"
    merged.to_csv(report_path, index=False, encoding="utf-8-sig")

    # 只保留 keep=1 的图像清单
    if "keep" in merged.columns:
        keep_df = merged[merged["keep"] == 1].copy()
        keep_df.to_csv(out_dir / "image_quality_keep.csv", index=False, encoding="utf-8-sig")

        reject_df = merged[merged["keep"] == 0].copy()
        reject_df.to_csv(out_dir / "image_quality_reject.csv", index=False, encoding="utf-8-sig")

    label_col_for_summary = args.label_col if args.label_col in merged.columns else None
    summarize_report(merged, out_dir, label_col=label_col_for_summary)

    print(f"Saved processed images to: {out_image_dir}")
    print(f"Saved quality report to: {report_path}")


if __name__ == "__main__":
    main()