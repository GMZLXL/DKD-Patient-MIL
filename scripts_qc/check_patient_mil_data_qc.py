import os
import pandas as pd


PATIENT_CSV = "data_qc/patient_level_mil.csv"
METADATA_CSV = "data_qc/metadata.csv"
IMAGE_ROOT = "data_qc/image"


def split_image_paths(x):
    if pd.isna(x):
        return []
    return [p.strip() for p in str(x).split(";") if p.strip()]


def main():
    print("=" * 80)
    print("检查 patient_level_mil.csv 和 metadata.csv")
    print("=" * 80)

    patient_df = pd.read_csv(PATIENT_CSV, dtype={"patient_id": str})
    meta_df = pd.read_csv(METADATA_CSV, dtype={"patient_id": str})

    patient_df["fold"] = patient_df["fold"].astype(int)
    patient_df["label"] = patient_df["label"].astype(int)
    patient_df["n_images"] = patient_df["n_images"].astype(int)

    meta_df["fold"] = meta_df["fold"].astype(int)
    meta_df["label"] = meta_df["label"].astype(int)

    print("\n一、基础信息")
    print("patient_level_mil.csv 行数:", len(patient_df))
    print("patient_level_mil.csv 患者数:", patient_df["patient_id"].nunique())
    print("metadata.csv 图像数:", len(meta_df))
    print("metadata.csv 患者数:", meta_df["patient_id"].nunique())

    # ======================================================
    # 1. 检查 patient_level_mil.csv 是否一行一个患者
    # ======================================================
    print("\n二、检查是否一行一个患者")

    duplicated_patients = patient_df[patient_df["patient_id"].duplicated(keep=False)]

    if len(duplicated_patients) == 0:
        print("通过：patient_level_mil.csv 中每个患者只出现一次。")
    else:
        print("错误：以下患者在 patient_level_mil.csv 中重复出现：")
        print(duplicated_patients[["patient_id", "label", "fold", "n_images"]])

    # ======================================================
    # 2. 检查 image_paths 数量是否等于 n_images
    # ======================================================
    print("\n三、检查 n_images 和 image_paths 数量是否一致")

    count_errors = []

    for _, row in patient_df.iterrows():
        patient_id = row["patient_id"]
        n_images = int(row["n_images"])
        image_paths = split_image_paths(row["image_paths"])

        if len(image_paths) != n_images:
            count_errors.append({
                "patient_id": patient_id,
                "n_images": n_images,
                "actual_image_paths_count": len(image_paths),
                "image_paths": row["image_paths"],
            })

    if len(count_errors) == 0:
        print("通过：所有患者的 n_images 都和 image_paths 数量一致。")
    else:
        print("错误：以下患者的 n_images 和 image_paths 数量不一致：")
        print(pd.DataFrame(count_errors))

    # ======================================================
    # 3. 检查图片文件是否真实存在
    # ======================================================
    print("\n四、检查 image_paths 中的图片文件是否存在")

    missing_images = []
    all_patient_image_paths = []

    for _, row in patient_df.iterrows():
        patient_id = row["patient_id"]
        image_paths = split_image_paths(row["image_paths"])

        for img in image_paths:
            all_patient_image_paths.append(img)
            img_path = os.path.join(IMAGE_ROOT, img)

            if not os.path.exists(img_path):
                missing_images.append({
                    "patient_id": patient_id,
                    "image_path": img,
                    "full_path": img_path,
                })

    if len(missing_images) == 0:
        print("通过：patient_level_mil.csv 中所有图片都能在 data/image 中找到。")
    else:
        print("错误：以下图片文件不存在：")
        print(pd.DataFrame(missing_images).head(50))
        print("缺失图片总数:", len(missing_images))

    # ======================================================
    # 4. 检查 patient_level_mil.csv 和 metadata.csv 的图片是否一致
    # ======================================================
    print("\n五、检查 patient_level_mil.csv 和 metadata.csv 的图片列表是否一致")

    patient_image_set = set(all_patient_image_paths)
    meta_image_set = set(meta_df["image_path"].astype(str).tolist())

    only_in_patient_csv = patient_image_set - meta_image_set
    only_in_metadata = meta_image_set - patient_image_set

    if len(only_in_patient_csv) == 0 and len(only_in_metadata) == 0:
        print("通过：patient_level_mil.csv 和 metadata.csv 中的图片列表完全一致。")
    else:
        print("警告：两个 CSV 的图片列表不完全一致。")
        print("只在 patient_level_mil.csv 中出现的图片数:", len(only_in_patient_csv))
        print("只在 metadata.csv 中出现的图片数:", len(only_in_metadata))

        if len(only_in_patient_csv) > 0:
            print("只在 patient_level_mil.csv 中的示例:")
            print(list(only_in_patient_csv)[:10])

        if len(only_in_metadata) > 0:
            print("只在 metadata.csv 中的示例:")
            print(list(only_in_metadata)[:10])

    # ======================================================
    # 5. 检查每个患者在 metadata.csv 里 label 和 fold 是否唯一
    # ======================================================
    print("\n六、检查 metadata.csv 中每个患者的 label 和 fold 是否唯一")

    meta_group_check = meta_df.groupby("patient_id").agg(
        n_labels=("label", "nunique"),
        n_folds=("fold", "nunique"),
        n_images=("image_path", "count"),
    ).reset_index()

    bad_meta_group = meta_group_check[
        (meta_group_check["n_labels"] != 1) | (meta_group_check["n_folds"] != 1)
    ]

    if len(bad_meta_group) == 0:
        print("通过：metadata.csv 中每个患者的所有图片 label 和 fold 都一致。")
    else:
        print("错误：以下患者在 metadata.csv 中存在多个 label 或多个 fold：")
        print(bad_meta_group)

    # ======================================================
    # 6. 检查 patient_level_mil.csv 和 metadata.csv 的 label/fold 是否一致
    # ======================================================
    print("\n七、检查 patient_level_mil.csv 和 metadata.csv 的 label/fold 是否一致")

    meta_patient_df = meta_df.groupby("patient_id").agg(
        meta_label=("label", "first"),
        meta_fold=("fold", "first"),
        meta_n_images=("image_path", "count"),
    ).reset_index()

    merged = patient_df.merge(meta_patient_df, on="patient_id", how="outer", indicator=True)

    missing_patient_info = merged[merged["_merge"] != "both"]

    if len(missing_patient_info) == 0:
        print("通过：两个 CSV 的患者列表一致。")
    else:
        print("错误：两个 CSV 的患者列表不一致：")
        print(missing_patient_info[["patient_id", "_merge"]])

    both_df = merged[merged["_merge"] == "both"].copy()

    label_fold_errors = both_df[
        (both_df["label"] != both_df["meta_label"]) |
        (both_df["fold"] != both_df["meta_fold"]) |
        (both_df["n_images"] != both_df["meta_n_images"])
    ]

    if len(label_fold_errors) == 0:
        print("通过：patient_level_mil.csv 和 metadata.csv 的 label、fold、n_images 完全一致。")
    else:
        print("错误：以下患者的 label、fold 或 n_images 不一致：")
        print(label_fold_errors[
            ["patient_id", "label", "meta_label", "fold", "meta_fold", "n_images", "meta_n_images"]
        ])

    # ======================================================
    # 7. 检查五折划分是否患者级无泄漏
    # ======================================================
    print("\n八、检查五折训练/验证是否患者无重叠")

    leakage_errors = []

    for fold in sorted(patient_df["fold"].unique()):
        train_patients = set(patient_df.loc[patient_df["fold"] != fold, "patient_id"])
        val_patients = set(patient_df.loc[patient_df["fold"] == fold, "patient_id"])

        overlap = train_patients & val_patients

        if len(overlap) > 0:
            leakage_errors.append({
                "fold": fold,
                "n_overlap_patients": len(overlap),
                "overlap_examples": list(overlap)[:10],
            })

    if len(leakage_errors) == 0:
        print("通过：五折划分中训练集和验证集患者没有重叠。")
    else:
        print("错误：存在患者级数据泄漏：")
        print(pd.DataFrame(leakage_errors))

    # ======================================================
    # 8. 输出 fold 分布
    # ======================================================
    print("\n九、患者级 fold 分布")

    fold_summary = patient_df.groupby("fold").agg(
        n_patients=("patient_id", "count"),
        n_images=("n_images", "sum"),
        n_positive=("label", "sum"),
    ).reset_index()

    fold_summary["n_negative"] = fold_summary["n_patients"] - fold_summary["n_positive"]
    fold_summary["positive_ratio"] = fold_summary["n_positive"] / fold_summary["n_patients"]

    print(fold_summary)

    # ======================================================
    # 9. 总结
    # ======================================================
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

    has_error = (
        len(duplicated_patients) > 0 or
        len(count_errors) > 0 or
        len(missing_images) > 0 or
        len(only_in_patient_csv) > 0 or
        len(only_in_metadata) > 0 or
        len(bad_meta_group) > 0 or
        len(missing_patient_info) > 0 or
        len(label_fold_errors) > 0 or
        len(leakage_errors) > 0
    )

    if has_error:
        print("结论：发现问题，请根据上面的错误信息修正数据。")
    else:
        print("结论：未发现明显数据组织问题，可以继续进行患者级多图实验。")


if __name__ == "__main__":
    main()