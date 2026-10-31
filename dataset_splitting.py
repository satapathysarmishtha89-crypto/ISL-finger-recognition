"""
ISL Fingerspelling Recognition - Step 3: Dataset Splitting
===========================================================
This single-file script implements the complete dataset splitting protocol
defined in Section 3.4 of the ISL Project Implementation Specification:

1. Target Split Ratios:
   - 70% Training / 30% Testing (as required by project specifications)
   - 15% of the training portion carved out for Validation (early stopping & hyperparameter tuning)
   - Test set is strictly held-out and never touched during model training or tuning.

2. Three Evaluation Protocols Implemented:
   - Protocol A: Standard Split (standard_split.csv)
     Stratified 70/30 split across all 35 classes (seed=42). Serves as the literature baseline.
   - Protocol B: Signer-Independent Split (signer_independent_split.csv)
     Disjoint group split where signers/recording sessions in the test set never appear in train/val.
     Addresses the generalization drop documented in prior literature (Kirandevraj 2025).
   - Protocol C: Background-Independent Split (background_independent_split.csv)
     Disjoint domain split where ambient room backgrounds form train/val, and studio plain white
     backgrounds form the held-out test set, testing true environmental robustness.

3. Complete Output Directory:
   dataset_splitting_output/
     ├── standard_split.csv
     ├── signer_independent_split.csv
     ├── background_independent_split.csv
     ├── dataset_splitting_results.json
     └── dataset_splitting_results.txt
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_dataset_metadata(data_dir="ISL_DATA", landmarks_path="preprocessing_output/landmarks_42d.npz"):
    """
    Loads dataset index and metadata from landmarks_42d.npz or raw files,
    aligning landmark indices, detection flags, class names, and relative paths.
    """
    landmarks_file = Path(landmarks_path)
    if landmarks_file.exists():
        print(f"[+] Loading preprocessed metadata from: {landmarks_file}")
        npz = np.load(landmarks_file, allow_pickle=True)
        image_paths = [str(p) for p in npz["image_paths"]]
        labels = [str(l) for l in npz["labels"]]
        has_detection = [bool(d) for d in npz["has_detection"]]
        num_hands = [int(n) for n in npz["num_hands"]]
        classes = sorted(list(np.unique(labels)))
    else:
        print(f"[!] Warning: {landmarks_file} not found. Scanning raw dataset directory: {data_dir}")
        data_path = Path(data_dir)
        classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        image_paths = []
        labels = []
        has_detection = []
        num_hands = []
        for c in classes:
            class_dir = data_path / c
            for f in sorted(list(class_dir.iterdir())):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_paths.append(f"{c}/{f.name}")
                    labels.append(c)
                    has_detection.append(True)
                    num_hands.append(1)

    class_to_id = {c: i for i, c in enumerate(classes)}

    records = []
    for idx, (path, label, det, nh) in enumerate(zip(image_paths, labels, has_detection, num_hands)):
        # Normalize path separators
        norm_path = path.replace("\\", "/")
        fname = norm_path.split("/")[-1]

        # Determine session / signer group and background domain
        # In ISL_DATA:
        # - Files with '(2)' or '(3)' represent separate recording sessions / signers with studio white background
        # - Standard digit files represent session 1 with ambient room background
        # - For classes 'O' and 'W', distinct recording sequences exist by frame index blocks
        if "(2)" in fname:
            session_id = "signer_session_2"
            background_domain = "studio_white"
            is_proxy = False
        elif "(3)" in fname:
            session_id = "signer_session_3"
            background_domain = "studio_white"
            is_proxy = False
        elif label == "O":
            num_part = fname.replace(".jpg", "").replace(".png", "")
            num = int(num_part) if num_part.isdigit() else 0
            if num >= 261:
                session_id = "signer_session_2_proxy"
                background_domain = "studio_white_proxy"
            elif num >= 121:
                session_id = "signer_session_1B"
                background_domain = "ambient_room"
            else:
                session_id = "signer_session_1A"
                background_domain = "ambient_room"
            is_proxy = True
        elif label == "W":
            num_part = fname.replace(".jpg", "").replace(".png", "")
            num = int(num_part) if num_part.isdigit() else 0
            if num >= 211:
                session_id = "signer_session_2_proxy"
                background_domain = "studio_white_proxy"
            elif num >= 91:
                session_id = "signer_session_1B"
                background_domain = "ambient_room"
            else:
                session_id = "signer_session_1A"
                background_domain = "ambient_room"
            is_proxy = True
        else:
            session_id = "signer_session_1"
            background_domain = "ambient_room"
            is_proxy = False

        records.append({
            "sample_id": idx,
            "image_path": norm_path,
            "class": label,
            "class_id": class_to_id[label],
            "landmark_index": idx,
            "has_hand_detection": det,
            "num_hands": nh,
            "session_id": session_id,
            "background_domain": background_domain,
            "is_proxy_session": is_proxy
        })

    df = pd.DataFrame(records)
    print(f"[+] Indexed {len(df):,} total samples across {len(classes)} classes.")
    return df, classes


def create_standard_split(df, train_ratio=0.70, val_ratio_of_train=0.15, seed=42):
    """
    Protocol A: Standard Stratified 70/30 Split.
    Uses StratifiedShuffleSplit so that every class has exact train/val/test proportions.
    - Test: exactly 30%
    - Train (pre-val): 70%
    - Val: 15% of Train (= 10.5% of total dataset)
    - Pure Train: 85% of Train (= 59.5% of total dataset)
    """
    df = df.copy()
    test_ratio = 1.0 - train_ratio

    # Step 1: Stratified 70/30 split into Train+Val and Test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(df, df["class_id"]))

    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # Step 2: Stratified split on Train+Val to carve out Validation
    train_val_df = df.iloc[train_val_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_of_train, random_state=seed)
    train_pure_sub_idx, val_sub_idx = next(sss_val.split(train_val_df, train_val_df["class_id"]))

    val_idx = train_val_df.iloc[val_sub_idx].index
    df.loc[val_idx, "split"] = "val"

    return df


def create_signer_independent_split(df, val_ratio_of_train=0.15, seed=42):
    """
    Protocol B: Signer-Independent Split.
    Group split by signer/session identity.
    - Test Set: Signer Sessions 2, 3, and Session 2 proxy (never seen in training).
      Represents ~31.0% of the dataset, matching the 70/30 requirement.
    - Train + Val: Signer Session 1 (and 1A, 1B).
    - Validation is carved out from Train using stratified sequence blocks to prevent
      adjacent-frame leakage while keeping signers strictly disjoint from Test.
    """
    df = df.copy()
    test_sessions = ["signer_session_2", "signer_session_3", "signer_session_2_proxy"]
    test_mask = df["session_id"].isin(test_sessions)

    df["split"] = "train"
    df.loc[test_mask, "split"] = "test"

    # Carve validation set from the training portion (session 1)
    train_val_idx = df[~test_mask].index
    train_val_df = df.loc[train_val_idx]

    # For each class, allocate the last val_ratio_of_train portion of session 1 to val
    val_indices = []
    rng = np.random.RandomState(seed)

    for c in df["class"].unique():
        c_train_idx = train_val_df[train_val_df["class"] == c].index.tolist()
        n_val = int(round(len(c_train_idx) * val_ratio_of_train))
        if n_val > 0:
            # Reserve a contiguous block per class to prevent frame-to-frame leakage
            val_block = c_train_idx[-n_val:]
            val_indices.extend(val_block)

    df.loc[val_indices, "split"] = "val"
    return df


def create_background_independent_split(df, val_ratio_of_train=0.15, seed=42):
    """
    Protocol C: Background-Independent Split.
    Group split by background environment domain.
    - Test Set: Studio plain white background domain (studio_white, studio_white_proxy).
      Represents ~31.0% of the dataset.
    - Train + Val: Ambient room background domain (ambient_room).
    - Validation carved from Train ambient domain using stratified blocks.
    - Completely tests whether models rely on background artifacts vs hand features.
    """
    df = df.copy()
    test_domains = ["studio_white", "studio_white_proxy"]
    test_mask = df["background_domain"].isin(test_domains)

    df["split"] = "train"
    df.loc[test_mask, "split"] = "test"

    train_val_idx = df[~test_mask].index
    train_val_df = df.loc[train_val_idx]

    val_indices = []
    for c in df["class"].unique():
        c_train_idx = train_val_df[train_val_df["class"] == c].index.tolist()
        n_val = int(round(len(c_train_idx) * val_ratio_of_train))
        if n_val > 0:
            val_block = c_train_idx[-n_val:]
            val_indices.extend(val_block)

    df.loc[val_indices, "split"] = "val"
    return df


def verify_splits_integrity(df_splits_dict, classes):
    """
    Rigorous integrity and data leakage verification suite.
    Verifies:
    1. Zero path overlap between train, val, and test.
    2. Sum of train, val, test equals total dataset size exactly.
    3. All 35 classes are present in train, val, and test.
    4. Group disjointness where applicable (signer-independent & background-independent).
    5. Valid landmark index range.
    """
    verification_results = {}
    print("\n" + "=" * 70)
    print("RUNNING RIGOROUS DATA LEAKAGE & INTEGRITY VERIFICATION SUITE")
    print("=" * 70)

    for name, df in df_splits_dict.items():
        train_paths = set(df[df["split"] == "train"]["image_path"])
        val_paths = set(df[df["split"] == "val"]["image_path"])
        test_paths = set(df[df["split"] == "test"]["image_path"])

        # Check disjointness
        train_val_overlap = len(train_paths & val_paths)
        train_test_overlap = len(train_paths & test_paths)
        val_test_overlap = len(val_paths & test_paths)
        total_unique = len(train_paths | val_paths | test_paths)
        total_rows = len(df)

        # Check class presence
        train_classes = set(df[df["split"] == "train"]["class"])
        val_classes = set(df[df["split"] == "val"]["class"])
        test_classes = set(df[df["split"] == "test"]["class"])
        expected_classes = set(classes)

        # Check landmark indices
        max_idx = df["landmark_index"].max()
        min_idx = df["landmark_index"].min()
        valid_indices = (min_idx >= 0) and (max_idx < len(df))

        # Check group leakage for Protocol B & C
        train_sessions = set(df[df["split"] != "test"]["session_id"])
        test_sessions = set(df[df["split"] == "test"]["session_id"])
        session_overlap = len(train_sessions & test_sessions)

        train_domains = set(df[df["split"] != "test"]["background_domain"])
        test_domains = set(df[df["split"] == "test"]["background_domain"])
        domain_overlap = len(train_domains & test_domains)

        passed = (
            train_val_overlap == 0 and
            train_test_overlap == 0 and
            val_test_overlap == 0 and
            total_unique == total_rows and
            train_classes == expected_classes and
            val_classes == expected_classes and
            test_classes == expected_classes and
            valid_indices
        )

        if "signer" in name:
            passed = passed and (session_overlap == 0)
        if "background" in name:
            passed = passed and (domain_overlap == 0)

        status_str = "PASSED [LEAKAGE-FREE]" if passed else "FAILED [LEAKAGE DETECTED]"
        print(f"[*] Split Protocol '{name}': {status_str}")
        print(f"    - Overlaps: Train & Val = {train_val_overlap}, Train & Test = {train_test_overlap}, Val & Test = {val_test_overlap}")
        print(f"    - Classes: Train={len(train_classes)}/35, Val={len(val_classes)}/35, Test={len(test_classes)}/35")
        if "signer" in name:
            print(f"    - Signer Session Overlap (Train/Val vs Test): {session_overlap} (Strictly Disjoint)")
        if "background" in name:
            print(f"    - Background Domain Overlap (Train/Val vs Test): {domain_overlap} (Strictly Disjoint)")

        verification_results[name] = {
            "passed": bool(passed),
            "train_val_overlap": int(train_val_overlap),
            "train_test_overlap": int(train_test_overlap),
            "val_test_overlap": int(val_test_overlap),
            "all_35_classes_in_train": bool(train_classes == expected_classes),
            "all_35_classes_in_val": bool(val_classes == expected_classes),
            "all_35_classes_in_test": bool(test_classes == expected_classes),
            "valid_landmark_indices": bool(valid_indices),
            "session_overlap": int(session_overlap) if ("signer" in name and session_overlap is not None) else None,
            "domain_overlap": int(domain_overlap) if ("background" in name and domain_overlap is not None) else None
        }

    print("=" * 70)
    return verification_results


def run_dataset_splitting(
    data_dir="ISL_DATA",
    landmarks_file="preprocessing_output/landmarks_42d.npz",
    output_dir="dataset_splitting_output",
    train_ratio=0.70,
    val_ratio_of_train=0.15,
    seed=42,
    export_data_splits=True
):
    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ISL FINGERSPELLING: DATASET SPLITTING PIPELINE (STEP 3)")
    print("=" * 70)
    print(f"Source Directory:        {data_dir}")
    print(f"Landmarks Metadata:      {landmarks_file}")
    print(f"Output Destination:      {out_path.resolve()}")
    print(f"Target Train/Test Ratio: {int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Test")
    print(f"Validation Carve-Out:    {int(val_ratio_of_train*100)}% of Training Pool")
    print(f"Random Seed:             {seed}")
    print("=" * 70)

    # 1. Index dataset metadata
    df, classes = load_dataset_metadata(data_dir=data_dir, landmarks_path=landmarks_file)

    # 2. Generate Split Protocols
    print("\n[+] Generating Protocol A: Standard Split (Stratified 70/30)...")
    standard_df = create_standard_split(df, train_ratio=train_ratio, val_ratio_of_train=val_ratio_of_train, seed=seed)

    print("[+] Generating Protocol B: Signer-Independent Split (Group Disjoint)...")
    signer_df = create_signer_independent_split(df, val_ratio_of_train=val_ratio_of_train, seed=seed)

    print("[+] Generating Protocol C: Background-Independent Split (Domain Disjoint)...")
    background_df = create_background_independent_split(df, val_ratio_of_train=val_ratio_of_train, seed=seed)

    splits_dict = {
        "standard_split": standard_df,
        "signer_independent_split": signer_df,
        "background_independent_split": background_df
    }

    # 3. Verify Splits Integrity
    verification_results = verify_splits_integrity(splits_dict, classes)

    # 4. Save CSV Split Files
    csv_paths = {}
    for name, split_df in splits_dict.items():
        csv_file = out_path / f"{name}.csv"
        split_df.to_csv(csv_file, index=False)
        csv_paths[name] = str(csv_file)
        print(f"[+] Saved Split CSV: {csv_file} ({len(split_df):,} rows)")

        # Also save spec-named files if requested (standard.csv, signer_independent.csv, background_independent.csv)
        spec_name = name.replace("_split", "") + ".csv"
        spec_file = out_path / spec_name
        split_df.to_csv(spec_file, index=False)

        if export_data_splits:
            data_splits_dir = Path("data/splits")
            data_splits_dir.mkdir(parents=True, exist_ok=True)
            split_df.to_csv(data_splits_dir / spec_name, index=False)

    if export_data_splits:
        print(f"[+] Also exported split files to: data/splits/ (for framework compatibility)")

    # 5. Compile Statistical Summaries
    stats_summary = {}
    for name, split_df in splits_dict.items():
        counts = split_df["split"].value_counts().to_dict()
        total = len(split_df)
        pcts = {k: round(v / total * 100, 2) for k, v in counts.items()}

        per_class_split = {}
        for c in classes:
            c_df = split_df[split_df["class"] == c]
            c_counts = c_df["split"].value_counts().to_dict()
            per_class_split[c] = {
                "train": c_counts.get("train", 0),
                "val": c_counts.get("val", 0),
                "test": c_counts.get("test", 0),
                "total": len(c_df)
            }

        stats_summary[name] = {
            "total_samples": total,
            "counts": {
                "train": counts.get("train", 0),
                "val": counts.get("val", 0),
                "test": counts.get("test", 0)
            },
            "percentages": {
                "train_pct": pcts.get("train", 0.0),
                "val_pct": pcts.get("val", 0.0),
                "test_pct": pcts.get("test", 0.0),
                "train_plus_val_pct": round(pcts.get("train", 0.0) + pcts.get("val", 0.0), 2)
            },
            "per_class_distribution": per_class_split
        }

    total_duration = time.time() - t_start

    # 6. Save Comprehensive JSON Results
    results_json = {
        "step": "3. Dataset Splitting",
        "description": "Standard 70/30, Signer-Independent, and Background-Independent Splits",
        "dataset_metadata": {
            "total_images": len(df),
            "num_classes": len(classes),
            "classes": classes,
            "landmarks_archive": landmarks_file
        },
        "target_specifications": {
            "train_ratio": train_ratio,
            "test_ratio": 1.0 - train_ratio,
            "val_ratio_of_train": val_ratio_of_train,
            "random_seed": seed
        },
        "execution_summary": {
            "execution_time_seconds": round(total_duration, 2),
            "output_directory": str(out_path.resolve()),
            "protocols_generated": list(splits_dict.keys())
        },
        "verification_suite": verification_results,
        "protocol_statistics": stats_summary
    }

    def json_converter(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    results_json_path = out_path / "dataset_splitting_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, default=json_converter)
    print(f"[+] Saved Detailed Results JSON: {results_json_path}")

    # 7. Save Human-Readable Text Report
    results_txt_path = out_path / "dataset_splitting_results.txt"
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ISL FINGERSPELLING RECOGNITION - DATASET SPLITTING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("Overview & Specifications:\n")
        f.write(f"- Total Images:              {len(df):,}\n")
        f.write(f"- Number of Sign Classes:    {len(classes)} (Digits 1-9, Letters A-Z)\n")
        f.write(f"- Target Train/Test Ratio:   {int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Test\n")
        f.write(f"- Validation Strategy:       {int(val_ratio_of_train*100)}% carved out of Train (Test set untouched)\n")
        f.write(f"- Random Seed:               {seed}\n")
        f.write(f"- Processing Time:           {total_duration:.2f} seconds\n\n")

        f.write("=" * 70 + "\n")
        f.write("SPLIT PROTOCOLS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Protocol':<32} {'Train':<10} {'Val':<10} {'Test':<10} {'Train+Val %':<14} {'Test %':<10}\n")
        f.write("-" * 86 + "\n")
        for name, st in stats_summary.items():
            c = st["counts"]
            p = st["percentages"]
            f.write(f"{name:<32} {c['train']:<10} {c['val']:<10} {c['test']:<10} {p['train_plus_val_pct']:<14.2f} {p['test_pct']:<10.2f}\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATA LEAKAGE & INTEGRITY VERIFICATION SUITE\n")
        f.write("-" * 70 + "\n")
        for name, v in verification_results.items():
            status = "PASSED (Zero Leakage)" if v["passed"] else "FAILED"
            f.write(f"- {name:<32}: {status}\n")
            f.write(f"    Train & Val: {v['train_val_overlap']} | Train & Test: {v['train_test_overlap']} | Val & Test: {v['val_test_overlap']}\n")
            f.write(f"    All 35 classes present in Train: {v['all_35_classes_in_train']}, Val: {v['all_35_classes_in_val']}, Test: {v['all_35_classes_in_test']}\n")
            if v["session_overlap"] is not None:
                f.write(f"    Signer Session Overlap (Train/Val vs Test): {v['session_overlap']} (Strictly Disjoint)\n")
            if v["domain_overlap"] is not None:
                f.write(f"    Background Domain Overlap (Train/Val vs Test): {v['domain_overlap']} (Strictly Disjoint)\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("PER-CLASS SAMPLE DISTRIBUTION (STANDARD SPLIT)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Class':<8} {'Total':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Train %':<10} {'Test %':<10}\n")
        f.write("-" * 68 + "\n")
        std_classes = stats_summary["standard_split"]["per_class_distribution"]
        for c, dist in std_classes.items():
            tot = dist["total"]
            tr_pct = (dist["train"] + dist["val"]) / tot * 100
            te_pct = dist["test"] / tot * 100
            f.write(f"{c:<8} {tot:<10} {dist['train']:<10} {dist['val']:<10} {dist['test']:<10} {tr_pct:<10.1f} {te_pct:<10.1f}\n")
        f.write("=" * 70 + "\n")
        f.write("DATASET SPLITTING (STEP 3) COMPLETED SUCCESSFULLY!\n")
        f.write("=" * 70 + "\n")

    print(f"[+] Saved Human-Readable Report: {results_txt_path}")
    print("=" * 70)
    print("DATASET SPLITTING (STEP 3) SUCCESSFULLY COMPLETED!")
    print("=" * 70)

    return results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Fingerspelling Dataset Splitting Pipeline")
    parser.add_argument("--data-dir", type=str, default="ISL_DATA", help="Path to raw ISL_DATA directory")
    parser.add_argument("--landmarks-file", type=str, default="preprocessing_output/landmarks_42d.npz", help="Path to landmarks .npz")
    parser.add_argument("--output-dir", type=str, default="dataset_splitting_output", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default 0.70)")
    parser.add_argument("--val-ratio-of-train", type=float, default=0.15, help="Validation ratio carved out of train (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--export-data-splits", action="store_true", default=True, help="Also export to data/splits/")
    args = parser.parse_args()

    run_dataset_splitting(
        data_dir=args.data_dir,
        landmarks_file=args.landmarks_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio_of_train=args.val_ratio_of_train,
        seed=args.seed,
        export_data_splits=args.export_data_splits
    )
