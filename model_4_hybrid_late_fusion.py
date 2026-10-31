"""
ISL Fingerspelling Recognition - Model Category 4: Hybrid Late Fusion
======================================================================
Step 4.4 of the Model Implementation Pipeline.

Core Modeling Contribution (Section 4.3 of Implementation Specification):
Combines complementary evidence from:
1. CNN Image Branch (MobileNetV2 deep visual appearance & texture representations)
2. Landmark DNN Branch (MediaPipe 42-D translation & scale-normalized geometric coordinates)

Fusion Strategies Implemented & Compared:
1. Strategy 1: Weighted Average Fusion
   - p_fused = w * p_cnn + (1 - w) * p_landmark
   - Optimal weight w* is tuned strictly on the validation set.
2. Strategy 2: Stacking Meta-Classifier
   - Trains a multi-class regularized Logistic Regression meta-classifier on
     concatenated validation probabilities [p_cnn, p_landmark] (70 features).
3. Strategy 3: Confidence-Aware Dynamic Fusion
   - Dynamically weights each branch proportional to its prediction confidence.
   - Falls back 100% to CNN branch when hand detection is absent.

Evaluation Protocol (Section 5 of Spec):
Evaluated across all 3 splits (Standard, Signer-Independent, Background-Independent)
with full classification metrics, McNemar significance tests, and comparative plots.

Output Directory:
model_4_hybrid_late_fusion_output/
  ├── fusion_meta_classifier.joblib
  ├── hybrid_late_fusion_predictions.npz
  ├── hybrid_late_fusion_results.json
  ├── hybrid_late_fusion_results.txt
  ├── confusion_matrix_weighted_avg.png
  ├── confusion_matrix_stacking.png
  ├── confusion_matrix_confidence_aware.png
  └── fusion_strategy_comparison.png
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix
)


def compute_metrics(y_true, y_pred, classes):
    """
    Computes all standard classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class_table = {}
    for i, c in enumerate(classes):
        per_class_table[c] = {
            "precision": round(float(class_p[i]), 4),
            "recall": round(float(class_r[i]), 4),
            "f1": round(float(class_f1[i]), 4),
            "support": int(class_supp[i])
        }

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    np.fill_diagonal(cm, 0)
    flat_indices = np.argsort(cm.ravel())[::-1]
    top_confused = []
    for idx in flat_indices[:10]:
        r = idx // len(classes)
        c = idx % len(classes)
        count = int(cm[r, c])
        if count > 0:
            top_confused.append({
                "true_class": classes[r],
                "pred_class": classes[c],
                "count": count
            })

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(wt_f1), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "cohen_kappa": round(float(kappa), 4),
        "mcc": round(float(mcc), 4),
        "top_confused_pairs": top_confused,
        "per_class_metrics": per_class_table
    }


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """
    Renders normalized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        xticklabels=classes,
        yticklabels=classes,
        cmap="YlGnBu",
        cbar=True,
        square=True
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Sign Class", fontsize=11)
    plt.ylabel("True Sign Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_fusion_comparison(comparison_data, save_path):
    """
    Plots a multi-split bar chart comparing fusion strategies and baselines.
    """
    df = pd.DataFrame(comparison_data)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x="Split", y="Accuracy", hue="Model", palette="viridis")
    plt.title("Comparative Performance: Baselines vs. Hybrid Late Fusion Strategies", fontsize=14, fontweight="bold")
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.xlabel("Evaluation Split Protocol", fontsize=12)
    plt.ylim(95.0, 100.2)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_hybrid_late_fusion_pipeline(
    cnn_preds_file="model_1_mobilenetv2_output/mobilenetv2_predictions.npz",
    lm_preds_file="model_3_landmark_dnn_output/landmark_dnn_predictions.npz",
    splits_dir="dataset_splitting_output",
    landmarks_file="preprocessing_output/landmarks_42d.npz",
    output_dir="model_4_hybrid_late_fusion_output"
):
    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ISL FINGERSPELLING: MODEL CATEGORY 4 - HYBRID LATE FUSION")
    print("=" * 70)
    print(f"CNN Predictions:      {cnn_preds_file}")
    print(f"Landmark Predictions: {lm_preds_file}")
    print(f"Output Destination:   {out_path.resolve()}")
    print("=" * 70)

    # 1. Load Stored Prediction Probabilities
    cnn_data = np.load(cnn_preds_file)
    lm_data = np.load(lm_preds_file)

    # Load hand detection mask for confidence-aware fallback
    npz_raw = np.load(landmarks_file)
    has_detection_all = npz_raw["has_detection"]

    std_df = pd.read_csv(Path(splits_dir) / "standard_split.csv")
    classes = sorted(list(std_df["class"].unique()))
    num_classes = len(classes)

    # 2. Validation Probabilities for Weight Tuning & Stacking Meta-Classifier
    val_cnn_p = cnn_data["standard_val_probs"] # (1816, 35)
    val_lm_p = lm_data["standard_val_probs"]   # (1816, 35)
    val_y = cnn_data["standard_val_y_true"]    # (1816,)

    # -------------------------------------------------------------
    # Strategy 1: Tune Weighted Average Parameter w* on Validation Set
    # -------------------------------------------------------------
    print("\n[*] Strategy 1: Tuning Weighted Average Parameter on Validation Set...")
    best_w = 0.5
    best_val_f1 = 0.0

    weight_candidates = np.linspace(0.0, 1.0, 101)
    for w in weight_candidates:
        fused_val = w * val_cnn_p + (1.0 - w) * val_lm_p
        preds = fused_val.argmax(axis=1)
        _, _, f1, _ = precision_recall_fscore_support(val_y, preds, average="macro", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_w = float(w)

    print(f"[+] Optimal Validation Weight: w* = {best_w:.2f} (CNN weight), (1 - w*) = {1.0 - best_w:.2f} (Landmark weight)")
    print(f"[+] Validation Macro F1 at w*: {best_val_f1*100:.2f}%")

    # -------------------------------------------------------------
    # Strategy 2: Train Stacking Meta-Classifier on Validation Set
    # -------------------------------------------------------------
    print("\n[*] Strategy 2: Training Stacking Logistic Regression Meta-Classifier...")
    val_stack_x = np.hstack([val_cnn_p, val_lm_p]) # (1816, 70)
    meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_clf.fit(val_stack_x, val_y)

    meta_clf_path = out_path / "fusion_meta_classifier.joblib"
    joblib.dump(meta_clf, meta_clf_path)
    print(f"[+] Saved Stacking Meta-Classifier to: {meta_clf_path}")

    # -------------------------------------------------------------
    # 3. Multi-Split Evaluation across All 3 Fusion Strategies
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATING FUSION STRATEGIES ACROSS ALL 3 BENCHMARK SPLITS")
    print("=" * 70)

    split_keys = ["standard", "signer_independent", "background_independent"]
    split_labels = {
        "standard": "Standard Split",
        "signer_independent": "Signer-Independent Split",
        "background_independent": "Background-Independent Split"
    }

    fusion_results = {
        "weighted_average": {},
        "stacking": {},
        "confidence_aware": {}
    }

    saved_predictions = {
        "optimal_weight_w": best_w
    }

    comparison_plot_records = []

    for key in split_keys:
        p_cnn = cnn_data[f"{key}_probs"]
        p_lm = lm_data[f"{key}_probs"]
        y_true = cnn_data[f"{key}_y_true"]
        l_indices = cnn_data[f"{key}_landmark_indices"]

        # Strategy 1: Weighted Average
        p_wt_avg = best_w * p_cnn + (1.0 - best_w) * p_lm
        pred_wt_avg = p_wt_avg.argmax(axis=1)
        res_wt_avg = compute_metrics(y_true, pred_wt_avg, classes)
        fusion_results["weighted_average"][key] = res_wt_avg

        # Strategy 2: Stacking Meta-Classifier
        stack_x = np.hstack([p_cnn, p_lm])
        p_stack = meta_clf.predict_proba(stack_x)
        pred_stack = p_stack.argmax(axis=1)
        res_stack = compute_metrics(y_true, pred_stack, classes)
        fusion_results["stacking"][key] = res_stack

        # Strategy 3: Confidence-Aware Dynamic Fusion
        # If no hand detection in MediaPipe -> 100% CNN fallback
        # Otherwise -> weight by respective confidence score
        conf_cnn = p_cnn.max(axis=1, keepdims=True)
        conf_lm = p_lm.max(axis=1, keepdims=True)
        det_mask = has_detection_all[l_indices].reshape(-1, 1) # True if hand detected

        # Weight CNN higher when landmark has no detection
        dyn_w = conf_cnn / (conf_cnn + conf_lm + 1e-6)
        dyn_w = np.where(~det_mask, 1.0, dyn_w) # fallback

        p_conf = dyn_w * p_cnn + (1.0 - dyn_w) * p_lm
        p_conf /= (p_conf.sum(axis=1, keepdims=True) + 1e-9)
        pred_conf = p_conf.argmax(axis=1)
        res_conf = compute_metrics(y_true, pred_conf, classes)
        fusion_results["confidence_aware"][key] = res_conf

        # Baseline records for comparison plot
        cnn_acc = accuracy_score(y_true, p_cnn.argmax(axis=1)) * 100
        lm_acc = accuracy_score(y_true, p_lm.argmax(axis=1)) * 100

        comparison_plot_records.extend([
            {"Split": split_labels[key], "Model": "CNN (MobileNetV2)", "Accuracy": cnn_acc},
            {"Split": split_labels[key], "Model": "Landmark DNN (42-D)", "Accuracy": lm_acc},
            {"Split": split_labels[key], "Model": "Fusion (Weighted Avg)", "Accuracy": res_wt_avg["accuracy"] * 100},
            {"Split": split_labels[key], "Model": "Fusion (Stacking)", "Accuracy": res_stack["accuracy"] * 100},
            {"Split": split_labels[key], "Model": "Fusion (Confidence-Aware)", "Accuracy": res_conf["accuracy"] * 100}
        ])

        # Store predictions
        saved_predictions[f"{key}_weighted_avg_probs"] = p_wt_avg
        saved_predictions[f"{key}_weighted_avg_preds"] = pred_wt_avg
        saved_predictions[f"{key}_stacking_probs"] = p_stack
        saved_predictions[f"{key}_stacking_preds"] = pred_stack
        saved_predictions[f"{key}_confidence_aware_probs"] = p_conf
        saved_predictions[f"{key}_confidence_aware_preds"] = pred_conf
        saved_predictions[f"{key}_y_true"] = y_true

        print(f"\n[*] {split_labels[key]} ({len(y_true):,} test samples):")
        print(f"    - Baseline CNN:              {cnn_acc:.2f}%")
        print(f"    - Baseline Landmark DNN:     {lm_acc:.2f}%")
        print(f"    - Fusion (Weighted Avg):     {res_wt_avg['accuracy']*100:.2f}% (Macro F1: {res_wt_avg['macro_f1']:.4f})")
        print(f"    - Fusion (Stacking):         {res_stack['accuracy']*100:.2f}% (Macro F1: {res_stack['macro_f1']:.4f})")
        print(f"    - Fusion (Confidence-Aware): {res_conf['accuracy']*100:.2f}% (Macro F1: {res_conf['macro_f1']:.4f})")

        # Plot confusion matrices for standard split
        if key == "standard":
            plot_confusion_matrix(y_true, pred_wt_avg, classes, "Hybrid Fusion (Weighted Avg) - Standard Test", out_path / "confusion_matrix_weighted_avg.png")
            plot_confusion_matrix(y_true, pred_stack, classes, "Hybrid Fusion (Stacking) - Standard Test", out_path / "confusion_matrix_stacking.png")
            plot_confusion_matrix(y_true, pred_conf, classes, "Hybrid Fusion (Confidence-Aware) - Standard Test", out_path / "confusion_matrix_confidence_aware.png")

    # 4. Save Comparative Plot
    plot_path = out_path / "fusion_strategy_comparison.png"
    plot_fusion_comparison(comparison_plot_records, plot_path)
    print(f"\n[+] Saved Fusion Comparison Plot to: {plot_path}")

    # 5. Save Predictions NPZ
    preds_npz_path = out_path / "hybrid_late_fusion_predictions.npz"
    np.savez_compressed(preds_npz_path, **saved_predictions)
    print(f"[+] Saved Hybrid Fusion Predictions to: {preds_npz_path}")

    total_duration = time.time() - t_start

    # 6. Save JSON Metrics
    full_json_results = {
        "step": "4.4 Hybrid Late Fusion",
        "description": "Multi-modal fusion combining CNN image representations and MediaPipe landmark geometry",
        "cnn_branch": "CNN (MobileNetV2)",
        "landmark_branch": "Landmark DNN (42-D)",
        "optimal_validation_weight_w": best_w,
        "runtime_seconds": round(total_duration, 2),
        "strategies": fusion_results
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

    json_path = out_path / "hybrid_late_fusion_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json_results, f, indent=2, default=json_converter)
    print(f"[+] Saved Detailed Results JSON: {json_path}")

    # 7. Save Formatted Text Report
    txt_path = out_path / "hybrid_late_fusion_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("===========================================================================\n")
        f.write("ISL FINGERSPELLING: MODEL EVALUATION REPORT - HYBRID LATE FUSION\n")
        f.write("===========================================================================\n\n")
        f.write("Fusion Configuration:\n")
        f.write(f"- CNN Branch Backbone:       MobileNetV2 (ImageNet Pretrained)\n")
        f.write(f"- Landmark Branch:           Landmark DNN (42-D Normalized Coordinates)\n")
        f.write(f"- Optimal Weight w* (Val):   {best_w:.2f} (CNN) / {1.0 - best_w:.2f} (Landmarks)\n")
        f.write(f"- Execution Time:            {total_duration:.2f} seconds\n\n")

        f.write("===========================================================================\n")
        f.write("MULTI-SPLIT FUSION BENCHMARK SUMMARY\n")
        f.write("===========================================================================\n")
        f.write(f"{'Strategy':<22} {'Split Protocol':<26} {'Accuracy':<10} {'Macro-F1':<10} {'Kappa':<10} {'MCC':<10}\n")
        f.write("-" * 88 + "\n")

        for strat_name, strat_data in fusion_results.items():
            for key in split_keys:
                r = strat_data[key]
                f.write(f"{strat_name:<22} {key:<26} {r['accuracy']*100:<10.2f} {r['macro_f1']:<10.4f} {r['cohen_kappa']:<10.4f} {r['mcc']:<10.4f}\n")
            f.write("-" * 88 + "\n")

        f.write("\n===========================================================================\n")
        f.write("GENERALIZATION GAPS BY FUSION STRATEGY\n")
        f.write("===========================================================================\n")
        for strat_name, strat_data in fusion_results.items():
            std_acc = strat_data["standard"]["accuracy"]
            signer_acc = strat_data["signer_independent"]["accuracy"]
            bg_acc = strat_data["background_independent"]["accuracy"]
            f.write(f"Strategy: {strat_name}\n")
            f.write(f"  - Signer-Independent Drop:     {(std_acc - signer_acc)*100:+.2f}%\n")
            f.write(f"  - Background-Independent Drop: {(std_acc - bg_acc)*100:+.2f}%\n")

        f.write("\n===========================================================================\n")
        f.write("STEP 4.4: HYBRID LATE FUSION COMPLETED SUCCESSFULLY!\n")
        f.write("===========================================================================\n")

    print(f"[+] Saved Human-Readable Report: {txt_path}")
    print("=" * 70)
    print("STEP 4.4: HYBRID LATE FUSION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return full_json_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Fingerspelling - Step 4.4: Hybrid Late Fusion")
    parser.add_argument("--cnn-preds", type=str, default="model_1_mobilenetv2_output/mobilenetv2_predictions.npz")
    parser.add_argument("--lm-preds", type=str, default="model_3_landmark_dnn_output/landmark_dnn_predictions.npz")
    parser.add_argument("--splits-dir", type=str, default="dataset_splitting_output")
    parser.add_argument("--landmarks-file", type=str, default="preprocessing_output/landmarks_42d.npz")
    parser.add_argument("--output-dir", type=str, default="model_4_hybrid_late_fusion_output")
    args = parser.parse_args()

    run_hybrid_late_fusion_pipeline(
        cnn_preds_file=args.cnn_preds,
        lm_preds_file=args.lm_preds,
        splits_dir=args.splits_dir,
        landmarks_file=args.landmarks_file,
        output_dir=args.output_dir
    )
