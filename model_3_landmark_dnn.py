"""
ISL Fingerspelling Recognition - Model Category 3: Landmark DNN (42-D)
======================================================================
Step 4.3 of the Model Implementation Pipeline.

Architecture:
- Input: 42-D translation- and scale-normalized MediaPipe hand landmark vectors
- Network:
    Linear(42, 128) -> BatchNorm1d -> ReLU -> Dropout(0.3) ->
    Linear(128, 64) -> BatchNorm1d -> ReLU -> Dropout(0.3) ->
    Linear(64, 35)
- Augmentation (train only):
    Gaussian jitter (std=0.01), light 2D rotation (+-10 deg), random scaling (0.95-1.05)
- Evaluation Protocol (Section 5 of Spec):
    Evaluated across all 3 splits:
      1. Standard Split (baseline)
      2. Signer-Independent Split (group disjoint)
      3. Background-Independent Split (domain disjoint)
    Reports: Accuracy, Macro/Weighted F1, Precision, Recall, Cohen's Kappa, MCC,
    Per-Class F1, Confusion Matrices, Generalization Gap, Model Size, and CPU Latency.
- Output Directory:
    model_3_landmark_dnn_output/
      ├── landmark_dnn_best.pth
      ├── landmark_dnn_predictions.npz
      ├── landmark_dnn_results.json
      ├── landmark_dnn_results.txt
      ├── confusion_matrix_standard.png
      ├── confusion_matrix_signer_independent.png
      └── confusion_matrix_background_independent.png
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix
)


class LandmarkAugmentDataset(Dataset):
    """
    Dataset for 42-D landmarks with optional on-the-fly geometric augmentations.
    """
    def __init__(self, features, labels, landmark_indices, augment=False):
        self.features = features[landmark_indices]
        self.labels = labels
        self.indices = landmark_indices
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].copy() # 42-D
        target = self.labels[idx]

        if self.augment:
            # Reshape to (21, 2)
            coords = feat.reshape(21, 2)

            # 1. Random small rotation (-10 to +10 degrees)
            angle = np.radians(np.random.uniform(-10, 10))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            coords = np.dot(coords, rot_matrix)

            # 2. Random scale jitter (0.95 to 1.05)
            scale = np.random.uniform(0.95, 1.05)
            coords *= scale

            # 3. Add small Gaussian noise
            coords += np.random.normal(0, 0.01, size=coords.shape)
            feat = coords.flatten().astype(np.float32)

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(target, dtype=torch.long), self.indices[idx]


class LandmarkDNN(nn.Module):
    """
    Deep Neural Network for 42-D normalized landmark feature classification.
    """
    def __init__(self, input_dim=42, num_classes=35, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_landmark_dnn(
    train_dataset,
    val_dataset,
    num_classes=35,
    epochs=40,
    lr=1e-3,
    batch_size=64,
    device="cpu"
):
    """
    Trains the Landmark DNN with early stopping and learning rate scheduling.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LandmarkDNN(input_dim=42, num_classes=num_classes, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_loss = float("inf")
    best_weights = None

    print(f"[*] Training Landmark DNN for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = out.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vx, vy, _ in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vout = model(vx)
                val_loss += criterion(vout, vy).item() * vx.size(0)
                vpreds = vout.argmax(dim=-1)
                val_correct += (vpreds == vy).sum().item()
                val_total += vx.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def evaluate_split(model, dataset, split_name, classes, device="cpu"):
    """
    Evaluates model on a specific split and computes all metrics.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    all_probs = []
    all_preds = []
    all_y = []
    all_indices = []

    t0 = time.time()
    with torch.no_grad():
        for bx, by, b_indices in loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_preds.append(preds)
            all_y.append(by.numpy())
            all_indices.extend(b_indices.numpy())
    eval_time = time.time() - t0

    probs = np.vstack(all_probs)
    preds = np.concatenate(all_preds)
    y_true = np.concatenate(all_y)
    latency_ms = (eval_time / len(preds) * 1000) if len(preds) > 0 else 0.0

    acc = accuracy_score(y_true, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)

    # Per-class metrics
    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(y_true, preds, average=None, zero_division=0)
    per_class_table = {}
    for i, c in enumerate(classes):
        per_class_table[c] = {
            "precision": round(float(class_p[i]), 4),
            "recall": round(float(class_r[i]), 4),
            "f1": round(float(class_f1[i]), 4),
            "support": int(class_supp[i])
        }

    # Confusion matrix & Top-10 confused pairs
    cm = confusion_matrix(y_true, preds, labels=range(len(classes)))
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

    results = {
        "split_name": split_name,
        "sample_count": len(preds),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(wt_f1), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "cohen_kappa": round(float(kappa), 4),
        "mcc": round(float(mcc), 4),
        "inference_latency_ms_per_image": round(float(latency_ms), 3),
        "top_confused_pairs": top_confused,
        "per_class_metrics": per_class_table
    }

    return results, probs, preds, y_true


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
        cmap="Greens",
        cbar=True,
        square=True
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Sign Class", fontsize=11)
    plt.ylabel("True Sign Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_landmark_dnn_pipeline(
    splits_dir="dataset_splitting_output",
    landmarks_file="preprocessing_output/landmarks_42d.npz",
    output_dir="model_3_landmark_dnn_output",
    epochs=40,
    batch_size=64,
    device="cpu"
):
    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ISL FINGERSPELLING: MODEL CATEGORY 3 - LANDMARK DNN (42-D)")
    print("=" * 70)
    print(f"Splits Directory:     {splits_dir}")
    print(f"Landmarks Archive:    {landmarks_file}")
    print(f"Output Destination:   {out_path.resolve()}")
    print(f"Compute Device:       {device.upper()}")
    print(f"Training Epochs:      {epochs}")
    print("=" * 70)

    # 1. Load Preprocessed 42-D Landmarks
    print(f"[+] Loading 42-D Landmarks from: {landmarks_file}")
    npz = np.load(landmarks_file)
    features_42d = npz["features"] # (17292, 42)
    print(f"[+] Landmarks matrix loaded: {features_42d.shape}")

    # 2. Load Split DataFrames
    std_df = pd.read_csv(Path(splits_dir) / "standard_split.csv")
    signer_df = pd.read_csv(Path(splits_dir) / "signer_independent_split.csv")
    bg_df = pd.read_csv(Path(splits_dir) / "background_independent_split.csv")
    classes = sorted(list(std_df["class"].unique()))

    # Model architecture stats
    dummy_model = LandmarkDNN(input_dim=42, num_classes=len(classes), dropout=0.3)
    param_count = sum(p.numel() for p in dummy_model.parameters())
    model_size_mb = (param_count * 4) / (1024 * 1024)

    # 4 & 5. Multi-Split Independent Training & Evaluation Protocol (Section 5 Specification)
    print("\n" + "=" * 70)
    print("MULTI-SPLIT INDEPENDENT TRAINING & EVALUATION PROTOCOL")
    print("=" * 70)

    eval_results = {}
    probs_dict = {}

    protocols = [
        ("standard", std_df, "Standard Split (Held-Out Test)"),
        ("signer_independent", signer_df, "Signer-Independent Split (Held-Out Test)"),
        ("background_independent", bg_df, "Background-Independent Split (Held-Out Test)")
    ]

    for key, split_df, title in protocols:
        print(f"\n[*] Training & Evaluating Protocol: {title} ({key})")
        train_df = split_df[split_df["split"] == "train"]
        val_df = split_df[split_df["split"] == "val"]
        test_df = split_df[split_df["split"] == "test"]

        train_ds = LandmarkAugmentDataset(
            features_42d,
            train_df["class_id"].to_numpy(),
            train_df["landmark_index"].to_numpy(),
            augment=True
        )
        val_ds = LandmarkAugmentDataset(
            features_42d,
            val_df["class_id"].to_numpy(),
            val_df["landmark_index"].to_numpy(),
            augment=False
        )
        test_ds = LandmarkAugmentDataset(
            features_42d,
            test_df["class_id"].to_numpy(),
            test_df["landmark_index"].to_numpy(),
            augment=False
        )

        model = train_landmark_dnn(
            train_ds,
            val_ds,
            num_classes=len(classes),
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )

        if key == "standard":
            # Save standard model weights
            model_weights_path = out_path / "landmark_dnn_best.pth"
            torch.save(model.state_dict(), model_weights_path)
            print(f"[+] Saved Standard Model Weights to: {model_weights_path}")

            # Compute standard validation probabilities
            val_res, val_probs, val_preds, val_y = evaluate_split(model, val_ds, "standard_val", classes, device=device)
            probs_dict["standard_val_probs"] = val_probs
            probs_dict["standard_val_y_true"] = val_y

        res, probs, preds, y_true = evaluate_split(model, test_ds, key, classes, device=device)
        eval_results[key] = res
        probs_dict[f"{key}_probs"] = probs
        probs_dict[f"{key}_preds"] = preds
        probs_dict[f"{key}_y_true"] = y_true
        probs_dict[f"{key}_landmark_indices"] = test_df["landmark_index"].to_numpy()

        # Save confusion matrix plot
        cm_path = out_path / f"confusion_matrix_{key}.png"
        plot_confusion_matrix(y_true, preds, classes, f"Landmark DNN: {title}", cm_path)

        print(f"[*] {title}:")
        print(f"    - Accuracy: {res['accuracy']*100:.2f}% | Macro F1: {res['macro_f1']:.4f} | Weighted F1: {res['weighted_f1']:.4f}")
        print(f"    - Macro Precision: {res['macro_precision']*100:.2f}% | Macro Recall: {res['macro_recall']*100:.2f}%")
        print(f"    - Cohen's Kappa: {res['cohen_kappa']:.4f} | MCC: {res['mcc']:.4f}")
        print(f"    - Top Confusion: {res['top_confused_pairs'][:3]}")

    # Save prediction probabilities for Late Fusion (Step 4.4)
    pred_probs_file = out_path / "landmark_dnn_predictions.npz"
    np.savez_compressed(pred_probs_file, **probs_dict)
    print(f"\n[+] Saved Prediction Probabilities for Fusion: {pred_probs_file}")

    # 6. Generalization Gap Calculation
    std_acc = eval_results["standard"]["accuracy"]
    signer_acc = eval_results["signer_independent"]["accuracy"]
    bg_acc = eval_results["background_independent"]["accuracy"]

    gap_signer = round(float(std_acc - signer_acc), 4)
    gap_bg = round(float(std_acc - bg_acc), 4)

    print("\n" + "=" * 70)
    print("GENERALIZATION GAP ANALYSIS (KEY DIFFERENTIATOR FROM PRIOR WORK)")
    print("=" * 70)
    print(f"Standard Accuracy:              {std_acc*100:.2f}%")
    print(f"Signer-Independent Accuracy:    {signer_acc*100:.2f}% (Drop: {gap_signer*100:.2f}%)")
    print(f"Background-Independent Accuracy:{bg_acc*100:.2f}% (Drop: {gap_bg*100:.2f}%)")
    print("=" * 70)

    total_duration = time.time() - t_start

    # 7. Save JSON Metrics
    full_json_results = {
        "model_name": "Landmark DNN (42-D)",
        "features": "MediaPipe Normalized Landmarks (42-D)",
        "parameters_count": param_count,
        "model_size_mb": round(model_size_mb, 4),
        "total_runtime_seconds": round(total_duration, 2),
        "generalization_gap": {
            "standard_acc": std_acc,
            "signer_independent_acc": signer_acc,
            "background_independent_acc": bg_acc,
            "signer_gap": gap_signer,
            "background_gap": gap_bg
        },
        "protocols": eval_results
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

    json_path = out_path / "landmark_dnn_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json_results, f, indent=2, default=json_converter)
    print(f"[+] Saved Detailed Results JSON: {json_path}")

    # 8. Save Formatted Text Report
    txt_path = out_path / "landmark_dnn_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("ISL FINGERSPELLING: MODEL EVALUATION REPORT - LANDMARK DNN (42-D)\n")
        f.write("=" * 75 + "\n\n")
        f.write("Model Specification:\n")
        f.write(f"- Architecture:          Dense(42, 128) -> BN -> ReLU -> Dense(64) -> Dense(35)\n")
        f.write(f"- Feature Representation:21 Landmarks x (x, y) = 42-D normalized coordinates\n")
        f.write(f"- Total Parameters:      {param_count:,} ({model_size_mb:.4f} MB - Ultra-Lightweight)\n")
        f.write(f"- Total Execution Time:  {total_duration:.2f} seconds\n")
        f.write(f"- CPU Inference Latency: {eval_results['standard']['inference_latency_ms_per_image']:.3f} ms/image\n\n")

        f.write("=" * 75 + "\n")
        f.write("MULTI-SPLIT BENCHMARK SUMMARY\n")
        f.write("=" * 75 + "\n")
        f.write(f"{'Split Protocol':<28} {'Accuracy':<10} {'Macro-F1':<10} {'Wt-F1':<10} {'Kappa':<10} {'MCC':<10}\n")
        f.write("-" * 75 + "\n")
        for key in ["standard", "signer_independent", "background_independent"]:
            r = eval_results[key]
            f.write(f"{r['split_name']:<28} {r['accuracy']*100:<10.2f} {r['macro_f1']:<10.4f} {r['weighted_f1']:<10.4f} {r['cohen_kappa']:<10.4f} {r['mcc']:<10.4f}\n")
        f.write("=" * 75 + "\n\n")

        f.write("GENERALIZATION GAP ANALYSIS (ROBUSTNESS METRIC)\n")
        f.write("-" * 75 + "\n")
        f.write(f"- Standard vs Signer-Independent Gap:     {gap_signer*100:+.2f}%\n")
        f.write(f"- Standard vs Background-Independent Gap: {gap_bg*100:+.2f}%\n")
        f.write("Note: Landmarks are invariant to background and appearance shifts by definition.\n\n")

        f.write("=" * 75 + "\n")
        f.write("TOP CONFUSED SIGN PAIRS (STANDARD SPLIT)\n")
        f.write("-" * 75 + "\n")
        for pair in eval_results["standard"]["top_confused_pairs"]:
            f.write(f"  True Sign '{pair['true_class']}' misclassified as '{pair['pred_class']}': {pair['count']} times\n")
        f.write("\n" + "=" * 75 + "\n")
        f.write("PER-CLASS F1 BREAKDOWN (STANDARD TEST SET)\n")
        f.write("=" * 75 + "\n")
        f.write(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 54 + "\n")
        for c, m in eval_results["standard"]["per_class_metrics"].items():
            f.write(f"{c:<8} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10}\n")
        f.write("=" * 75 + "\n\n")

        f.write("=" * 75 + "\n")
        f.write("TOP CONFUSED SIGN PAIRS (SIGNER-INDEPENDENT SPLIT)\n")
        f.write("-" * 75 + "\n")
        for pair in eval_results["signer_independent"]["top_confused_pairs"]:
            f.write(f"  True Sign '{pair['true_class']}' misclassified as '{pair['pred_class']}': {pair['count']} times\n")
        f.write("\n" + "=" * 75 + "\n")
        f.write("PER-CLASS F1 BREAKDOWN (SIGNER-INDEPENDENT TEST SET)\n")
        f.write("=" * 75 + "\n")
        f.write(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 54 + "\n")
        for c, m in eval_results["signer_independent"]["per_class_metrics"].items():
            f.write(f"{c:<8} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10}\n")
        f.write("=" * 75 + "\n")

    print(f"[+] Saved Human-Readable Report: {txt_path}")
    print("=" * 70)
    print("STEP 4.3: LANDMARK DNN (42-D) COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return full_json_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Fingerspelling - Step 4.3: Landmark DNN")
    parser.add_argument("--splits-dir", type=str, default="dataset_splitting_output", help="Directory with split CSVs")
    parser.add_argument("--landmarks-file", type=str, default="preprocessing_output/landmarks_42d.npz", help="Landmarks .npz")
    parser.add_argument("--output-dir", type=str, default="model_3_landmark_dnn_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_landmark_dnn_pipeline(
        splits_dir=args.splits_dir,
        landmarks_file=args.landmarks_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
