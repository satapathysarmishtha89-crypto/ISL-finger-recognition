"""
ISL Fingerspelling Recognition - Model Category 2: CNN (EfficientNet-B0)
=========================================================================
Step 4.2 of the Model Implementation Pipeline.

Architecture:
- Backbone: ImageNet-pretrained EfficientNet-B0 (compound scaling, MBConv blocks)
- Input: MediaPipe-cropped 224x224 RGB images
- Feature Extractor: MBConv features -> AdaptiveAvgPool2d -> 1280-D representations
- Classification Head:
    BatchNorm -> Dropout(0.4) -> Linear(1280, 256) ->
    ReLU -> BatchNorm -> Dropout(0.3) -> Linear(256, 35)
- Training Protocol:
    Adam optimizer with label smoothing=0.05, lr=1e-3, ReduceLROnPlateau,
    early stopping on validation loss.
- Evaluation Protocol (Section 5 of Spec):
    Evaluated across all 3 splits:
      1. Standard Split (baseline)
      2. Signer-Independent Split (group disjoint)
      3. Background-Independent Split (domain disjoint)
    Reports: Accuracy, Macro/Weighted F1, Precision, Recall, Cohen's Kappa, MCC,
    Per-Class F1, Confusion Matrices, Generalization Gap, Model Size, and CPU Latency.
- Output Directory:
    model_2_efficientnet_b0_output/
      ├── efficientnet_b0_best.pth
      ├── efficientnet_b0_predictions.npz
      ├── efficientnet_b0_results.json
      ├── efficientnet_b0_results.txt
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
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix
)


class ISLImageDataset(Dataset):
    """
    Loads preprocessed images or cached features.
    """
    def __init__(self, df, img_dir="preprocessing_output/preprocessed_images", transform=None, cached_features=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.cached_features = cached_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = int(row["class_id"])
        landmark_idx = int(row["landmark_index"])

        if self.cached_features is not None:
            feat = torch.tensor(self.cached_features[landmark_idx], dtype=torch.float32)
            return feat, target, landmark_idx

        img_path = self.img_dir / row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, target, landmark_idx


def precompute_efficientnet_features(df_all, img_dir, batch_size=64, device="cpu", cache_path=None):
    """
    Precomputes EfficientNet-B0 1280-D features across all images.
    """
    if cache_path and Path(cache_path).exists():
        print(f"[+] Found cached EfficientNet-B0 features at: {cache_path}")
        data = np.load(cache_path)
        return data["features"]

    print("[*] Precomputing EfficientNet-B0 1280-D features across dataset...")
    weights = models.EfficientNet_B0_Weights.DEFAULT
    backbone = models.efficientnet_b0(weights=weights)
    backbone.classifier = nn.Identity()
    backbone.to(device)
    backbone.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ISLImageDataset(df_all, img_dir=img_dir, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feature_list = [None] * len(df_all)
    t0 = time.time()

    with torch.no_grad():
        for i, (images, targets, l_indices) in enumerate(loader):
            images = images.to(device)
            feats = backbone(images).cpu().numpy()
            for feat, idx in zip(feats, l_indices.numpy()):
                feature_list[idx] = feat

            if (i + 1) % 30 == 0 or (i + 1) == len(loader):
                elapsed = time.time() - t0
                done = (i + 1) * batch_size
                done = min(done, len(df_all))
                speed = done / elapsed if elapsed > 0 else 0
                print(f"    - Extracted {done:,}/{len(df_all):,} images ({speed:.1f} imgs/sec)")

    features_arr = np.array(feature_list, dtype=np.float32)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features_arr)
        print(f"[+] Saved cached features to: {cache_path}")

    return features_arr


def train_head_on_features(train_features, train_labels, val_features, val_labels, num_classes=35, epochs=20, lr=1e-3, device="cpu"):
    """
    Trains classification head on EfficientNet-B0 features.
    """
    head = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(1280),
        nn.Dropout(0.4),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_x = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    val_x = torch.tensor(val_features, dtype=torch.float32)
    val_y = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    best_loss = float("inf")
    best_weights = None

    print(f"[*] Training EfficientNet-B0 Head for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = head(bx)
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
        head.eval()
        with torch.no_grad():
            vx, vy = val_x.to(device), val_y.to(device)
            vout = head(vx)
            val_loss = criterion(vout, vy).item()
            val_preds = vout.argmax(dim=-1)
            val_acc = (val_preds == vy).float().mean().item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

    if best_weights:
        head.load_state_dict(best_weights)
    return head


def evaluate_split(head, features, df_split, split_name, classes, device="cpu"):
    """
    Evaluates model on a specific split and computes all metrics.
    """
    head.eval()
    split_indices = df_split["landmark_index"].to_numpy()
    x = torch.tensor(features[split_indices], dtype=torch.float32).to(device)
    y_true = df_split["class_id"].to_numpy()

    t0 = time.time()
    with torch.no_grad():
        logits = head(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
    eval_time = time.time() - t0
    latency_ms = (eval_time / len(df_split) * 1000) if len(df_split) > 0 else 0.0

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
        "sample_count": len(df_split),
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
        cmap="Purples",
        cbar=True,
        square=True
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Sign Class", fontsize=11)
    plt.ylabel("True Sign Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_efficientnet_pipeline(
    splits_dir="dataset_splitting_output",
    img_dir="preprocessing_output/preprocessed_images",
    output_dir="model_2_efficientnet_b0_output",
    epochs=20,
    batch_size=64,
    device="cpu"
):
    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ISL FINGERSPELLING: MODEL CATEGORY 2 - CNN (EfficientNet-B0)")
    print("=" * 70)
    print(f"Splits Directory:     {splits_dir}")
    print(f"Preprocessed Images:  {img_dir}")
    print(f"Output Destination:   {out_path.resolve()}")
    print(f"Compute Device:       {device.upper()}")
    print(f"Training Epochs:      {epochs}")
    print("=" * 70)

    # 1. Load Split DataFrames
    std_df = pd.read_csv(Path(splits_dir) / "standard_split.csv")
    signer_df = pd.read_csv(Path(splits_dir) / "signer_independent_split.csv")
    bg_df = pd.read_csv(Path(splits_dir) / "background_independent_split.csv")
    classes = sorted(list(std_df["class"].unique()))

    # 2. Precompute Backbone Features (or load cache)
    cache_file = out_path / "efficientnet_b0_features_1280d.npz"
    features_1280d = precompute_efficientnet_features(
        std_df,
        img_dir=img_dir,
        batch_size=batch_size,
        device=device,
        cache_path=cache_file
    )

    # 3. Model Architecture Parameters
    weights = models.EfficientNet_B0_Weights.DEFAULT
    backbone = models.efficientnet_b0(weights=weights)
    param_count = sum(p.numel() for p in backbone.parameters())
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

        train_feats = features_1280d[train_df["landmark_index"].to_numpy()]
        train_labels = train_df["class_id"].to_numpy()
        val_feats = features_1280d[val_df["landmark_index"].to_numpy()]
        val_labels = val_df["class_id"].to_numpy()

        head = train_head_on_features(
            train_feats, train_labels,
            val_feats, val_labels,
            num_classes=len(classes),
            epochs=epochs,
            device=device
        )

        if key == "standard":
            # Save standard model weights
            model_weights_path = out_path / "efficientnet_b0_best.pth"
            torch.save(head.state_dict(), model_weights_path)
            print(f"[+] Saved Standard Model Weights to: {model_weights_path}")

            # Compute standard validation probabilities (needed for late fusion weight tuning)
            val_res, val_probs, val_preds, val_y = evaluate_split(head, features_1280d, val_df, "standard_val", classes, device=device)
            probs_dict["standard_val_probs"] = val_probs
            probs_dict["standard_val_y_true"] = val_y

        res, probs, preds, y_true = evaluate_split(head, features_1280d, test_df, key, classes, device=device)
        eval_results[key] = res
        probs_dict[f"{key}_probs"] = probs
        probs_dict[f"{key}_preds"] = preds
        probs_dict[f"{key}_y_true"] = y_true
        probs_dict[f"{key}_landmark_indices"] = test_df["landmark_index"].to_numpy()

        # Save confusion matrix plot
        cm_path = out_path / f"confusion_matrix_{key}.png"
        plot_confusion_matrix(y_true, preds, classes, f"EfficientNet-B0: {title}", cm_path)

        print(f"[*] {title}:")
        print(f"    - Accuracy: {res['accuracy']*100:.2f}% | Macro F1: {res['macro_f1']:.4f} | Weighted F1: {res['weighted_f1']:.4f}")
        print(f"    - Macro Precision: {res['macro_precision']*100:.2f}% | Macro Recall: {res['macro_recall']*100:.2f}%")
        print(f"    - Cohen's Kappa: {res['cohen_kappa']:.4f} | MCC: {res['mcc']:.4f}")
        print(f"    - Top Confusion: {res['top_confused_pairs'][:3]}")

    # Save prediction probabilities for Late Fusion (Step 4.4)
    pred_probs_file = out_path / "efficientnet_b0_predictions.npz"
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
        "model_name": "CNN (EfficientNet-B0)",
        "backbone": "EfficientNet-B0 (ImageNet Pretrained)",
        "parameters_count": param_count,
        "model_size_mb": round(model_size_mb, 2),
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

    json_path = out_path / "efficientnet_b0_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json_results, f, indent=2, default=json_converter)
    print(f"[+] Saved Detailed Results JSON: {json_path}")

    # 8. Save Formatted Text Report
    txt_path = out_path / "efficientnet_b0_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("ISL FINGERSPELLING: MODEL EVALUATION REPORT - CNN (EfficientNet-B0)\n")
        f.write("=" * 75 + "\n\n")
        f.write("Model Specification:\n")
        f.write(f"- Architecture:          EfficientNet-B0 (MBConv + Swish + Squeeze-and-Excitation)\n")
        f.write(f"- Pretrained Weights:    ImageNet-1K Default\n")
        f.write(f"- Total Parameters:      {param_count:,} ({model_size_mb:.2f} MB)\n")
        f.write(f"- Total Execution Time:  {total_duration:.2f} seconds\n")
        f.write(f"- CPU Inference Latency: {eval_results['standard']['inference_latency_ms_per_image']:.2f} ms/image\n\n")

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
        f.write("Note: A smaller drop indicates higher real-world invariant robustness.\n\n")

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
    print("STEP 4.2: CNN (EfficientNet-B0) COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return full_json_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Fingerspelling - Step 4.2: EfficientNet-B0 CNN")
    parser.add_argument("--splits-dir", type=str, default="dataset_splitting_output", help="Directory with split CSVs")
    parser.add_argument("--img-dir", type=str, default="preprocessing_output/preprocessed_images", help="Preprocessed images folder")
    parser.add_argument("--output-dir", type=str, default="model_2_efficientnet_b0_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for classification head")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_efficientnet_pipeline(
        splits_dir=args.splits_dir,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
