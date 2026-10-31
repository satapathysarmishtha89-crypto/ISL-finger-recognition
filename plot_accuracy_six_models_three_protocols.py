"""
Plot: Accuracy of the Six Models Under the Three Evaluation Protocols
Dataset: 35-Class Indian Sign Language (ISL) Fingerspelling Dataset (17,292 images)
Evaluates 6 model architectures under 3 benchmark protocols:
  1. Standard Stratified 70/30 Split (N_test = 5,188)
  2. Signer-Independent Split (N_test = 5,368, unseen signers)
  3. Background-Independent Split (N_test = 5,368, unseen environments)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_accuracy_diagram():
    # Set publication styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    # Models and performance data
    models = [
        "CNN\n(MobileNetV2)",
        "CNN\n(EfficientNet-B0)",
        "Landmark DNN\n(42-D)",
        "Hybrid Fusion\n(Weighted Avg)",
        "Hybrid Fusion\n(Stacking)",
        "Hybrid Fusion\n(Confidence-Aware)"
    ]

    # Exact accuracies (%)
    std_acc = [99.94, 99.94, 97.59, 99.96, 99.96, 99.92]
    signer_acc = [97.78, 99.42, 68.52, 95.62, 94.54, 93.72]
    bg_acc = [96.50, 98.64, 71.93, 95.62, 95.06, 93.93]

    x = np.arange(len(models))
    bar_width = 0.26

    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300)

    # Color palette: Academic publication palette
    c_std = '#1f4e78'     # Deep Navy Blue
    c_signer = '#d95f02'  # Terracotta / Amber Orange
    c_bg = '#2ca25f'      # Emerald Forest Green

    # Create bars
    rects1 = ax.bar(x - bar_width, std_acc, bar_width, label='Protocol 1: Standard Stratified 70/30 Split (N=5,188)',
                    color=c_std, edgecolor='#0f283c', linewidth=1.2, zorder=3)
    rects2 = ax.bar(x, signer_acc, bar_width, label='Protocol 2: Signer-Independent Split (N=5,368, Unseen Signers)',
                    color=c_signer, edgecolor='#873600', linewidth=1.2, zorder=3)
    rects3 = ax.bar(x + bar_width, bg_acc, bar_width, label='Protocol 3: Background-Independent Split (N=5,368, Unseen Environments)',
                    color=c_bg, edgecolor='#145a32', linewidth=1.2, zorder=3)

    # Helper function to add labels on top of bars
    def add_labels(rects, text_color='#111111'):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9.5, fontweight='bold', color=text_color,
                        rotation=0)

    add_labels(rects1, '#0f283c')
    add_labels(rects2, '#873600')
    add_labels(rects3, '#145a32')

    # Reference line at 100%
    ax.axhline(100.0, color='#888888', linestyle=':', linewidth=1.2, alpha=0.85, zorder=2)
    ax.text(len(models) - 0.45, 100.4, '100% Baseline Limit', color='#666666', fontsize=9, fontstyle='italic', ha='right')

    # Highlight box for Landmark DNN vulnerability
    ax.annotate('Coordinate sensitivity to\ninterpersonal hand anatomy\n(Δ = +29.07% drop)',
                xy=(2, 68.52), xytext=(2.28, 61.2),
                arrowprops=dict(facecolor='#c0392b', shrink=0.1, width=1.5, headwidth=6, headlength=7),
                bbox=dict(boxstyle="round,pad=0.45", fc="#fadbd8", ec="#e74c3c", lw=1.2),
                fontsize=8.5, color='#78281f', fontweight='bold', zorder=4)

    # Highlight box for EfficientNet-B0 robustness
    ax.annotate('Best OOD Robustness\n(Signer: 99.42% | BG: 98.64%)',
                xy=(1 + bar_width, 98.64), xytext=(0.48, 103.5),
                arrowprops=dict(facecolor='#145a32', shrink=0.1, width=1.5, headwidth=6, headlength=7),
                bbox=dict(boxstyle="round,pad=0.45", fc="#d4efdf", ec="#27ae60", lw=1.2),
                fontsize=8.5, color='#145a32', fontweight='bold', zorder=4)

    # Highlight box for Hybrid Fusion top standard
    ax.annotate('Best Standard Accuracy (99.96%)\nOnly 2 errors in 5,188 samples',
                xy=(3 - bar_width, 99.96), xytext=(3.32, 103.5),
                arrowprops=dict(facecolor='#1f4e78', shrink=0.1, width=1.5, headwidth=6, headlength=7),
                bbox=dict(boxstyle="round,pad=0.45", fc="#d6eaf8", ec="#2980b9", lw=1.2),
                fontsize=8.5, color='#1b4f72', fontweight='bold', zorder=4)

    # Titles and labels
    ax.set_title('Accuracy of the Six Models Under the Three Evaluation Protocols',
                 fontsize=16, fontweight='bold', pad=25, color='#111111')
    ax.text(0.5, 1.02, 'Empirical Evaluation on 35-Class Indian Sign Language (ISL) Fingerspelling Dataset (N = 17,292 images)',
            transform=ax.transAxes, ha='center', fontsize=11, color='#555555', style='italic')

    ax.set_ylabel('Test Classification Accuracy (%)', fontsize=13, fontweight='bold', labelpad=12, color='#111111')
    ax.set_xlabel('Evaluated Model Architecture', fontsize=13, fontweight='bold', labelpad=14, color='#111111')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight='bold', color='#222222')

    # Y-axis bounds and formatting
    ax.set_ylim(55, 107.5)
    ax.set_yticks(np.arange(60, 105, 5))
    ax.yaxis.set_major_formatter('{x:.0f}%')
    ax.tick_params(axis='y', labelsize=10.5)

    # Grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc', zorder=1)
    ax.grid(axis='x', visible=False)

    # Legend
    legend = ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='#ffffff',
                       edgecolor='#bbbbbb', fontsize=10.5, title="Evaluation Benchmark Protocols",
                       title_fontproperties={'weight': 'bold', 'size': 11})
    legend.get_frame().set_boxstyle('round,pad=0.5')

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    # Save to required locations
    output_filenames = [
        "accuracy_of_the_six_models_under_the_three_evaluation_protocols.png",
        "accuracy_of_the_six_models_under_the_three_evaluation_protocols.pdf",
        "model_6_evaluation_metrics_output/accuracy_of_the_six_models_under_the_three_evaluation_protocols.png",
        "results/accuracy_of_the_six_models_under_the_three_evaluation_protocols.png"
    ]

    base_dir = Path(__file__).parent.resolve()
    for rel_path in output_filenames:
        target_path = base_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved diagram to: {target_path}")

    plt.close()
    print("[+] Diagram generation completed successfully.")

if __name__ == "__main__":
    generate_accuracy_diagram()
