"""
Plot: Generalisation Drop Relative to the Standard Split
Evaluates the generalization gap across all six models:
  Δ_signer = Accuracy_standard - Accuracy_signer_independent
  Δ_bg     = Accuracy_standard - Accuracy_background_independent
A lower drop indicates superior out-of-distribution (OOD) robustness.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_generalisation_drop_diagram():
    # Academic publication styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    # Models
    models = [
        "CNN\n(MobileNetV2)",
        "CNN\n(EfficientNet-B0)",
        "Landmark DNN\n(42-D)",
        "Hybrid Fusion\n(Weighted Avg)",
        "Hybrid Fusion\n(Stacking)",
        "Hybrid Fusion\n(Confidence-Aware)"
    ]

    # Exact Generalisation Drops (%) [Standard Acc - OOD Acc]
    # MobileNetV2:  99.94 - 97.78 = 2.16; 99.94 - 96.50 = 3.44
    # EfficientNet: 99.94 - 99.42 = 0.52; 99.94 - 98.64 = 1.30
    # Landmark DNN: 97.59 - 68.52 = 29.07; 97.59 - 71.93 = 25.66
    # Weighted Avg: 99.96 - 95.62 = 4.34; 99.96 - 95.62 = 4.34
    # Stacking:     99.96 - 94.54 = 5.42; 99.96 - 95.06 = 4.90
    # Conf-Aware:   99.92 - 93.72 = 6.20; 99.92 - 93.93 = 5.99
    signer_drop = [2.16, 0.52, 29.07, 4.34, 5.42, 6.20]
    bg_drop = [3.44, 1.30, 25.66, 4.34, 4.90, 5.99]

    x = np.arange(len(models))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300)

    # Color palette
    c_signer = '#c0392b'  # Strong Crimson Red
    c_bg = '#7d3c98'      # Elegant Amethyst Purple

    # Create bars
    rects1 = ax.bar(x - bar_width / 2, signer_drop, bar_width,
                    label='Δ_Signer: Signer-Independent Drop (Unseen Signers)',
                    color=c_signer, edgecolor='#78281f', linewidth=1.2, zorder=3)
    rects2 = ax.bar(x + bar_width / 2, bg_drop, bar_width,
                    label='Δ_BG: Background-Independent Drop (Unseen Environments)',
                    color=c_bg, edgecolor='#4a235a', linewidth=1.2, zorder=3)

    # Add data labels
    def add_labels(rects, text_color):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'+{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9.5, fontweight='bold', color=text_color)

    add_labels(rects1, '#78281f')
    add_labels(rects2, '#4a235a')

    # Reference 5% threshold line (Desirable robustness boundary)
    ax.axhline(5.0, color='#27ae60', linestyle='--', linewidth=1.3, alpha=0.85, zorder=2)
    ax.text(-0.45, 5.6, '5% Robustness Boundary (Desirable)', color='#1e8449',
            fontsize=9, fontweight='bold', fontstyle='italic', ha='left', zorder=5)

    # Annotations / Insights
    # 1. EfficientNet-B0: Most robust
    ax.annotate('Most Robust Architecture\nMinimal generalization drops:\nΔ_Signer: +0.52% | Δ_BG: +1.30%',
                xy=(1, 1.30), xytext=(0.55, 12.0),
                arrowprops=dict(facecolor='#145a32', shrink=0.08, width=1.5, headwidth=6, headlength=7),
                bbox=dict(boxstyle="round,pad=0.45", fc="#d4efdf", ec="#27ae60", lw=1.2),
                fontsize=8.5, color='#145a32', fontweight='bold', zorder=4)

    # 2. Landmark DNN: Severe vulnerability
    ax.annotate('Severe Representation Vulnerability\nCoordinate shifts from anatomical variations\n(Finger bone length, joint spacing: Δ = +29.07%)',
                xy=(2 - bar_width / 2, 29.07), xytext=(1.35, 33.5),
                arrowprops=dict(facecolor='#922b21', shrink=0.08, width=1.5, headwidth=6, headlength=7),
                bbox=dict(boxstyle="round,pad=0.45", fc="#fadbd8", ec="#e74c3c", lw=1.2),
                fontsize=8.5, color='#78281f', fontweight='bold', zorder=4)

    # 3. Hybrid Fusion: Resilience stabilization
    ax.annotate('Multi-Modal Stabilization\nCombines landmark geometry with CNN visual invariance;\nPrevents the 29% pure landmark collapse (Drop limited to 4.34%)',
                xy=(3, 4.34), xytext=(2.9, 15.0),
                arrowprops=dict(facecolor='#1f4e78', shrink=0.08, width=1.5, headwidth=6, headlength=7),
                bbox=dict(boxstyle="round,pad=0.45", fc="#d6eaf8", ec="#2980b9", lw=1.2),
                fontsize=8.5, color='#1b4f72', fontweight='bold', zorder=4)

    # Titles and axis labels
    ax.set_title('Generalisation Drop Relative to the Standard Split',
                 fontsize=16, fontweight='bold', pad=25, color='#111111')
    ax.text(0.5, 1.02,
            'Quantifying Out-of-Distribution Generalization Gap: Δ = Accuracy_Standard - Accuracy_OOD  [Lower Drop = Higher Robustness]',
            transform=ax.transAxes, ha='center', fontsize=11, color='#555555', style='italic')

    ax.set_ylabel('Generalisation Drop (%) [Lower is More Robust]', fontsize=13, fontweight='bold', labelpad=12, color='#111111')
    ax.set_xlabel('Evaluated Model Architecture', fontsize=13, fontweight='bold', labelpad=14, color='#111111')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight='bold', color='#222222')

    # Y-axis bounds and formatting
    ax.set_ylim(0, 38)
    ax.set_yticks(np.arange(0, 36, 5))
    ax.yaxis.set_major_formatter('{x:.0f}%')
    ax.tick_params(axis='y', labelsize=10.5)

    # Grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc', zorder=1)
    ax.grid(axis='x', visible=False)

    # Legend placed in upper-right (clean and compact)
    legend = ax.legend(loc='upper right', frameon=True, framealpha=0.95, facecolor='#ffffff',
                       edgecolor='#bbbbbb', fontsize=10, title="Protocol Generalisation Drop",
                       title_fontproperties={'weight': 'bold', 'size': 10.5})
    legend.get_frame().set_boxstyle('round,pad=0.5')

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    # Save to both British and American spellings and all target directories
    output_filenames = [
        "generalisation_drop_relative_to_the_standard_split.png",
        "generalisation_drop_relative_to_the_standard_split.pdf",
        "generalization_drop_relative_to_the_standard_split.png",
        "generalization_drop_relative_to_the_standard_split.pdf",
        "model_6_evaluation_metrics_output/generalisation_drop_relative_to_the_standard_split.png",
        "results/generalisation_drop_relative_to_the_standard_split.png"
    ]

    base_dir = Path(__file__).parent.resolve()
    for rel_path in output_filenames:
        target_path = base_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved diagram to: {target_path}")

    plt.close()
    print("[+] Generalisation drop diagram generation completed successfully.")

if __name__ == "__main__":
    generate_generalisation_drop_diagram()
