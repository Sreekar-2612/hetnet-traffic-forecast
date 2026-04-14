import matplotlib.pyplot as plt
import numpy as np


def plot_tier_predictions(file_path: str = "results.npz", out_path: str = "tier_predictions.png") -> None:
    try:
        data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Results file {file_path} not found. Run train.py first.")
        return

    te_pred = data["pred"]
    te_true = data["true"]
    macro_idx = data["macro"]
    pico_idx = data["pico"]
    femto_idx = data["femto"]

    if macro_idx.size == 0 or pico_idx.size == 0 or femto_idx.size == 0:
        print("Tier indices empty (e.g. homo-only run). Plotting first three nodes instead.")
        plot_indices = [0, min(1, te_pred.shape[2] - 1), min(2, te_pred.shape[2] - 1)]
        labels = ["Node 0", "Node 1", "Node 2"]
        colors = ["#FF4B4B", "#1C83E1", "#00D67D"]
    else:
        plot_indices = [macro_idx[0], pico_idx[0], femto_idx[0]]
        labels = ["Macro BS", "Pico BS", "Femto BS"]
        colors = ["#FF4B4B", "#1C83E1", "#00D67D"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    h = 0
    for i, (idx, label, color) in enumerate(zip(plot_indices, labels, colors)):
        ax = axes[i]
        truth = te_true[:100, h, idx]
        pred = te_pred[:100, h, idx]
        ax.plot(truth, label="Ground Truth", color="black", alpha=0.5, linewidth=2)
        ax.plot(pred, "--", label="TASTF Forecast", color=color, linewidth=2)
        ax.set_title(f"{label} Activity (Cell {idx})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (test samples)", fontsize=12)
        ax.set_ylabel("Normalized Activity", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_tier_predictions()
