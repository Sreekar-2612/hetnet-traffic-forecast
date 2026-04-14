"""Spatial error maps and optional attention placeholders (Phase 4)."""
from __future__ import annotations

import numpy as np


def plot_spatial_error_map(
    coords: np.ndarray,
    per_node_mae: np.ndarray,
    out_path: str = "spatial_error_map.png",
    title: str = "Per-node MAE (test)",
) -> None:
    """
    coords: (N, 2) — lon/lat or layout x/y
    per_node_mae: (N,) mean absolute error per cell
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=per_node_mae, cmap="magma", s=40, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="MAE")
    ax.set_title(title)
    ax.set_xlabel("x / lon")
    ax.set_ylabel("y / lat")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def per_node_mae_from_results(te_pred: np.ndarray, te_true: np.ndarray) -> np.ndarray:
    """(S, H, N) -> (N,) mean |error| over samples and horizon."""
    return np.abs(te_pred - te_true).mean(axis=(0, 1))


def plot_error_map_from_npz(npz_path: str = "results.npz", out_path: str = "spatial_error_map.png") -> None:
    d = np.load(npz_path, allow_pickle=True)
    if "coords" not in d.files:
        print("No coords in npz; cannot plot spatial map.")
        return
    mae_n = per_node_mae_from_results(d["pred"], d["true"])
    plot_spatial_error_map(np.asarray(d["coords"]), mae_n, out_path=out_path)


def attention_heatmap_stub(attn_matrix: np.ndarray, out_path: str = "attention_stub.png") -> None:
    """Save a generic heatmap (placeholder until GAT attention weights are exported)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title("Attention weights (stub)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")
