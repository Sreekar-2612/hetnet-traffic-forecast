"""Load a saved TASTF checkpoint for inference or fine-tuning."""
from __future__ import annotations

from pathlib import Path

import torch

from model import TASTF, TASTFHomo


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Load `tastf_checkpoint.pt` written by train.py.

    Returns
    -------
    model : TASTF or TASTFHomo
    ckpt : dict with 'config', 'split_info', 'best_val_loss', etc.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    c = cfg

    if c["ablation"] == "hetero":
        model = TASTF(
            N=c["n_nodes"],
            horizon=c["horizon"],
            macro_idx=c["macro_idx"],
            pico_idx=c["pico_idx"],
            femto_idx=c["femto_idx"],
            use_gat=c["use_gat"],
            temporal_dim=c["temporal_dim"],
            probabilistic=c["probabilistic"],
        )
    else:
        model = TASTFHomo(
            N=c["n_nodes"],
            horizon=c["horizon"],
            use_gat=c["use_gat"],
            temporal_dim=c["temporal_dim"],
            probabilistic=c["probabilistic"],
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "tastf_checkpoint.pt"
    m, ck = load_model_from_checkpoint(path)
    print("Loaded:", type(m).__name__, "best_val=", ck.get("best_val_loss"))
