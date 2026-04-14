"""Train TASTF (hetero/homo, SAGE/GAT, optional temporal + probabilistic head)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines import eval_np, naive_persistence, print_baseline_report, ridge_forecast
from data_loader import load_telecom_italia
from graph_builder import build_hetero_graph, build_homogeneous_graph
from metrics import inverse_metrics, mae_rmse, smape
from model import TASTF, TASTFHomo, gaussian_nll
from paths import resolve_wireless_dataset_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TASTF training")
    ap.add_argument("--data-dir", default=None, help="Override data directory (default: resolve Wireless Dataset)")
    ap.add_argument("--geojson", default=None, help="Path to milano-grid.geojson for KNN coordinates")
    ap.add_argument("--ablation", choices=["hetero", "homo"], default="hetero")
    ap.add_argument("--use-gat", action="store_true", help="Use GATConv instead of SAGEConv")
    ap.add_argument("--no-temporal", action="store_true", help="Disable hour/dow sin-cos features")
    ap.add_argument("--probabilistic", action="store_true", help="Gaussian head + NLL loss")
    ap.add_argument("--baselines-only", action="store_true", help="Only run naive + Ridge baselines")
    ap.add_argument("--run-baselines", action="store_true", help="Also run baselines after training")
    ap.add_argument("--mlflow", action="store_true", help="Log to MLflow (if installed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-6)
    ap.add_argument("--device", default=None)
    return ap


def default_training_args() -> SimpleNamespace:
    """Defaults for notebooks / `run_training(**kwargs)`."""
    return SimpleNamespace(
        data_dir=None,
        geojson=None,
        ablation="hetero",
        use_gat=False,
        no_temporal=False,
        probabilistic=False,
        baselines_only=False,
        run_baselines=False,
        mlflow=False,
        seed=42,
        epochs=50,
        batch_size=32,
        lr=1e-3,
        patience=10,
        min_delta=1e-6,
        device=None,
    )


def run_training(**kwargs) -> None:
    """
    Programmatic entry (e.g. Jupyter). Example:
        run_training(epochs=10, data_dir="/content/Wireless Dataset")
    """
    args = default_training_args()
    for k, v in kwargs.items():
        if not hasattr(args, k):
            raise TypeError(f"Unknown training arg: {k}")
        setattr(args, k, v)
    train_core(args)


def train_core(args: SimpleNamespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Prefer hetnet-traffic-forecast/Wireless Dataset next to this package
        _local = _ROOT.parent / "Wireless Dataset"
        if _local.is_dir() and any(_local.glob("*.txt")):
            data_dir = str(_local.resolve())
        else:
            data_dir = resolve_wireless_dataset_dir(None)
    print(f"DATA_DIR = {data_dir}")

    r = load_telecom_italia(data_dir, random_seed=args.seed)
    split_info = r.split_info
    print("Split info:", json.dumps({k: v for k, v in split_info.items() if k != "files_loaded"}, indent=2))
    print(f"Files loaded: {len(split_info.get('files_loaded', []))} txt files")

    (X_tr, y_tr) = r.train
    (X_val, y_val) = r.val
    (X_te, y_te) = r.test
    scaler = r.scaler
    grid_ids = r.grid_ids

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    time_tr = torch.tensor(r.time_origin_train, dtype=torch.float32)
    time_val = torch.tensor(r.time_origin_val, dtype=torch.float32)
    time_te = torch.tensor(r.time_origin_test, dtype=torch.float32)

    N_CELLS = split_info["n_nodes"]
    split_info["seq_len"]
    HORIZON = split_info["horizon"]

    naive_p = naive_persistence(y_te, X_te)
    ridge_p = ridge_forecast(X_tr, y_tr, X_te)
    print("--- Baselines (test) ---")
    print_baseline_report("Naive persistence", eval_np(naive_p, y_te))
    print_baseline_report("Ridge", eval_np(ridge_p, y_te))

    if args.baselines_only:
        return

    mlflow_run = None
    if args.mlflow:
        try:
            import mlflow

            mlflow.set_experiment("tastf-hetnet")
            mlflow_run = mlflow.start_run()
            mlflow.log_params(
                {
                    "ablation": args.ablation,
                    "use_gat": args.use_gat,
                    "temporal": not args.no_temporal,
                    "probabilistic": args.probabilistic,
                    "seed": args.seed,
                    "data_dir": data_dir,
                }
            )
            mlflow.log_dict(split_info, "split_info.json")
        except ImportError:
            print("MLflow not installed; pip install mlflow to enable --mlflow")
            args.mlflow = False

    mean_mat = X_tr.reshape(-1, N_CELLS)
    hetero, macro_idx, pico_idx, femto_idx, coords = build_hetero_graph(
        mean_mat, k=5, grid_ids=grid_ids, geojson_path=args.geojson
    )
    hetero = hetero.to(device)
    homo_g, _ = build_homogeneous_graph(
        mean_mat, k=5, grid_ids=grid_ids, geojson_path=args.geojson, coords=coords
    )
    homo_g = homo_g.to(device)

    temporal_dim = 0 if args.no_temporal else 4

    if args.ablation == "hetero":
        model = TASTF(
            N=N_CELLS,
            horizon=HORIZON,
            macro_idx=macro_idx,
            pico_idx=pico_idx,
            femto_idx=femto_idx,
            use_gat=args.use_gat,
            temporal_dim=temporal_dim,
            probabilistic=args.probabilistic,
        ).to(device)
        graph = hetero
    else:
        model = TASTFHomo(
            N=N_CELLS,
            horizon=HORIZON,
            use_gat=args.use_gat,
            temporal_dim=temporal_dim,
            probabilistic=args.probabilistic,
        ).to(device)
        graph = homo_g

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4, min_lr=1e-5
    )
    mse = nn.MSELoss()

    best_val = float("inf")
    wait = 0
    batch_size = args.batch_size
    n_train_batches = (len(X_tr_t) + batch_size - 1) // batch_size

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        epoch_loss = 0.0
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_tr_t[idx].to(device)
            yb = y_tr_t[idx].to(device)
            tb = time_tr[idx].to(device) if temporal_dim > 0 else None

            opt.zero_grad()
            if args.probabilistic:
                mu, logv = model(xb, graph, tb)
                loss = gaussian_nll(mu, logv, yb)
            else:
                pred = model(xb, graph, tb)
                loss = mse(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / n_train_batches

        model.eval()
        with torch.no_grad():
            if args.probabilistic:
                mu, _ = model(X_val_t.to(device), graph, time_val.to(device) if temporal_dim > 0 else None)
                val_loss = mse(mu, y_val_t.to(device)).item()
            else:
                val_loss = mse(
                    model(X_val_t.to(device), graph, time_val.to(device) if temporal_dim > 0 else None),
                    y_val_t.to(device),
                ).item()

        sched.step(val_loss)
        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.6f} | Val: {val_loss:.6f}")

        improved = val_loss < best_val - args.min_delta
        if improved:
            best_val = val_loss
            torch.save(model.state_dict(), _ROOT / "tastf_best.pt")
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    ckpt = _ROOT / "tastf_best.pt"
    if ckpt.is_file():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        if args.probabilistic:
            te_mu, _ = model(X_te_t.to(device), graph, time_te.to(device) if temporal_dim > 0 else None)
            te_pred = te_mu.cpu().numpy()
        else:
            te_pred = model(X_te_t.to(device), graph, time_te.to(device) if temporal_dim > 0 else None).cpu().numpy()
    te_true = y_te

    mae, rmse = mae_rmse(te_pred, te_true)
    sm = smape(te_pred, te_true)
    inv_mae, inv_rmse, inv_sm = inverse_metrics(te_pred, te_true, scaler, N_CELLS)

    print("-" * 40)
    print("TEST (normalized space): MAE=%.6f RMSE=%.6f sMAPE=%.2f%%" % (mae, rmse, sm))
    print("TEST (original units):   MAE=%.6f RMSE=%.6f sMAPE=%.2f%%" % (inv_mae, inv_rmse, inv_sm))
    print("-" * 40)

    if args.run_baselines:
        print("(Baselines already printed above.)")

    np.savez(
        _ROOT / "results.npz",
        pred=te_pred,
        true=te_true,
        macro=macro_idx,
        pico=pico_idx,
        femto=femto_idx,
        coords=coords,
        split_info_json=json.dumps(split_info),
        scaler_min_=scaler.min_,
        scaler_scale_=scaler.scale_,
        n_nodes=N_CELLS,
    )

    def _idx_list(idx) -> list:
        a = np.asarray(idx)
        return a.tolist()

    full_ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "ablation": args.ablation,
            "n_nodes": N_CELLS,
            "horizon": HORIZON,
            "use_gat": args.use_gat,
            "temporal_dim": temporal_dim,
            "probabilistic": args.probabilistic,
            "macro_idx": _idx_list(macro_idx),
            "pico_idx": _idx_list(pico_idx),
            "femto_idx": _idx_list(femto_idx),
        },
        "training_args": dict(vars(args)),
        "best_val_loss": best_val,
        "data_dir": data_dir,
        "split_info": split_info,
    }
    ckpt_path = _ROOT / "tastf_checkpoint.pt"
    torch.save(full_ckpt, ckpt_path)
    weights_only = _ROOT / "tastf_model_weights.pt"
    torch.save(model.state_dict(), weights_only)
    print(
        "Saved model for future inference:\n"
        f"  - {weights_only.name}  (state_dict only — load with model.load_state_dict)\n"
        f"  - {ckpt_path.name}       (full checkpoint — weights + config + split metadata)\n"
        f"  - tastf_best.pt          (best val weights during training; same as loaded for test)"
    )

    if args.mlflow and mlflow_run is not None:
        import mlflow

        mlflow.log_metrics(
            {
                "test_mae_norm": mae,
                "test_rmse_norm": rmse,
                "test_smape": sm,
                "test_mae_orig": inv_mae,
                "test_rmse_orig": inv_rmse,
                "val_best": best_val,
            }
        )
        mlflow.end_run()


def train() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    train_core(args)


if __name__ == "__main__":
    train()
