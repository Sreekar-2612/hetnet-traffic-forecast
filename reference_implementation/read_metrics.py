"""Print metrics from results.npz (normalized + original units when scaler arrays present)."""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from metrics import inverse_metrics_from_npz, mae_rmse, smape


def main(file_path: str = "results.npz") -> None:
    data = np.load(file_path, allow_pickle=True)
    te_pred = data["pred"]
    te_true = data["true"]

    mae, rmse = mae_rmse(te_pred, te_true)
    sm = smape(te_pred, te_true)
    print("Normalized space (MinMax train fit):")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  sMAPE: {sm:.2f}%")

    if "scaler_min_" in data.files and "scaler_scale_" in data.files:
        inv_mae, inv_rmse, inv_sm = inverse_metrics_from_npz(
            te_pred, te_true, data["scaler_min_"], data["scaler_scale_"]
        )
        print("Original activity units (inverse transform):")
        print(f"  MAE:  {inv_mae:.6f}")
        print(f"  RMSE: {inv_rmse:.6f}")
        print(f"  sMAPE: {inv_sm:.2f}%")
    else:
        print("(No scaler_min_/scaler_scale_ in npz — re-run train.py to save them.)")


if __name__ == "__main__":
    main()
