"""Forecasting metrics: normalized space and optional inverse-transform."""
from __future__ import annotations

import numpy as np


def smape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE in percent."""
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(100.0 * np.mean(2.0 * num / den))


def mae_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def masked_mape(y_pred: np.ndarray, y_true: np.ndarray, mask_min: float = 0.01) -> float:
    """MAPE (%) only where |y_true| >= mask_min."""
    m = np.abs(y_true) >= mask_min
    if not np.any(m):
        return float("nan")
    return float(
        100.0 * np.mean(np.abs((y_pred[m] - y_true[m]) / y_true[m]))
    )


def inverse_transform_arrays(
    y_norm: np.ndarray,
    scaler_min_: np.ndarray,
    scaler_scale_: np.ndarray,
) -> np.ndarray:
    """Inverse MinMax using saved `min_` and `scale_` vectors (from results.npz)."""
    return y_norm * scaler_scale_ + scaler_min_


def inverse_metrics_from_npz(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    scaler_min_: np.ndarray,
    scaler_scale_: np.ndarray,
) -> tuple[float, float, float]:
    """Same as inverse_metrics but using arrays saved in results.npz."""
    inv_p = inverse_transform_arrays(y_pred.reshape(-1, y_pred.shape[-1]), scaler_min_, scaler_scale_)
    inv_t = inverse_transform_arrays(y_true.reshape(-1, y_true.shape[-1]), scaler_min_, scaler_scale_)
    mae, rmse = mae_rmse(inv_p, inv_t)
    sm = smape(inv_p, inv_t)
    return mae, rmse, sm


def inverse_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    scaler,
    n_nodes: int,
) -> tuple[float, float, float]:
    """
    MAE / RMSE / sMAPE in original activity units after inverse MinMaxScaler.
    y_* shape (S, H, N) or flatten-compatible.
    """
    s, h, n = y_pred.shape
    inv_p = scaler.inverse_transform(y_pred.reshape(-1, n))
    inv_t = scaler.inverse_transform(y_true.reshape(-1, n))
    mae, rmse = mae_rmse(inv_p, inv_t)
    sm = smape(inv_p, inv_t)
    return mae, rmse, sm
