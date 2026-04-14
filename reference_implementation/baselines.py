"""Naive persistence and Ridge baselines on the same tensors as TASTF."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from metrics import mae_rmse, smape


def naive_persistence(y_true: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Predict each horizon step as the last input timestep.
    X: (S, seq, N), y_true shape (S, H, N) (unused for pred, for API symmetry).
    Returns (S, H, N).
    """
    last = X[:, -1, :]
    S, H, N = y_true.shape
    return np.broadcast_to(last[:, np.newaxis, :], (S, H, N)).copy()


def ridge_forecast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Independent Ridge per horizon: flattened lags -> all nodes.
    X_*: (S, seq, N), y_train: (S, horizon, N)
    """
    S, seq_len, N = X_train.shape
    _, horizon, _ = y_train.shape
    Xtr = X_train.reshape(S, seq_len * N)
    Xte = X_test.reshape(len(X_test), seq_len * N)
    pred = np.zeros((len(X_test), horizon, N), dtype=np.float32)
    for h in range(horizon):
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, y_train[:, h, :])
        pred[:, h, :] = reg.predict(Xte).astype(np.float32)
    return pred


def eval_np(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mae, rmse = mae_rmse(pred, y)
    return {"mae": mae, "rmse": rmse, "smape": smape(pred, y)}


def print_baseline_report(name: str, metrics: dict[str, float]) -> None:
    print(f"  {name}: MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} sMAPE={metrics['smape']:.2f}%")
