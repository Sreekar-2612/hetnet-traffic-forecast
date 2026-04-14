"""Telecom Italia Milan loader: chronological split, train-only scaler fit, temporal features."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def intervals_ms_to_sin_cos(intervals_ms: np.ndarray) -> np.ndarray:
    """Shape (...,) ms epochs -> (..., 4) sin/cos hour and day-of-week."""
    ts = pd.to_datetime(np.asarray(intervals_ms).ravel(), unit="ms")
    hour = ts.hour.values + ts.minute.values / 60.0
    dow = ts.dayofweek.values.astype(np.float64)
    hr = 2.0 * np.pi * hour / 24.0
    dw = 2.0 * np.pi * dow / 7.0
    out = np.stack([np.sin(hr), np.cos(hr), np.sin(dw), np.cos(dw)], axis=-1).astype(np.float32)
    return out.reshape(*np.asarray(intervals_ms).shape, 4)


@dataclass
class TelecomLoadResult:
    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    scaler: MinMaxScaler
    split_info: dict[str, Any]
    grid_ids: np.ndarray
    time_origin_train: np.ndarray
    time_origin_val: np.ndarray
    time_origin_test: np.ndarray


def load_telecom_italia(
    data_dir: str,
    n_cells: int = 100,
    seq_len: int = 12,
    horizon: int = 3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    random_seed: int | None = 42,
) -> TelecomLoadResult:
    """
    Load all `*.txt` in `data_dir`, build sequences, chronological 70/10/20 split.
    MinMaxScaler is fit only on the first `train_ratio` fraction of *timesteps* (no leakage).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if not files:
        raise FileNotFoundError(f"No .txt data files found in {data_dir}")

    all_data: list[pd.DataFrame] = []
    for file in files:
        filepath = os.path.join(data_dir, file)
        print(f"Loading {filepath}...")
        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=[
                "grid",
                "interval",
                "country",
                "sms_in",
                "sms_out",
                "call_in",
                "call_out",
                "internet",
            ],
        )
        df = df[df["grid"] <= n_cells].fillna(0)
        df["activity"] = df["internet"] + df["call_in"] + df["call_out"]
        pivot = df.pivot_table(index="interval", columns="grid", values="activity", fill_value=0)
        all_data.append(pivot)

    full_pivot = pd.concat(all_data).sort_index()
    raw = full_pivot.values.astype(np.float32)
    intervals = full_pivot.index.values
    grid_ids = full_pivot.columns.values.astype(np.int64)
    n_time, n_nodes = raw.shape

    train_time_end = int(train_ratio * n_time)
    scaler = MinMaxScaler()
    scaler.fit(raw[:train_time_end])
    data = scaler.transform(raw)

    X_list, y_list, origin_ms = [], [], []
    for t in range(n_time - seq_len - horizon):
        X_list.append(data[t : t + seq_len])
        y_list.append(data[t + seq_len : t + seq_len + horizon])
        origin_ms.append(intervals[t + seq_len - 1])

    X = np.stack(X_list)
    y = np.stack(y_list)
    origin_ms = np.array(origin_ms)
    time_full = intervals_ms_to_sin_cos(origin_ms)

    n_samples = len(X)
    s1 = int(train_ratio * n_samples)
    s2 = int((train_ratio + val_ratio) * n_samples)

    split_info = {
        "n_timesteps": n_time,
        "n_nodes": n_nodes,
        "n_sequences": n_samples,
        "train_samples": (0, s1),
        "val_samples": (s1, s2),
        "test_samples": (s2, n_samples),
        "seq_len": seq_len,
        "horizon": horizon,
        "first_interval_ms": int(intervals[0]),
        "last_interval_ms": int(intervals[-1]),
        "scaler_fit_timesteps": (0, train_time_end),
        "files_loaded": files,
        "random_seed": random_seed,
    }

    return TelecomLoadResult(
        train=(X[:s1], y[:s1]),
        val=(X[s1:s2], y[s1:s2]),
        test=(X[s2:], y[s2:]),
        scaler=scaler,
        split_info=split_info,
        grid_ids=grid_ids,
        time_origin_train=time_full[:s1],
        time_origin_val=time_full[s1:s2],
        time_origin_test=time_full[s2:],
    )


if __name__ == "__main__":
    from paths import resolve_wireless_dataset_dir

    path = resolve_wireless_dataset_dir(None)
    r = load_telecom_italia(path, n_cells=100)
    print(r.split_info)
    print("train", r.train[0].shape, r.time_origin_train.shape)
