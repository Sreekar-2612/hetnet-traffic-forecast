"""Resolve canonical Milan dataset directory (`Wireless Dataset` preferred)."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np


def write_synthetic_milan_demo(out_dir, n_files=2, n_cells=100, n_intervals=400, seed=42):
    """Write Telecom-Italia-shaped .txt files for dry runs only."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ts = 1383260400000
    step = 600000
    for fi in range(n_files):
        fp = out_dir / f"synthetic-mi-demo-{fi + 1}.txt"
        buf = []
        for t in range(n_intervals):
            ts = base_ts + (t + fi * n_intervals) * step
            for g in range(1, n_cells + 1):
                internet = float(rng.uniform(0.5, 12.0))
                call_in = float(rng.uniform(0, 0.35))
                call_out = float(rng.uniform(0, 0.35))
                sms_in = float(rng.uniform(0, 0.35))
                sms_out = float(rng.uniform(0, 0.35))
                buf.append(
                    f"{g}\t{int(ts)}\t39\t{sms_in}\t{sms_out}\t{call_in}\t{call_out}\t{internet}\n"
                )
        fp.write_text("".join(buf), encoding="utf-8")


def _has_txt(p: Path) -> bool:
    return p.is_dir() and any(p.glob("*.txt"))


def resolve_wireless_dataset_dir(explicit: str | None = None) -> str:
    """
    Prefer `hetnet-traffic-forecast/Wireless Dataset`, then legacy `wireless dataset`.
    Synthetic demo data is only created when TASTF_USE_SYNTHETIC=1 (never silent Colab default).
    """
    if explicit:
        p = Path(explicit).expanduser()
        p = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
        if _has_txt(p):
            return str(p)

    # Canonical layout: .../hetnet-traffic-forecast/Wireless Dataset (sibling of reference_implementation/)
    _ref_dir = Path(__file__).resolve().parent
    _repo_root = _ref_dir.parent
    for cand in (
        _repo_root / "Wireless Dataset",
        _repo_root / "wireless dataset",
        _repo_root / "wireless_dataset",
    ):
        if _has_txt(cand):
            return str(cand)

    f = inspect.currentframe()
    try:
        while f:
            nb = f.f_globals.get("__vsc_ipynb_file__")
            if nb:
                nd = Path(nb).resolve().parent
                for cand in (
                    nd / "Wireless Dataset",
                    nd / "wireless dataset",
                    nd / "wireless_dataset",
                ):
                    if _has_txt(cand):
                        return str(cand)
                break
            f = f.f_back
    finally:
        del f

    cwd = Path.cwd().resolve()
    candidates = (
        cwd.parent / "Wireless Dataset",
        cwd.parent / "wireless dataset",
        cwd / "hetnet-traffic-forecast" / "Wireless Dataset",
        cwd / "hetnet-traffic-forecast" / "wireless dataset",
        cwd / "hetnet-traffic-forecast" / "wireless_dataset",
        cwd / "Wireless Dataset",
        cwd / "wireless dataset",
        cwd / "wireless_dataset",
        cwd.parent / "hetnet-traffic-forecast" / "Wireless Dataset",
        cwd.parent / "hetnet-traffic-forecast" / "wireless dataset",
        cwd.parent / "hetnet-traffic-forecast" / "wireless_dataset",
    )
    for cand in candidates:
        if _has_txt(cand):
            return str(cand)

    for base in [cwd, *cwd.parents[:12]]:
        for name in ("Wireless Dataset", "wireless dataset", "wireless_dataset"):
            for cand in (base / name, base / "hetnet-traffic-forecast" / name):
                if _has_txt(cand):
                    return str(cand)

    if os.environ.get("TASTF_USE_SYNTHETIC") == "1":
        target = Path("/content/wireless dataset") if Path("/content").exists() else (cwd / "wireless dataset")
        target.mkdir(parents=True, exist_ok=True)
        print(
            "TASTF_USE_SYNTHETIC=1: writing demo .txt files to:",
            target,
            "\n  Replace with real Milan data for meaningful metrics.",
        )
        write_synthetic_milan_demo(target)
        return str(target.resolve())

    raise FileNotFoundError(
        "No Milan .txt files found. Place Telecom Italia `sms-call-internet-mi-*.txt` files under "
        "`hetnet-traffic-forecast/Wireless Dataset`, pass an explicit path to resolve_wireless_dataset_dir, "
        "or set TASTF_USE_SYNTHETIC=1 for demo data only."
    )
