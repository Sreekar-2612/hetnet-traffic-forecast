"""Heterogeneous and homogeneous graph construction with optional GeoJSON coordinates."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData


def _polygon_centroid(ring: list) -> tuple[float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return float(np.mean(xs)), float(np.mean(ys))


def load_cell_centroids(geojson_path: str | Path) -> dict[int, tuple[float, float]]:
    """
    Load square_id -> (lon, lat) from milano-grid-style GeoJSON.
    """
    path = Path(geojson_path)
    if not path.is_file():
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    with path.open(encoding="utf-8") as f:
        gj = json.load(f)

    out: dict[int, tuple[float, float]] = {}
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        sid = (
            props.get("id")
            or props.get("square_id")
            or props.get("cellId")
            or props.get("SQUARE_ID")
        )
        if sid is None:
            continue
        try:
            sid = int(sid)
        except (TypeError, ValueError):
            continue
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue
        if gtype == "Polygon":
            lon, lat = _polygon_centroid(coords[0])
        elif gtype == "MultiPolygon":
            lon, lat = _polygon_centroid(coords[0][0])
        else:
            continue
        out[sid] = (lon, lat)
    return out


def _sqrt_layout(N: int) -> np.ndarray:
    sqrt_n = int(np.ceil(np.sqrt(N)))
    return np.array([[i // sqrt_n, i % sqrt_n] for i in range(N)], dtype=np.float32)


def coords_for_grids(
    grid_ids: np.ndarray | None,
    N: int,
    geojson_path: str | None = None,
) -> tuple[np.ndarray, str]:
    """
    Returns (N, 2) float32 coordinates and a description of the source.
    grid_ids: length N column order matching the traffic matrix.
    """
    if geojson_path and Path(geojson_path).is_file() and grid_ids is not None:
        cmap = load_cell_centroids(geojson_path)
        coords = np.zeros((N, 2), dtype=np.float32)
        fallback = 0
        layout = _sqrt_layout(N)
        for i, gid in enumerate(grid_ids):
            if int(gid) in cmap:
                lon, lat = cmap[int(gid)]
                coords[i] = (lon, lat)
            else:
                coords[i] = layout[i]
                fallback += 1
        if fallback:
            warnings.warn(
                f"GeoJSON: {fallback}/{N} grid ids missing; filled with sqrt layout positions for KNN.",
                UserWarning,
                stacklevel=2,
            )
        return coords, f"geojson:{geojson_path}"

    warnings.warn(
        "No valid GeoJSON: using sqrt(N) grid layout for KNN edges (not geographic). "
        "Add milano-grid.geojson from the Milan open-data bundle for real coordinates.",
        UserWarning,
        stacklevel=2,
    )
    return _sqrt_layout(N), "sqrt_heuristic"


def knn_edges_from_coords(
    src_ids: np.ndarray,
    dst_ids: np.ndarray,
    coords: np.ndarray,
    k: int,
    local_map_src: dict,
    local_map_dst: dict,
) -> torch.Tensor:
    src_e, dst_e = [], []
    for s_g in src_ids:
        dists = np.linalg.norm(coords[dst_ids] - coords[s_g], axis=1)
        n_to_find = k + 1 if np.array_equal(src_ids, dst_ids) else k
        nns = np.argsort(dists)[:n_to_find]
        for d_l in nns:
            d_g = dst_ids[d_l]
            if s_g == d_g:
                continue
            src_e.append(local_map_src[s_g])
            dst_e.append(local_map_dst[d_g])
            if len(src_e) >= k * len(src_ids):
                break
    return torch.tensor([src_e, dst_e], dtype=torch.long)


def build_hetero_graph(
    data_matrix: np.ndarray,
    k: int = 5,
    macro_pct: float = 0.10,
    femto_pct: float = 0.30,
    grid_ids: np.ndarray | None = None,
    geojson_path: str | None = None,
    coords: np.ndarray | None = None,
) -> tuple[HeteroData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Heterogeneous tier graph. coords from GeoJSON when provided.
    Returns hetero graph and macro_idx, pico_idx, femto_idx (global grid indices).
    """
    T, N = data_matrix.shape
    mean_act = data_matrix.mean(axis=0)
    sorted_idx = np.argsort(mean_act)

    n_femto = int(N * femto_pct)
    n_macro = int(N * macro_pct)

    femto_idx = sorted_idx[:n_femto]
    macro_idx = sorted_idx[-n_macro:]
    pico_idx = sorted_idx[n_femto:-n_macro]

    macro_local = {int(g): l for l, g in enumerate(macro_idx)}
    pico_local = {int(g): l for l, g in enumerate(pico_idx)}
    femto_local = {int(g): l for l, g in enumerate(femto_idx)}

    if coords is None:
        coords, _src = coords_for_grids(grid_ids, N, geojson_path)

    hetero = HeteroData()
    hetero["macro"].x = torch.tensor(mean_act[macro_idx]).unsqueeze(1).float()
    hetero["pico"].x = torch.tensor(mean_act[pico_idx]).unsqueeze(1).float()
    hetero["femto"].x = torch.tensor(mean_act[femto_idx]).unsqueeze(1).float()

    def knn_edges(src_ids, dst_ids, k_):
        if any(g in macro_local for g in dst_ids):
            local_map_dst = macro_local
        elif any(g in pico_local for g in dst_ids):
            local_map_dst = pico_local
        else:
            local_map_dst = femto_local

        if any(g in macro_local for g in src_ids):
            local_map_src = macro_local
        elif any(g in pico_local for g in src_ids):
            local_map_src = pico_local
        else:
            local_map_src = femto_local

        return knn_edges_from_coords(src_ids, dst_ids, coords, k_, local_map_src, local_map_dst)

    hetero["macro", "geo", "macro"].edge_index = knn_edges(macro_idx, macro_idx, k)
    hetero["pico", "geo", "pico"].edge_index = knn_edges(pico_idx, pico_idx, k)
    hetero["femto", "geo", "femto"].edge_index = knn_edges(femto_idx, femto_idx, k)
    hetero["macro", "cross", "pico"].edge_index = knn_edges(macro_idx, pico_idx, k)
    hetero["macro", "cross", "femto"].edge_index = knn_edges(macro_idx, femto_idx, k)

    return hetero, macro_idx, pico_idx, femto_idx, coords


def build_homogeneous_graph(
    data_matrix: np.ndarray,
    k: int = 5,
    grid_ids: np.ndarray | None = None,
    geojson_path: str | None = None,
    coords: np.ndarray | None = None,
) -> tuple[Data, np.ndarray]:
    """Single graph type over all N nodes; KNN edges by geographic or sqrt layout."""
    T, N = data_matrix.shape
    mean_act = data_matrix.mean(axis=0)
    if coords is None:
        coords, _ = coords_for_grids(grid_ids, N, geojson_path)
    x = torch.tensor(mean_act, dtype=torch.float32).unsqueeze(1)

    src, dst = [], []
    for i in range(N):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        nn = np.argsort(dists)[1 : k + 1]
        for j in nn:
            src.append(i)
            dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    return data, coords


if __name__ == "__main__":
    data = np.random.rand(100, 100)
    h, m, p, f, c = build_hetero_graph(data, grid_ids=np.arange(1, 101))
    print("hetero ok", h.node_types, c.shape)
    d, _ = build_homogeneous_graph(data, grid_ids=np.arange(1, 101))
    print("homo ok", d.edge_index.shape)
