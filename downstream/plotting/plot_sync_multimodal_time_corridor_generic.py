#!/usr/bin/env python3
"""Render synchronized multimodal temporal trajectories as an all-time ribbon corridor.

Standalone generic entrypoint:
- Inputs are explicit paths (dynamic adata, two processed adata files, map model).
- Direction is configurable (1->2 or 2->1).
- Works with variable numbers of real time points, then inserts light interpolated stations.
- Writes static figures plus an HTML preview and a manifest.
"""
# Example: python downstream/plotting/plot_sync_multimodal_time_corridor_generic.py \
#   --dynamic-adata paper_data/dynamics/hspc_31800/adata.h5ad \
#   --space1-processed paper_data/processed/hspc_31800/rna.h5ad \
#   --space2-processed paper_data/processed/hspc_31800/protein.h5ad \
#   --map-model paper_data/maps/hspc_31800/best_model.pt \
#   --direction 2to1 --output-dir results/corridor_hspc_2to1 --n-traj 4
from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter

mpl.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("umap-learn is required. Please install umap-learn.") from exc

# UMAP estimators archived with older numba releases reference the former
# ``old_scalars`` module path.  Alias it to the current module so the original
# paper pickle can be loaded without refitting or changing its embedding.
try:  # pragma: no cover - only needed for legacy release assets
    import numba.core.types.scalars as _numba_scalars
    from pynndescent.pynndescent_ import NNDescent as _NNDescent

    sys.modules.setdefault("numba.core.types.old_scalars", _numba_scalars)
    if not getattr(_NNDescent.__setstate__, "_mofi_legacy_compat", False):
        _nn_setstate = _NNDescent.__setstate__

        def _legacy_nn_setstate(self, state):
            state = dict(state)
            state.setdefault("quantization", None)
            state.setdefault("parallel_batch_queries", False)
            if "_min_distance" not in state:
                graph = state.get("_search_graph")
                graph_data = getattr(graph, "data", None)
                state["_min_distance"] = (
                    float(np.min(graph_data)) if graph_data is not None and len(graph_data) else 0.0
                )
            return _nn_setstate(self, state)

        _legacy_nn_setstate._mofi_legacy_compat = True
        _NNDescent.__setstate__ = _legacy_nn_setstate
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.Map.tl.transport_factory import build_transport_map
from CytoBridge.tl.analysis_dense_time import _build_ode_trajectory, _map_to_secondary
from CytoBridge.utils import load_model_from_adata

DEFAULT_OUT_SUBDIR = "sync_multimodal_time_tower"

TRAJECTORY_PALETTE = [
    "#0B5D7A",
    "#D97A2B",
    "#4B8F4B",
    "#C55252",
    "#7C62A3",
    "#A55C84",
    "#4C7F6F",
    "#9A6B3F",
    "#4E6FAE",
    "#B59A2E",
]

TIME_GRADIENT_HEX = {
    0: "#6A4C93", 1: "#7B5CB5", 2: "#8C6CD7", 3: "#9D7CF9", 4: "#B565D1",
    5: "#CD4EA9", 6: "#E33D8F", 7: "#F56B5C", 8: "#FDBF2D", 9: "#FFF04A"
}
TIME_COLOR_MIN = 0.0
TIME_COLOR_MAX = 9.0
RNA_CARD_FACE = "#F5F8FB"
RNA_CARD_EDGE = "#96AFC2"
RNA_BG_POINT = "#B7C7D4"
ATAC_CARD_FACE = "#FBF7F2"
ATAC_CARD_EDGE = "#BCA894"
ATAC_BG_POINT = "#D5C4B6"
STATION_FOG_GRAY = "#BFC5CC"
SYNC_BRIDGE_COLOR = "#D7C8AE"
SHADOW_COLOR = (0.12, 0.16, 0.20, 0.08)
TITLE_COLOR = "#22313F"
SUBTITLE_COLOR = "#5A6B78"
TEXT_COLOR = "#2F3D4A"
TRAJECTORY_CONTOUR_COLOR = "#394852"
IDENTITY_TABLE_FACE = "#F6F1E9"
IDENTITY_TABLE_EDGE = "#C9BAA8"


@dataclass(frozen=True)
class CardSpec:
    x: float
    y: float
    w: float
    h: float
    shear: float
    label: str
    kind: str


@dataclass(frozen=True)
class RibbonSpec:
    x0: float
    y0: float
    width: float
    height: float
    slope: float
    label: str
    kind: str


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at: {path}")
    return data


def _build_time_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if t_end <= t_start:
        raise ValueError("t_end must be > t_start")
    n_steps = int(round((t_end - t_start) / dt))
    grid = np.linspace(float(t_start), float(t_end), n_steps + 1, dtype=float)
    return np.asarray(grid, dtype=float)


def _select_integer_days(time_grid: np.ndarray, t_start: float, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
    day_start = int(np.ceil(t_start))
    day_end = int(np.floor(t_end))
    if day_end < day_start:
        raise ValueError("No integer day in selected time range.")
    days = np.arange(day_start, day_end + 1, dtype=int)
    idx = []
    for d in days:
        d_f = float(d)
        j = int(np.argmin(np.abs(time_grid - d_f)))
        if abs(float(time_grid[j]) - d_f) > 1e-6:
            raise ValueError(f"Cannot match integer day {d} in time_grid.")
        idx.append(j)
    return np.asarray(days, dtype=int), np.asarray(idx, dtype=int)


def _build_display_times(anchor_days: np.ndarray, midpoints_per_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    if int(midpoints_per_gap) < 0:
        raise ValueError("midpoints_per_gap must be >= 0")
    days = np.asarray(anchor_days, dtype=float)
    if days.ndim != 1 or len(days) == 0:
        raise ValueError("anchor_days must be a non-empty 1D array")

    times: List[float] = []
    is_anchor: List[bool] = []
    m = int(midpoints_per_gap)
    for i in range(len(days)):
        t0 = float(days[i])
        times.append(t0)
        is_anchor.append(True)
        if i < len(days) - 1 and m > 0:
            t1 = float(days[i + 1])
            for k in range(m):
                frac = float(k + 1) / float(m + 1)
                times.append((1.0 - frac) * t0 + frac * t1)
                is_anchor.append(False)
    return np.asarray(times, dtype=float), np.asarray(is_anchor, dtype=bool)


def _map_times_to_grid_indices(time_grid: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    grid = np.asarray(time_grid, dtype=float)
    q = np.asarray(query_times, dtype=float)
    if grid.ndim != 1 or len(grid) == 0:
        raise ValueError("time_grid must be a non-empty 1D array")
    if q.ndim != 1 or len(q) == 0:
        raise ValueError("query_times must be a non-empty 1D array")

    idx = np.searchsorted(grid, q, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    left = np.clip(idx - 1, 0, len(grid) - 1)
    right = idx
    use_left = np.abs(q - grid[left]) <= np.abs(q - grid[right])
    out = np.where(use_left, left, right)
    return np.asarray(out, dtype=int)


def _select_obs_time_mask(obs_times: np.ndarray, target_time: float, atol: float = 1e-6) -> Tuple[np.ndarray, float]:
    arr = np.asarray(obs_times, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"obs_times must be 1D, got shape={arr.shape}")
    if arr.size == 0:
        return np.zeros(0, dtype=bool), float(target_time)
    mask = np.isclose(arr, float(target_time), atol=float(atol))
    if np.any(mask):
        return np.asarray(mask, dtype=bool), float(target_time)
    uniq = np.unique(arr)
    nearest = float(uniq[int(np.argmin(np.abs(uniq - float(target_time))))])
    mask_near = np.isclose(arr, nearest, atol=float(atol))
    if np.any(mask_near):
        return np.asarray(mask_near, dtype=bool), nearest
    return np.asarray(arr == nearest, dtype=bool), nearest


def _station_target_count(n_real: int) -> int:
    if n_real <= 1:
        return int(n_real)
    target = 9 if (int(n_real) % 2 == 1) else 10
    return int(max(int(n_real), target))


def _build_station_template(real_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(sorted({float(x) for x in np.asarray(real_times, dtype=float).tolist()}), dtype=float)
    if len(times) == 0:
        raise ValueError("real_times is empty")
    if len(times) == 1:
        return times.copy(), np.asarray([True], dtype=bool), np.zeros(0, dtype=int)
    gaps = int(len(times) - 1)
    target = _station_target_count(len(times))
    extra_total = int(max(target - len(times), 0))
    base = int(extra_total // gaps)
    rem = int(extra_total % gaps)
    extras = np.asarray([base + (1 if i < rem else 0) for i in range(gaps)], dtype=int)

    station_times: List[float] = []
    station_is_real: List[bool] = []
    for i in range(len(times)):
        t0 = float(times[i])
        station_times.append(t0)
        station_is_real.append(True)
        if i >= len(times) - 1:
            continue
        t1 = float(times[i + 1])
        m = int(extras[i])
        for k in range(m):
            frac = float(k + 1) / float(m + 1)
            station_times.append((1.0 - frac) * t0 + frac * t1)
            station_is_real.append(False)
    return np.asarray(station_times, dtype=float), np.asarray(station_is_real, dtype=bool), extras


def _load_source_indices(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    vals: List[int] = []
    for token in text.replace(",", " ").split():
        vals.append(int(token))
    if len(vals) == 0:
        raise ValueError(f"No indices found in file: {path}")
    return np.asarray(vals, dtype=int)


def _select_source_indices(
    latent: np.ndarray,
    obs_times: np.ndarray,
    *,
    n_traj: int,
    seed: int,
    source_index_file: str,
) -> np.ndarray:
    n = int(n_traj)
    if n <= 0:
        raise ValueError("n_traj must be > 0")
    idx_file = str(source_index_file).strip()
    if idx_file:
        idx = _load_source_indices(Path(idx_file).resolve())
        if len(idx) < n:
            raise ValueError(f"source_index_file has {len(idx)} indices, but n_traj={n}")
        return np.asarray(idx[:n], dtype=int)
    if n == 4:
        raise ValueError("n_traj=4 requires --source-index-file per current workflow rule.")

    times = np.asarray(obs_times, dtype=float)
    if times.ndim != 1 or len(times) != int(np.asarray(latent).shape[0]):
        raise ValueError("obs_times shape is incompatible with latent rows")
    t0 = float(np.min(times))
    pool = np.flatnonzero(np.isclose(times, t0, atol=1e-6))
    if len(pool) < n:
        raise ValueError(f"Not enough cells at earliest time ({t0:g}): need {n}, got {len(pool)}")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(pool.astype(int), size=n, replace=False).astype(int))


def _parse_bridge_stations(text: str, n_stations: int) -> List[int]:
    raw = [x.strip() for x in str(text).split(",") if x.strip()]
    if not raw:
        raw = ["0", "-1"]
    out: List[int] = []
    seen = set()
    for x in raw:
        idx = int(x)
        if idx < 0:
            idx = int(n_stations + idx)
        idx = int(np.clip(idx, 0, max(n_stations - 1, 0)))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _resolve_device(requested: str, manifest_device: str) -> str:
    req = str(requested).strip().lower()
    mf = str(manifest_device).strip().lower()
    if req in {"", "auto"}:
        if mf.startswith("cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return req


def _rank_trajs_day9(key_path_df: pd.DataFrame, t_end: float) -> pd.DataFrame:
    required = {"time", "trajectory_index", "path_score"}
    if not required.issubset(set(key_path_df.columns)):
        raise ValueError("key_path_cells.csv must contain columns: time, trajectory_index, path_score")
    day_df = key_path_df[np.isclose(key_path_df["time"].to_numpy(dtype=float), float(t_end), atol=1e-6)].copy()
    if day_df.empty:
        raise ValueError(f"No path-score rows found at time={t_end:g}")
    rank_df = (
        day_df[["trajectory_index", "path_score"]]
        .drop_duplicates(subset=["trajectory_index"])
        .sort_values(["path_score", "trajectory_index"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return rank_df


def _select_top_traj_records(
    rank_df: pd.DataFrame,
    pred_main: ad.AnnData,
    n_traj: int,
) -> List[Dict[str, Any]]:
    if n_traj <= 0:
        raise ValueError("n_traj must be > 0")
    top = rank_df.head(int(n_traj)).copy()
    if top.empty:
        raise ValueError("No trajectory selected.")

    obs = pred_main.obs.copy()
    required = {"pred_time", "trajectory_index", "source_cell_index"}
    if not required.issubset(set(obs.columns)):
        raise ValueError("pred_main.h5ad obs must contain pred_time/trajectory_index/source_cell_index")
    t0 = float(np.min(np.asarray(obs["pred_time"], dtype=float)))
    obs0 = obs[np.isclose(np.asarray(obs["pred_time"], dtype=float), t0, atol=1e-6)].copy()
    if obs0.empty:
        raise ValueError("No t=0 rows found in pred_main.obs")

    by_traj = (
        obs0[["trajectory_index", "source_cell_index"]]
        .drop_duplicates(subset=["trajectory_index"])
        .set_index("trajectory_index")
    )
    records: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        traj_idx = int(row["trajectory_index"])
        if traj_idx not in by_traj.index:
            raise ValueError(f"trajectory_index={traj_idx} missing at t=0 in pred_main.obs")
        src_idx = int(by_traj.loc[traj_idx, "source_cell_index"])
        records.append(
            {
                "trajectory_index": traj_idx,
                "source_cell_index": src_idx,
                "path_score_day9": float(row["path_score"]),
            }
        )
    return records


def _fit_umap(
    latent: np.ndarray,
    seed: int = 42,
    n_neighbors: int = 40,
    min_dist: float = 0.25,
) -> Tuple[Any, np.ndarray]:
    if int(n_neighbors) < 2:
        raise ValueError("UMAP n_neighbors must be >= 2")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(seed),
    )
    emb = reducer.fit_transform(np.asarray(latent, dtype=np.float32))
    return reducer, np.asarray(emb, dtype=float)


def _load_umap_transform(model_path: str, latent: np.ndarray) -> Tuple[Any, np.ndarray]:
    path = Path(str(model_path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"UMAP model not found: {path}")
    with path.open("rb") as f:
        reducer = pickle.load(f)
    emb = reducer.transform(np.asarray(latent, dtype=np.float32))
    return reducer, np.asarray(emb, dtype=float)


def _score_time_centroid_geometry(embedding: np.ndarray, obs_times: np.ndarray) -> Dict[str, Any]:
    emb = np.asarray(embedding, dtype=float)
    times = np.asarray(obs_times, dtype=float)
    uniq = np.unique(times)
    if len(uniq) < 2:
        return {
            "unique_times": [float(x) for x in uniq.tolist()],
            "centroid_dist_to_mean": [0.0 for _ in uniq.tolist()],
            "step_dists": [],
            "max_step_over_median": 0.0,
            "max_centroid_dist": 0.0,
        }

    centroids = []
    for t in uniq.tolist():
        pts = emb[np.isclose(times, float(t), atol=1e-6)]
        if len(pts) == 0:
            continue
        centroids.append(np.mean(pts, axis=0))
    cents = np.asarray(centroids, dtype=float)
    global_cent = np.mean(cents, axis=0)
    centroid_dists = np.linalg.norm(cents - global_cent[None, :], axis=1)
    step_dists = np.linalg.norm(np.diff(cents, axis=0), axis=1)
    median_step = float(np.median(step_dists)) if len(step_dists) > 0 else 0.0
    if median_step <= 1e-12:
        ratio = float(np.max(step_dists)) if len(step_dists) > 0 else 0.0
    else:
        ratio = float(np.max(step_dists) / median_step)
    return {
        "unique_times": [float(x) for x in uniq.tolist()],
        "centroid_dist_to_mean": [float(x) for x in centroid_dists.tolist()],
        "step_dists": [float(x) for x in step_dists.tolist()],
        "max_step_over_median": float(ratio),
        "max_centroid_dist": float(np.max(centroid_dists)) if len(centroid_dists) > 0 else 0.0,
    }


def _fit_umap_auto_by_time_geometry(
    latent: np.ndarray,
    obs_times: np.ndarray,
    seed: int = 42,
    n_neighbors_candidates: Sequence[int] = (100, 150, 200),
    min_dist_candidates: Sequence[float] = (0.70, 0.90),
    max_points_per_time: int = 150,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    X = np.asarray(latent, dtype=np.float32)
    times = np.asarray(obs_times, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected latent matrix [n_obs, dim], got shape={X.shape}")
    if len(times) != X.shape[0]:
        raise ValueError("obs_times length must match latent rows")

    rng = np.random.default_rng(int(seed))
    sampled_idx: List[int] = []
    for t in np.unique(times).tolist():
        idx = np.flatnonzero(np.isclose(times, float(t), atol=1e-6))
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        keep_n = min(len(idx), int(max_points_per_time))
        sampled_idx.extend(idx[:keep_n].tolist())
    if len(sampled_idx) == 0:
        raise RuntimeError("RNA UMAP auto-selection could not sample any observations")
    sampled_idx = sorted(sampled_idx)
    X_search = np.asarray(X[sampled_idx], dtype=np.float32)
    times_search = np.asarray(times[sampled_idx], dtype=float)

    best_payload: Optional[Dict[str, Any]] = None
    candidate_scores: List[Dict[str, Any]] = []

    for nn in n_neighbors_candidates:
        for md in min_dist_candidates:
            reducer, emb = _fit_umap(
                X_search,
                seed=int(seed),
                n_neighbors=int(nn),
                min_dist=float(md),
            )
            score = _score_time_centroid_geometry(emb, times_search)
            payload = {
                "n_neighbors": int(nn),
                "min_dist": float(md),
                "score": float(score["max_step_over_median"]),
                "max_centroid_dist": float(score["max_centroid_dist"]),
                "step_dists": [float(x) for x in score["step_dists"]],
                "centroid_dist_to_mean": [float(x) for x in score["centroid_dist_to_mean"]],
            }
            candidate_scores.append(payload)
            if best_payload is None:
                best_payload = payload
                continue
            is_better = (
                float(payload["score"]) < float(best_payload["score"]) - 1e-12
                or (
                    abs(float(payload["score"]) - float(best_payload["score"])) <= 1e-12
                    and float(payload["max_centroid_dist"]) < float(best_payload["max_centroid_dist"]) - 1e-12
                )
            )
            if is_better:
                best_payload = payload

    if best_payload is None:
        raise RuntimeError("RNA UMAP auto-selection failed to produce any candidate")

    best_reducer, best_emb = _fit_umap(
        X,
        seed=int(seed),
        n_neighbors=int(best_payload["n_neighbors"]),
        min_dist=float(best_payload["min_dist"]),
    )

    summary = {
        "selection_mode": "auto_grid_search",
        "selected_n_neighbors": int(best_payload["n_neighbors"]),
        "selected_min_dist": float(best_payload["min_dist"]),
        "best_max_step_over_median": float(best_payload["score"]),
        "best_max_centroid_dist": float(best_payload["max_centroid_dist"]),
        "search_sample_size": int(len(sampled_idx)),
        "search_max_points_per_time": int(max_points_per_time),
        "candidate_scores": candidate_scores,
    }
    return best_reducer, np.asarray(best_emb, dtype=float), summary


def _transform_series_2d(series_3d: np.ndarray, reducer: Any) -> np.ndarray:
    arr = np.asarray(series_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D trajectory array, got shape={arr.shape}")
    t_num, n_cells, dim = arr.shape
    flat = arr.reshape(t_num * n_cells, dim)
    out = reducer.transform(flat)
    return np.asarray(out, dtype=float).reshape(t_num, n_cells, 2)


def _rgba(hex_color: str, alpha: float) -> Tuple[float, float, float, float]:
    rgba = list(to_rgba(hex_color))
    rgba[3] = float(np.clip(alpha, 0.0, 1.0))
    return tuple(rgba)


def _soften_color(color: str, white_mix: float = 0.20, saturation_keep: float = 0.85) -> Tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    rgb = np.asarray(sns.desaturate(tuple(rgb.tolist()), prop=float(saturation_keep)), dtype=float)
    rgb = (1.0 - float(white_mix)) * rgb + float(white_mix) * np.ones(3, dtype=float)
    return tuple(np.clip(rgb, 0.0, 1.0).tolist())


def _set_time_color_range(t_min: float, t_max: float) -> None:
    global TIME_COLOR_MIN, TIME_COLOR_MAX
    lo = float(t_min)
    hi = float(t_max)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 9.0
    if hi <= lo:
        hi = lo + 1.0
    TIME_COLOR_MIN = float(lo)
    TIME_COLOR_MAX = float(hi)


def _to_time_gradient_axis(time_value: float) -> float:
    span = max(float(TIME_COLOR_MAX - TIME_COLOR_MIN), 1e-9)
    return 9.0 * (float(time_value) - float(TIME_COLOR_MIN)) / span


def _time_gradient_rgb(time_value: float) -> Tuple[float, float, float]:
    anchors = np.asarray(sorted(TIME_GRADIENT_HEX.keys()), dtype=float)
    t_norm = _to_time_gradient_axis(float(time_value))
    t = float(np.clip(float(t_norm), float(anchors[0]), float(anchors[-1])))
    if t <= float(anchors[0]):
        return tuple(np.asarray(to_rgb(TIME_GRADIENT_HEX[int(anchors[0])]), dtype=float).tolist())
    if t >= float(anchors[-1]):
        return tuple(np.asarray(to_rgb(TIME_GRADIENT_HEX[int(anchors[-1])]), dtype=float).tolist())
    right = int(np.searchsorted(anchors, t, side="right"))
    left = max(right - 1, 0)
    t0 = float(anchors[left])
    t1 = float(anchors[right])
    if abs(t1 - t0) < 1e-9:
        return tuple(np.asarray(to_rgb(TIME_GRADIENT_HEX[int(t0)]), dtype=float).tolist())
    frac = (t - t0) / (t1 - t0)
    rgb0 = np.asarray(to_rgb(TIME_GRADIENT_HEX[int(t0)]), dtype=float)
    rgb1 = np.asarray(to_rgb(TIME_GRADIENT_HEX[int(t1)]), dtype=float)
    rgb = (1.0 - frac) * rgb0 + frac * rgb1
    return tuple(np.clip(rgb, 0.0, 1.0).tolist())


def _time_gradient_rgba(time_value: float, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    rgb = np.asarray(_time_gradient_rgb(time_value), dtype=float)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(np.clip(alpha, 0.0, 1.0)))


def _make_time_palette(unique_times: Sequence[float]) -> Dict[float, Tuple[float, float, float]]:
    uniq = np.asarray(sorted({float(t) for t in unique_times}), dtype=float)
    if len(uniq) == 0:
        return {}
    return {
        float(t): tuple(
            _soften_color(
                _rgb_to_hex(_time_gradient_rgb(float(t))),
                white_mix=0.48,
                saturation_keep=0.58,
            )
        )
        for t in uniq.tolist()
    }


def _make_global_time_palette(unique_times: Sequence[float]) -> Dict[float, Tuple[float, float, float]]:
    uniq = np.asarray(sorted({float(t) for t in unique_times}), dtype=float)
    if len(uniq) == 0:
        return {}
    return {
        float(t): tuple(_time_gradient_rgb(float(t)))
        for t in uniq.tolist()
    }


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    vals = [int(np.clip(np.round(float(v) * 255.0), 0, 255)) for v in rgb[:3]]
    return f"#{vals[0]:02X}{vals[1]:02X}{vals[2]:02X}"


def _line_segments(path_xy: np.ndarray) -> np.ndarray:
    path = np.asarray(path_xy, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return np.zeros((0, 2, 2), dtype=float)
    return np.stack([path[:-1], path[1:]], axis=1).astype(float)


def _draw_time_gradient_line(
    ax: plt.Axes,
    *,
    path_xy: np.ndarray,
    times: np.ndarray,
    zorder: float,
    outer_linewidth: float = 4.8,
    inner_linewidth: float = 2.2,
    outer_alpha: float = 0.18,
) -> None:
    path = np.asarray(path_xy, dtype=float)
    tvals = np.asarray(times, dtype=float)
    if len(path) < 2 or len(path) != len(tvals):
        return
    segments = _line_segments(path)
    if len(segments) == 0:
        return
    seg_times = 0.5 * (tvals[:-1] + tvals[1:])
    outer = LineCollection(
        segments,
        colors=[_rgba(TRAJECTORY_CONTOUR_COLOR, outer_alpha)],
        linewidths=float(outer_linewidth),
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    inner = LineCollection(
        segments,
        colors=[_time_gradient_rgba(float(t), alpha=0.98) for t in seg_times.tolist()],
        linewidths=float(inner_linewidth),
        capstyle="round",
        joinstyle="round",
        zorder=zorder + 0.06,
    )
    # ax.add_collection(outer)
    ax.add_collection(inner)


def _draw_time_markers(
    ax: plt.Axes,
    *,
    anchor_points: np.ndarray,
    anchor_times: np.ndarray,
    zorder: float,
) -> None:
    pts = np.asarray(anchor_points, dtype=float)
    tvals = np.asarray(anchor_times, dtype=float)
    if len(pts) == 0 or len(pts) != len(tvals):
        return
    outline = _rgba("#121418", 0.52)
    if len(pts) > 3:
        mid = pts[2:-1]
        mid_t = tvals[2:-1]
        ax.scatter(
            mid[:, 0],
            mid[:, 1],
            s=18,
            c=[_time_gradient_rgba(float(t), alpha=0.92) for t in mid_t.tolist()],
            edgecolors=[outline],
            linewidths=0.72,
            zorder=zorder,
        )
    if len(pts) > 2:
        d1 = pts[1]
        d1_color = _time_gradient_rgba(float(tvals[1]), alpha=0.96)
        ax.scatter(
            [d1[0]],
            [d1[1]],
            s=30,
            c=[d1_color],
            edgecolors=[outline],
            linewidths=0.82,
            zorder=zorder + 0.02,
        )
    start = pts[0]
    end = pts[-1]
    start_color = _time_gradient_rgba(float(tvals[0]), alpha=1.0)
    end_color = _time_gradient_rgba(float(tvals[-1]), alpha=1.0)
    ax.scatter(
        [start[0]],
        [start[1]],
        s=46,
        facecolors="white",
        edgecolors=[start_color],
        linewidths=1.35,
        zorder=zorder + 0.05,
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        s=56,
        c=[end_color],
        edgecolors=[_rgba("#1C2024", 0.95)],
        linewidths=0.55,
        zorder=zorder + 0.06,
    )


def _draw_global_velocity_stream(
    ax: plt.Axes,
    *,
    projected_paths: np.ndarray,
    clip_patch: Polygon,
    card: CardSpec,
    color: str,
    zorder: float,
) -> int:
    paths = np.asarray(projected_paths, dtype=float)
    if paths.ndim != 3 or paths.shape[2] != 2 or paths.shape[0] < 3:
        return 0
    p0 = paths[:-1].reshape(-1, 2)
    p1 = paths[1:].reshape(-1, 2)
    vel = p1 - p0
    speed = np.linalg.norm(vel, axis=1)
    keep = speed > 1e-4
    if int(np.sum(keep)) < 12:
        return 0
    pos = p0[keep]
    vel = vel[keep]

    x_grid = np.linspace(float(card.x + 0.08), float(card.x + card.w - 0.10), 26, dtype=float)
    y_grid = np.linspace(float(card.y + 0.08), float(card.y + card.h + card.shear - 0.08), 24, dtype=float)
    X, Y = np.meshgrid(x_grid, y_grid)
    sigma = 0.16 * max(float(card.w), float(card.h))
    U = np.full_like(X, np.nan, dtype=float)
    V = np.full_like(Y, np.nan, dtype=float)
    for iy in range(X.shape[0]):
        for ix in range(X.shape[1]):
            dx = pos[:, 0] - float(X[iy, ix])
            dy = pos[:, 1] - float(Y[iy, ix])
            dist2 = dx * dx + dy * dy
            weights = np.exp(-dist2 / max(2.0 * sigma * sigma, 1e-8))
            if float(np.sum(weights)) < 0.30:
                continue
            vec = np.sum(vel * weights[:, None], axis=0) / max(float(np.sum(weights)), 1e-8)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-4:
                continue
            U[iy, ix] = float(vec[0])
            V[iy, ix] = float(vec[1])
    if not np.isfinite(U).any() or not np.isfinite(V).any():
        return 0
    stream = ax.streamplot(
        x_grid,
        y_grid,
        np.ma.masked_invalid(U),
        np.ma.masked_invalid(V),
        density=1.0,
        linewidth=0.9,
        color=_rgba(color, 0.42),
        arrowsize=0.72,
        arrowstyle="-|>",
        minlength=0.10,
        maxlength=4.2,
        integration_direction="forward",
        zorder=zorder,
        broken_streamlines=True,
    )
    stream.lines.set_clip_path(clip_patch)
    stream.arrows.set_clip_path(clip_patch)
    return int(len(stream.lines.get_paths()))


def _connector_edge_points(
    projected_points: np.ndarray,
    *,
    card: CardSpec,
    ribbon: RibbonSpec,
    n_bands: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(projected_points, dtype=float)
    q = np.linspace(0.22, 0.78, int(max(n_bands, 2)), dtype=float)
    card_y0 = float(card.y + card.shear + 0.06)
    card_y1 = float(card.y + card.h + card.shear - 0.06)
    if len(pts) >= 6:
        source_y = np.quantile(pts[:, 1], q).astype(float)
    elif len(pts) > 0:
        source_y = np.interp(q, np.linspace(0.0, 1.0, len(pts)), np.sort(pts[:, 1])).astype(float)
    else:
        source_y = np.linspace(card_y0 + 0.18, card_y1 - 0.18, len(q), dtype=float)
    source_y = np.clip(source_y, card_y0, card_y1)
    start_x = float(card.x + card.w + 0.04)
    start_pts = np.column_stack([np.full(len(source_y), start_x, dtype=float), source_y]).astype(float)

    center_left = float(_ribbon_center_y(ribbon, ribbon.x0))
    half_band = min(0.15, 0.22 * float(ribbon.height))
    target_y = np.linspace(center_left - half_band, center_left + half_band, len(q), dtype=float)
    end_x = float(ribbon.x0 - 0.05)
    end_pts = np.column_stack([np.full(len(target_y), end_x, dtype=float), target_y]).astype(float)
    return start_pts, end_pts


def _draw_mist_ribbon_connector(
    ax: plt.Axes,
    *,
    projected_day0_points: np.ndarray,
    card: CardSpec,
    ribbon: RibbonSpec,
    modality: str,
    zorder: float,
) -> Dict[str, Any]:
    start_pts, end_pts = _connector_edge_points(
        np.asarray(projected_day0_points, dtype=float),
        card=card,
        ribbon=ribbon,
        n_bands=3,
    )
    gap = max(float(end_pts[0, 0] - start_pts[0, 0]), 0.18)
    day_colors = [0.0, 0.55, 1.15]
    band_summary: List[Dict[str, Any]] = []
    for i, (p0, p1, tval) in enumerate(zip(start_pts, end_pts, day_colors)):
        p0a = np.asarray(p0, dtype=float)
        p1a = np.asarray(p1, dtype=float)
        span_y = float(p1a[1] - p0a[1])
        bend = 0.05 if str(modality).lower() == "rna" else -0.05
        ctrl1 = np.asarray([float(p0a[0] + 0.28 * gap), float(p0a[1] + 0.18 * span_y + bend)], dtype=float)
        ctrl2 = np.asarray([float(p1a[0] - 0.34 * gap), float(p1a[1] - 0.10 * span_y + 0.4 * bend)], dtype=float)
        path = MplPath(
            [
                (float(p0a[0]), float(p0a[1])),
                (float(ctrl1[0]), float(ctrl1[1])),
                (float(ctrl2[0]), float(ctrl2[1])),
                (float(p1a[0]), float(p1a[1])),
            ],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        outer_color = _time_gradient_rgba(float(tval), alpha=0.08)
        mid_color = _time_gradient_rgba(float(tval), alpha=0.15)
        core_color = _time_gradient_rgba(float(tval), alpha=0.22)
        for width, color, extra in [(14.0, outer_color, 0.0), (9.0, mid_color, 0.05), (4.4, core_color, 0.10)]:
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=float(width),
                    capstyle="round",
                    joinstyle="round",
                    zorder=zorder + extra,
                )
            )
        band_summary.append(
            {
                "start": [float(p0a[0]), float(p0a[1])],
                "end": [float(p1a[0]), float(p1a[1])],
                "color_day_hint": float(tval),
            }
        )
    return {
        "modality": str(modality),
        "origin": "global_right_outer_edge_aligned_to_day0_real_data",
        "target": "d0_left_outer_edge",
        "bands": band_summary,
    }


def _add_time_gradient_legend(fig: plt.Figure, *, start_label: str, end_label: str) -> None:
    leg_ax = fig.add_axes([0.70, 0.83, 0.19, 0.09], zorder=20)
    leg_ax.set_facecolor("none")
    grad = np.linspace(0.0, 9.0, 320, dtype=float)[None, :]
    rgba = np.asarray([_time_gradient_rgba(float(t), alpha=1.0) for t in grad[0].tolist()], dtype=float)[None, :, :]
    leg_ax.imshow(rgba, aspect="auto", extent=(0.0, 9.0, 0.0, 1.0), interpolation="bicubic")
    leg_ax.plot([0.0, 9.0], [0.0, 0.0], color=_rgba(TRAJECTORY_CONTOUR_COLOR, 0.18), lw=0.8)
    leg_ax.plot([0.0, 9.0], [1.0, 1.0], color=_rgba(TRAJECTORY_CONTOUR_COLOR, 0.18), lw=0.8)
    for side in leg_ax.spines.values():
        side.set_visible(False)
    leg_ax.set_xticks([0.0, 9.0], labels=[str(start_label), str(end_label)])
    leg_ax.tick_params(axis="x", labelsize=8.5, colors=SUBTITLE_COLOR, length=0, pad=3)
    leg_ax.set_yticks([])
    leg_ax.text(0.0, 1.26, "Time gradient", fontsize=9.6, color=TEXT_COLOR, ha="left", va="bottom", transform=leg_ax.transData)
    leg_ax.text(9.0, 1.26, "cold to warm", fontsize=8.6, color=SUBTITLE_COLOR, ha="right", va="bottom", transform=leg_ax.transData)
    leg_ax.set_xlim(0.0, 9.0)
    leg_ax.set_ylim(-0.1, 1.45)


def _add_identity_table(fig: plt.Figure, traj_ids: Sequence[int]) -> List[Dict[str, Any]]:
    table_ax = fig.add_axes([0.80, 0.24, 0.16, 0.28], zorder=20)
    table_ax.set_facecolor(IDENTITY_TABLE_FACE)
    for spine in table_ax.spines.values():
        spine.set_edgecolor(IDENTITY_TABLE_EDGE)
        spine.set_linewidth(0.9)
    table_ax.set_xticks([])
    table_ax.set_yticks([])
    table_ax.set_xlim(0.0, 1.0)
    table_ax.set_ylim(0.0, 1.0)
    table_ax.text(0.08, 0.91, "Trajectory identities", fontsize=10.0, color=TEXT_COLOR, fontweight="bold", ha="left", va="center")
    table_ax.text(0.08, 0.82, "separate reference table", fontsize=8.2, color=SUBTITLE_COLOR, ha="left", va="center")
    table_ax.plot([0.08, 0.92], [0.76, 0.76], color=_rgba(IDENTITY_TABLE_EDGE, 0.80), lw=0.8)

    entries: List[Dict[str, Any]] = []
    y_positions = np.linspace(0.66, 0.12, max(len(traj_ids), 1), dtype=float)
    for i, (y, traj_id) in enumerate(zip(y_positions.tolist(), traj_ids), start=1):
        table_ax.text(0.10, y, f"T{i}", fontsize=9.0, color=TEXT_COLOR, fontweight="bold", ha="left", va="center")
        table_ax.text(0.29, y, f"Traj {int(traj_id)}", fontsize=9.0, color=TEXT_COLOR, ha="left", va="center")
        table_ax.text(0.90, y, f"#{int(traj_id)}", fontsize=8.3, color=SUBTITLE_COLOR, ha="right", va="center")
        if i < len(traj_ids):
            table_ax.plot([0.10, 0.90], [y - 0.055, y - 0.055], color=_rgba(IDENTITY_TABLE_EDGE, 0.46), lw=0.6)
        entries.append({"slot": int(i), "label": f"T{i}", "trajectory_index": int(traj_id)})
    return entries


def _expand_bounds(points: np.ndarray, pad_ratio: float = 0.08) -> Tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return (-1.0, 1.0, -1.0, 1.0)
    x0, x1 = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y0, y1 = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    span = max(x1 - x0, y1 - y0, 1.0)
    pad = float(pad_ratio) * span
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def _normalize_points(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if len(pts) == 0:
        return pts
    pts -= np.asarray(center, dtype=float)[None, :]
    return pts


def _card_polygon(card: CardSpec) -> np.ndarray:
    return np.asarray(
        [
            [float(card.x), float(card.y)],
            [float(card.x + card.w), float(card.y + card.shear)],
            [float(card.x + card.w), float(card.y + card.h + card.shear)],
            [float(card.x), float(card.y + card.h)],
        ],
        dtype=float,
    )


def _add_card_patch(
    ax: plt.Axes,
    card: CardSpec,
    *,
    facecolor: str,
    edgecolor: str,
    zorder: float,
) -> Polygon:
    shadow_poly = _card_polygon(CardSpec(card.x + 0.045, card.y - 0.045, card.w, card.h, card.shear, card.label, card.kind))
    ax.add_patch(
        Polygon(
            shadow_poly,
            closed=True,
            facecolor=SHADOW_COLOR,
            edgecolor="none",
            zorder=zorder - 0.2,
        )
    )
    patch = Polygon(
        _card_polygon(card),
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _project_to_card(points: np.ndarray, bounds: Tuple[float, float, float, float], card: CardSpec) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected [N,2] points, got shape={pts.shape}")
    x0, x1, y0, y1 = bounds
    dx = max(float(x1 - x0), 1e-9)
    dy = max(float(y1 - y0), 1e-9)
    nx = (pts[:, 0] - float(x0)) / dx
    ny = (pts[:, 1] - float(y0)) / dy
    xw = float(card.x) + nx * float(card.w)
    yw = float(card.y) + ny * float(card.h) + nx * float(card.shear)
    return np.column_stack([xw, yw]).astype(float)


def _project_grid_to_card(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bounds: Tuple[float, float, float, float],
    card: CardSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = bounds
    dx = max(float(x1 - x0), 1e-9)
    dy = max(float(y1 - y0), 1e-9)
    nx = (np.asarray(x_grid, dtype=float) - float(x0)) / dx
    ny = (np.asarray(y_grid, dtype=float) - float(y0)) / dy
    xw = float(card.x) + nx * float(card.w)
    yw = float(card.y) + ny * float(card.h) + nx * float(card.shear)
    return np.asarray(xw, dtype=float), np.asarray(yw, dtype=float)


def _ribbon_center_y(ribbon: RibbonSpec, x_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x_val, dtype=float)
    rel = (arr - float(ribbon.x0)) / max(float(ribbon.width), 1e-9)
    return float(ribbon.y0) + rel * float(ribbon.slope)


def _ribbon_polygon(ribbon: RibbonSpec) -> np.ndarray:
    x0 = float(ribbon.x0)
    x1 = float(ribbon.x0 + ribbon.width)
    c0 = float(_ribbon_center_y(ribbon, x0))
    c1 = float(_ribbon_center_y(ribbon, x1))
    half_h = 0.5 * float(ribbon.height)
    return np.asarray(
        [
            [x0, c0 - half_h],
            [x1, c1 - half_h],
            [x1, c1 + half_h],
            [x0, c0 + half_h],
        ],
        dtype=float,
    )


def _add_ribbon_patch(
    ax: plt.Axes,
    ribbon: RibbonSpec,
    *,
    facecolor: str,
    edgecolor: str,
    zorder: float,
) -> Polygon:
    shadow = _ribbon_polygon(
        RibbonSpec(
            x0=float(ribbon.x0) + 0.05,
            y0=float(ribbon.y0) - 0.05,
            width=float(ribbon.width),
            height=float(ribbon.height),
            slope=float(ribbon.slope),
            label=ribbon.label,
            kind=ribbon.kind,
        )
    )
    ax.add_patch(
        Polygon(
            shadow,
            closed=True,
            facecolor=SHADOW_COLOR,
            edgecolor="none",
            zorder=zorder - 0.2,
        )
    )
    patch = Polygon(
        _ribbon_polygon(ribbon),
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _build_station_positions(anchor_days: np.ndarray, time_x_step: float) -> np.ndarray:
    days = np.asarray(anchor_days, dtype=float)
    if len(days) == 0:
        raise ValueError("anchor_days must be non-empty")
    gap_weights = np.ones(max(len(days) - 1, 0), dtype=float)
    x_positions = [3.02]
    for weight in gap_weights.tolist():
        x_positions.append(float(x_positions[-1] + float(time_x_step) * float(weight)))
    return np.asarray(x_positions, dtype=float)


def _project_times_to_ribbon(
    times: np.ndarray,
    anchor_days: np.ndarray,
    station_x: np.ndarray,
) -> np.ndarray:
    return np.interp(np.asarray(times, dtype=float), np.asarray(anchor_days, dtype=float), np.asarray(station_x, dtype=float))


def _project_path_to_ribbon(
    points: np.ndarray,
    *,
    times: np.ndarray,
    bounds: Tuple[float, float, float, float],
    ribbon: RibbonSpec,
    anchor_days: np.ndarray,
    station_x: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected [N,2] points, got shape={pts.shape}")
    x_times = _project_times_to_ribbon(times, anchor_days, station_x)
    x0, x1, y0, y1 = bounds
    nx = (pts[:, 0] - float(x0)) / max(float(x1 - x0), 1e-9)
    window_width = 0.82
    x_proj = x_times - 0.5 * float(window_width) + nx * float(window_width)
    ny = (pts[:, 1] - float(y0)) / max(float(y1 - y0), 1e-9)
    center = _ribbon_center_y(ribbon, x_proj)
    y_proj = center - 0.5 * float(ribbon.height) + ny * float(ribbon.height)
    return np.column_stack([x_proj, y_proj]).astype(float)


def _project_points_to_station_window(
    points: np.ndarray,
    *,
    bounds: Tuple[float, float, float, float],
    ribbon: RibbonSpec,
    station_x: float,
    window_width: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected [N,2] points, got shape={pts.shape}")
    x0, x1, y0, y1 = bounds
    nx = (pts[:, 0] - float(x0)) / max(float(x1 - x0), 1e-9)
    ny = (pts[:, 1] - float(y0)) / max(float(y1 - y0), 1e-9)
    x_proj = float(station_x) - 0.5 * float(window_width) + nx * float(window_width)
    center = _ribbon_center_y(ribbon, x_proj)
    y_proj = center - 0.5 * float(ribbon.height) + ny * float(ribbon.height)
    return np.column_stack([x_proj, y_proj]).astype(float)


def _draw_station_separator(
    ax: plt.Axes,
    *,
    ribbon: RibbonSpec,
    station_x: float,
    color: str,
    linewidth: float,
    alpha: float,
    clip_patch: Polygon,
    zorder: float,
) -> None:
    center = float(_ribbon_center_y(ribbon, station_x))
    y0 = center - 0.5 * float(ribbon.height)
    y1 = center + 0.5 * float(ribbon.height)
    line = ax.plot(
        [float(station_x), float(station_x)],
        [float(y0), float(y1)],
        color=_rgba(color, alpha),
        linewidth=float(linewidth),
        zorder=zorder,
    )[0]
    line.set_clip_path(clip_patch)


def _draw_station_window_tint(
    ax: plt.Axes,
    *,
    ribbon: RibbonSpec,
    station_x: float,
    window_width: float,
    color: str,
    alpha: float,
    clip_patch: Polygon,
    zorder: float,
) -> None:
    x0 = float(station_x) - 0.5 * float(window_width)
    x1 = float(station_x) + 0.5 * float(window_width)
    c0 = float(_ribbon_center_y(ribbon, x0))
    c1 = float(_ribbon_center_y(ribbon, x1))
    hh = 0.5 * float(ribbon.height)
    poly = np.asarray(
        [
            [x0, c0 - hh],
            [x1, c1 - hh],
            [x1, c1 + hh],
            [x0, c0 + hh],
        ],
        dtype=float,
    )
    patch = Polygon(
        poly,
        closed=True,
        facecolor=_rgba(color, float(alpha)),
        edgecolor="none",
        zorder=zorder,
    )
    patch.set_clip_path(clip_patch)
    ax.add_patch(patch)


def _draw_station_fog(
    ax: plt.Axes,
    *,
    points: np.ndarray,
    bounds: Tuple[float, float, float, float],
    ribbon: RibbonSpec,
    station_x: float,
    base_color: str,
    clip_patch: Polygon,
    prominence: str,
    max_points: int,
    zorder: float,
) -> int:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 10:
        return 0
    if int(max_points) > 0 and len(pts) > int(max_points):
        idx = _subsample_indices(len(pts), int(max_points), np.random.default_rng(len(pts) + int(station_x * 1000.0)))
        pts = pts[idx]
    window_width = 0.88 if prominence == "key" else 0.74
    proj = _project_points_to_station_window(
        pts,
        bounds=bounds,
        ribbon=ribbon,
        station_x=float(station_x),
        window_width=float(window_width),
    )
    x_grid = np.linspace(float(station_x) - 0.5 * window_width, float(station_x) + 0.5 * window_width, 48, dtype=float)
    y_pad = 0.06 if prominence == "key" else 0.03
    center = _ribbon_center_y(ribbon, x_grid)
    y_low = np.min(center - 0.5 * float(ribbon.height)) - float(y_pad)
    y_high = np.max(center + 0.5 * float(ribbon.height)) + float(y_pad)
    y_grid = np.linspace(float(y_low), float(y_high), 44, dtype=float)
    hist, xe, ye = np.histogram2d(proj[:, 0], proj[:, 1], bins=[x_grid, y_grid])
    dense = gaussian_filter(hist.T.astype(float), sigma=1.2 if prominence == "key" else 1.0)
    if not np.isfinite(dense).any() or float(np.max(dense)) <= 0.0:
        return 0
    dense = dense / float(np.max(dense))
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    Xc, Yc = np.meshgrid(xc, yc)
    base_rgb = _soften_color(base_color, white_mix=0.35 if prominence == "key" else 0.48, saturation_keep=0.80)
    alphas = [0.05, 0.09, 0.14, 0.22] if prominence == "key" else [0.03, 0.06, 0.10, 0.15]
    colors = [tuple(list(base_rgb) + [a]) for a in alphas]
    levels = [0.14, 0.28, 0.46, 0.66, 1.01]
    cs = ax.contourf(Xc, Yc, dense, levels=levels, colors=colors, antialiased=True, zorder=zorder)
    cs.set_clip_path(clip_patch)
    return max(int(len(levels) - 1), 0)


def _draw_station_crosses(
    ax: plt.Axes,
    *,
    points: np.ndarray,
    bounds: Tuple[float, float, float, float],
    ribbon: RibbonSpec,
    station_x: float,
    clip_patch: Polygon,
    max_points: int,
    color: str,
    alpha: float,
    zorder: float,
) -> int:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return 0
    if int(max_points) > 0 and len(pts) > int(max_points):
        idx = _subsample_indices(len(pts), int(max_points), np.random.default_rng(len(pts) + int(station_x * 1000.0)))
        pts = pts[idx]
    # Reuse the original station-window projection so real-data placement stays
    # consistent with the previous Di panel layout.
    proj = _project_points_to_station_window(
        pts,
        bounds=bounds,
        ribbon=ribbon,
        station_x=float(station_x),
        window_width=0.82,
    )
    artist = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        s=14.0,
        marker="x",
        c=[_rgba(color, alpha)],
        linewidths=0.7,
        zorder=zorder,
    )
    artist.set_clip_path(clip_patch)
    return int(len(proj))


def _subsample_indices(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=int)
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    return np.sort(rng.choice(n, size=int(max_points), replace=False).astype(int))


def _draw_time_density_cloud(
    ax: plt.Axes,
    *,
    points: np.ndarray,
    times: np.ndarray,
    bounds: Tuple[float, float, float, float],
    card: CardSpec,
    time_palette: Dict[float, Tuple[float, float, float]],
    clip_patch: Polygon,
    bins: int = 72,
    sigma: float = 1.5,
    alpha_ramp: Optional[Sequence[float]] = None,
    zorder: float = 2.0,
) -> int:
    pts = np.asarray(points, dtype=float)
    tvals = np.asarray(times, dtype=float)
    if len(pts) == 0 or len(tvals) != len(pts):
        return 0

    x0, x1, y0, y1 = bounds
    xedges = np.linspace(float(x0), float(x1), int(bins) + 1, dtype=float)
    yedges = np.linspace(float(y0), float(y1), int(bins) + 1, dtype=float)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xc, yc)
    Xw, Yw = _project_grid_to_card(Xc, Yc, bounds, card)

    drawn = 0
    unique_times = sorted({float(t) for t in np.unique(tvals).tolist()})
    alpha_levels = list(alpha_ramp) if alpha_ramp is not None else [0.05, 0.08, 0.12, 0.18]
    for t in unique_times:
        mask = np.isclose(tvals, float(t), atol=1e-6)
        sub = pts[mask]
        if len(sub) < 8:
            continue
        hist, _, _ = np.histogram2d(sub[:, 0], sub[:, 1], bins=[xedges, yedges])
        dense = gaussian_filter(hist.T.astype(float), sigma=float(sigma))
        if not np.isfinite(dense).any() or float(np.max(dense)) <= 0.0:
            continue
        dense = dense / float(np.max(dense))
        levels = [0.16, 0.32, 0.54, 0.78, 1.01]
        base_rgb = time_palette.get(float(t), _soften_color("#9BB7C9"))
        colors = [tuple(list(base_rgb) + [float(alpha)]) for alpha in alpha_levels]
        cs = ax.contourf(Xw, Yw, dense, levels=levels, colors=colors, antialiased=True, zorder=zorder)
        cs.set_clip_path(clip_patch)
        drawn += max(int(len(levels) - 1), 0)
    return drawn


def _draw_sparse_time_points(
    ax: plt.Axes,
    *,
    points: np.ndarray,
    times: np.ndarray,
    bounds: Tuple[float, float, float, float],
    card: CardSpec,
    time_palette: Dict[float, Tuple[float, float, float]],
    clip_patch: Polygon,
    rng: np.random.Generator,
    max_points: int,
    point_alpha: float = 0.28,
    zorder: float,
) -> int:
    pts = np.asarray(points, dtype=float)
    tvals = np.asarray(times, dtype=float)
    if len(pts) == 0 or len(tvals) != len(pts):
        return 0
    idx = _subsample_indices(len(pts), int(max_points), rng)
    proj = _project_to_card(pts[idx], bounds, card)
    colors = [tuple(list(time_palette.get(float(tvals[i]), _soften_color("#8CAABD"))) + [float(point_alpha)]) for i in idx.tolist()]
    artist = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        s=7.0,
        c=colors,
        linewidths=0.0,
        zorder=zorder,
    )
    artist.set_clip_path(clip_patch)
    return int(len(idx))


def _draw_time_card_points(
    ax: plt.Axes,
    *,
    points: np.ndarray,
    bounds: Tuple[float, float, float, float],
    card: CardSpec,
    clip_patch: Polygon,
    color: str,
    alpha: float,
    max_points: int,
    rng: np.random.Generator,
    zorder: float,
) -> int:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return 0
    idx = _subsample_indices(len(pts), int(max_points), rng)
    proj = _project_to_card(pts[idx], bounds, card)
    artist = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        s=8.2,
        c=[_rgba(color, alpha)],
        linewidths=0.0,
        zorder=zorder,
    )
    artist.set_clip_path(clip_patch)
    return int(len(idx))


def _draw_double_line(ax: plt.Axes, path_xy: np.ndarray, color: str, zorder: float) -> None:
    ax.plot(
        path_xy[:, 0],
        path_xy[:, 1],
        color=_rgba(color, 0.22),
        linewidth=4.8,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )
    ax.plot(
        path_xy[:, 0],
        path_xy[:, 1],
        color=_rgba(color, 0.98),
        linewidth=2.15,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder + 0.1,
    )


def _draw_anchor_markers(
    ax: plt.Axes,
    *,
    anchor_points: np.ndarray,
    color: str,
    zorder: float,
) -> None:
    if len(anchor_points) == 0:
        return
    if len(anchor_points) > 2:
        mid = anchor_points[1:-1]
        ax.scatter(
            mid[:, 0],
            mid[:, 1],
            s=12,
            c=[_rgba(color, 0.95)],
            linewidths=0.0,
            zorder=zorder,
        )
    start = anchor_points[0]
    end = anchor_points[-1]
    ax.scatter(
        [start[0]],
        [start[1]],
        s=44,
        facecolors="white",
        edgecolors=[_rgba(color, 1.0)],
        linewidths=1.3,
        zorder=zorder + 0.05,
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        s=52,
        c=[_rgba(color, 1.0)],
        edgecolors=[_rgba("#1C2024", 0.95)],
        linewidths=0.5,
        zorder=zorder + 0.06,
    )


def _draw_soft_bridge(
    ax: plt.Axes,
    *,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    direction: str,
    color: str,
    zorder: float,
) -> None:
    p0a = np.asarray(p0, dtype=float)
    p1a = np.asarray(p1, dtype=float)
    dx = float(p1a[0] - p0a[0])
    direction_sign = -1.0 if str(direction).lower() == "left" else 1.0
    bend = max(abs(dx) * 0.45, 0.42)
    ctrl1 = np.asarray([p0a[0] + direction_sign * bend, p0a[1] - 0.10], dtype=float)
    ctrl2 = np.asarray([p1a[0] + direction_sign * bend, p1a[1] + 0.10], dtype=float)

    verts = [
        (float(p0a[0]), float(p0a[1])),
        (float(ctrl1[0]), float(ctrl1[1])),
        (float(ctrl2[0]), float(ctrl2[1])),
        (float(p1a[0]), float(p1a[1])),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    path = MplPath(verts, codes)
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=_rgba(color, 0.20),
            linewidth=6.8,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=_rgba(color, 0.92),
            linewidth=2.1,
            capstyle="round",
            joinstyle="round",
            zorder=zorder + 0.05,
        )
    )


def _draw_resonance_bridge(
    ax: plt.Axes,
    *,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: str,
    zorder: float,
) -> None:
    p0a = np.asarray(p0, dtype=float)
    p1a = np.asarray(p1, dtype=float)
    x_shift = 0.12
    ctrl1 = np.asarray([0.5 * (p0a[0] + p1a[0]) - x_shift, p0a[1] - 0.18], dtype=float)
    ctrl2 = np.asarray([0.5 * (p0a[0] + p1a[0]) + x_shift, p1a[1] + 0.18], dtype=float)
    path = MplPath(
        [
            (float(p0a[0]), float(p0a[1])),
            (float(ctrl1[0]), float(ctrl1[1])),
            (float(ctrl2[0]), float(ctrl2[1])),
            (float(p1a[0]), float(p1a[1])),
        ],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=_rgba(color, 0.18),
            linewidth=5.0,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=_rgba(color, 0.90),
            linewidth=1.7,
            capstyle="round",
            joinstyle="round",
            zorder=zorder + 0.05,
        )
    )


def _build_ribbon_layout(
    anchor_days: np.ndarray,
    *,
    time_x_step: float,
    time_rise_step: float,
    space_gap_factor: float,
) -> Tuple[CardSpec, CardSpec, RibbonSpec, RibbonSpec, np.ndarray]:
    days = np.asarray(anchor_days, dtype=float)
    if len(days) == 0:
        raise ValueError("anchor_days must be non-empty")

    overview_w = 1.46
    overview_h = 1.14
    overview_rna = CardSpec(x=0.0, y=2.64, w=overview_w, h=overview_h, shear=0.08, label="Global", kind="overview_rna")

    station_x = _build_station_positions(days, float(time_x_step))
    ribbon_x0 = float(station_x[0] - 0.42)
    ribbon_width = float(station_x[-1] - station_x[0] + 0.92)
    rna_y0 = 3.16
    row_gap = 1.72 * float(space_gap_factor) / 1.35
    atac_y0 = rna_y0 - row_gap
    ribbon_slope = float(time_rise_step) * float(len(days) - 1)
    ribbon_h = 0.86

    rna_ribbon = RibbonSpec(
        x0=ribbon_x0,
        y0=rna_y0,
        width=ribbon_width,
        height=ribbon_h,
        slope=ribbon_slope,
        label="RNA ribbon corridor",
        kind="rna_ribbon",
    )
    atac_ribbon = RibbonSpec(
        x0=ribbon_x0,
        y0=atac_y0,
        width=ribbon_width,
        height=ribbon_h,
        slope=ribbon_slope * 0.98,
        label="ATAC ribbon corridor",
        kind="atac_ribbon",
    )
    overview_atac = CardSpec(
        x=0.0,
        y=float(atac_y0 - 0.5 * ribbon_h - 0.26),
        w=overview_w,
        h=overview_h,
        shear=0.08,
        label="Global",
        kind="overview_atac",
    )
    return overview_rna, overview_atac, rna_ribbon, atac_ribbon, station_x


def _figure_png_bytes(fig: plt.Figure, dpi: int = 220) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, dpi=int(dpi), bbox_inches="tight", facecolor=fig.get_facecolor())
    return buf.getvalue()


def _write_html_preview(html_path: Path, png_bytes: bytes, title: str, subtitle: str) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f5; color: #24323e; }}
    .wrap {{ max-width: 1800px; margin: 0 auto; padding: 20px 24px 28px; }}
    h1 {{ margin: 0 0 6px; font-family: 'STIXGeneral', 'Liberation Serif', serif; font-size: 28px; font-weight: 700; }}
    p {{ margin: 0 0 16px; color: #586977; font-size: 15px; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 12px; box-shadow: 0 12px 36px rgba(20, 26, 32, 0.12); background: white; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>{title}</h1>
    <p>{subtitle}</p>
    <img alt=\"{title}\" src=\"data:image/png;base64,{encoded}\" />
  </div>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


def _render_reference_2d_panel(
    *,
    rna_bg: np.ndarray,
    atac_bg: np.ndarray,
    time_key_values: np.ndarray,
    time_palette: Dict[float, Tuple[float, float, float]],
    traj_rna_2d: np.ndarray,
    traj_atac_2d: np.ndarray,
    traj_ids: Sequence[int],
    color_by_traj: Dict[int, str],
    output_png: Path,
) -> str:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.grid": False,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), dpi=300, facecolor="#FCFBF9")
    panels = [(axes[0], rna_bg, traj_rna_2d, "RNA embedding reference"), (axes[1], atac_bg, traj_atac_2d, "ATAC embedding reference")]
    bg_colors = [time_palette.get(float(t), _soften_color("#9DB4C1")) for t in np.asarray(time_key_values, dtype=float).tolist()]
    for ax, bg, traj, title in panels:
        ax.scatter(bg[:, 0], bg[:, 1], s=5.0, c=bg_colors, alpha=0.14, linewidths=0, rasterized=True, zorder=1)
        for j, traj_id in enumerate(traj_ids):
            path = np.asarray(traj[:, j, :], dtype=float)
            color = str(color_by_traj[int(traj_id)])
            ax.plot(path[:, 0], path[:, 1], color=_rgba(color, 0.22), linewidth=4.0, zorder=2)
            ax.plot(path[:, 0], path[:, 1], color=_rgba(color, 0.98), linewidth=1.8, zorder=2.1)
            ax.scatter(path[0, 0], path[0, 1], s=24, facecolors="white", edgecolors=color, linewidths=1.0, zorder=2.2)
            ax.scatter(path[-1, 0], path[-1, 1], s=28, c=[color], edgecolors="#1C2024", linewidths=0.35, zorder=2.3)
        ax.set_title(title, fontsize=11, color=TEXT_COLOR)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        for spine in ax.spines.values():
            spine.set_color("#B8C1C8")
            spine.set_linewidth(0.8)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(output_png)


def _render_global_corridor_figure(
    *,
    rna_bg_all: np.ndarray,
    atac_bg_all: np.ndarray,
    rna_bg_by_time: Dict[float, np.ndarray],
    atac_bg_by_time: Dict[float, np.ndarray],
    rna_times_all: np.ndarray,
    atac_times_all: np.ndarray,
    traj_rna_display: np.ndarray,
    traj_atac_display: np.ndarray,
    traj_rna_anchor: np.ndarray,
    traj_atac_anchor: np.ndarray,
    display_times: np.ndarray,
    display_is_anchor: np.ndarray,
    station_times: np.ndarray,
    station_is_real: np.ndarray,
    station_labels: Sequence[str],
    traj_ids: Sequence[int],
    color_by_traj: Dict[int, str],
    bridge_station_indices: Sequence[int],
    overview_rna_card: CardSpec,
    overview_atac_card: CardSpec,
    rna_ribbon: RibbonSpec,
    atac_ribbon: RibbonSpec,
    station_x: np.ndarray,
    output_html: Path,
    output_png: Path,
    output_svg: Path,
    output_pdf: Path,
    export_static: bool,
    overview_max_points: int,
    time_card_max_points: int,
    seed: int,
) -> Dict[str, Any]:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    all_rna = np.vstack([np.asarray(rna_bg_all, dtype=float), np.asarray(traj_rna_display, dtype=float).reshape(-1, 2)])
    all_atac = np.vstack([np.asarray(atac_bg_all, dtype=float), np.asarray(traj_atac_display, dtype=float).reshape(-1, 2)])
    rna_center = np.mean(all_rna, axis=0)
    atac_center = np.mean(all_atac, axis=0)
    rna_bg_all = _normalize_points(rna_bg_all, rna_center)
    atac_bg_all = _normalize_points(atac_bg_all, atac_center)
    traj_rna_display = _normalize_points(np.asarray(traj_rna_display, dtype=float).reshape(-1, 2), rna_center).reshape(np.asarray(traj_rna_display).shape)
    traj_atac_display = _normalize_points(np.asarray(traj_atac_display, dtype=float).reshape(-1, 2), atac_center).reshape(np.asarray(traj_atac_display).shape)
    traj_rna_anchor = _normalize_points(np.asarray(traj_rna_anchor, dtype=float).reshape(-1, 2), rna_center).reshape(np.asarray(traj_rna_anchor).shape)
    traj_atac_anchor = _normalize_points(np.asarray(traj_atac_anchor, dtype=float).reshape(-1, 2), atac_center).reshape(np.asarray(traj_atac_anchor).shape)
    rna_bg_by_time = {float(k): _normalize_points(v, rna_center) for k, v in rna_bg_by_time.items()}
    atac_bg_by_time = {float(k): _normalize_points(v, atac_center) for k, v in atac_bg_by_time.items()}
    rna_bounds = _expand_bounds(np.vstack([rna_bg_all, traj_rna_display.reshape(-1, 2)]), pad_ratio=0.08)
    atac_bounds = _expand_bounds(np.vstack([atac_bg_all, traj_atac_display.reshape(-1, 2)]), pad_ratio=0.08)
    traj_rna_overview = np.stack(
        [_project_to_card(np.asarray(traj_rna_display[:, j, :], dtype=float), rna_bounds, overview_rna_card) for j in range(np.asarray(traj_rna_display).shape[1])],
        axis=1,
    ) if np.asarray(traj_rna_display).shape[1] > 0 else np.zeros((0, 0, 2), dtype=float)
    traj_atac_overview = np.stack(
        [_project_to_card(np.asarray(traj_atac_display[:, j, :], dtype=float), atac_bounds, overview_atac_card) for j in range(np.asarray(traj_atac_display).shape[1])],
        axis=1,
    ) if np.asarray(traj_atac_display).shape[1] > 0 else np.zeros((0, 0, 2), dtype=float)
    global_time_palette = _make_global_time_palette(np.concatenate([np.asarray(rna_times_all, dtype=float), np.asarray(atac_times_all, dtype=float)]))

    fig, ax = plt.subplots(figsize=(19.2, 9.3), dpi=300, facecolor="#FBFAF8")
    ax.set_facecolor("#FBFAF8")
    ax.set_aspect("equal")
    ax.axis("off")

    rna_overview_patch = _add_card_patch(ax, overview_rna_card, facecolor=RNA_CARD_FACE, edgecolor=RNA_CARD_EDGE, zorder=1.0)
    atac_overview_patch = _add_card_patch(ax, overview_atac_card, facecolor=ATAC_CARD_FACE, edgecolor=ATAC_CARD_EDGE, zorder=1.0)
    rna_ribbon_patch = _add_ribbon_patch(ax, rna_ribbon, facecolor=RNA_CARD_FACE, edgecolor=RNA_CARD_EDGE, zorder=1.0)
    atac_ribbon_patch = _add_ribbon_patch(ax, atac_ribbon, facecolor=ATAC_CARD_FACE, edgecolor=ATAC_CARD_EDGE, zorder=1.0)

    overview_density_layers = {
        "rna": _draw_time_density_cloud(
            ax,
            points=rna_bg_all,
            times=np.asarray(rna_times_all, dtype=float),
            bounds=rna_bounds,
            card=overview_rna_card,
            time_palette=global_time_palette,
            clip_patch=rna_overview_patch,
            bins=78,
            sigma=1.6,
            alpha_ramp=(0.025, 0.045, 0.070, 0.105),
            zorder=3.4,
        ),
        "atac": _draw_time_density_cloud(
            ax,
            points=atac_bg_all,
            times=np.asarray(atac_times_all, dtype=float),
            bounds=atac_bounds,
            card=overview_atac_card,
            time_palette=global_time_palette,
            clip_patch=atac_overview_patch,
            bins=78,
            sigma=1.6,
            alpha_ramp=(0.025, 0.045, 0.070, 0.105),
            zorder=3.4,
        ),
    }
    overview_point_counts = {
        "rna": _draw_sparse_time_points(
            ax,
            points=rna_bg_all,
            times=np.asarray(rna_times_all, dtype=float),
            bounds=rna_bounds,
            card=overview_rna_card,
            time_palette=global_time_palette,
            clip_patch=rna_overview_patch,
            rng=np.random.default_rng(int(seed)),
            max_points=int(overview_max_points),
            point_alpha=0.16,
            zorder=3.55,
        ),
        "atac": _draw_sparse_time_points(
            ax,
            points=atac_bg_all,
            times=np.asarray(atac_times_all, dtype=float),
            bounds=atac_bounds,
            card=overview_atac_card,
            time_palette=global_time_palette,
            clip_patch=atac_overview_patch,
            rng=np.random.default_rng(int(seed) + 1),
            max_points=int(overview_max_points),
            point_alpha=0.16,
            zorder=3.55,
        ),
    }
    overview_stream_counts = {
        "rna": _draw_global_velocity_stream(
            ax,
            projected_paths=traj_rna_overview,
            clip_patch=rna_overview_patch,
            card=overview_rna_card,
            color=RNA_CARD_EDGE,
            zorder=1.74,
        ),
        "atac": _draw_global_velocity_stream(
            ax,
            projected_paths=traj_atac_overview,
            clip_patch=atac_overview_patch,
            card=overview_atac_card,
            color=ATAC_CARD_EDGE,
            zorder=1.74,
        ),
    }

    real_station_idx = np.flatnonzero(np.asarray(station_is_real, dtype=bool))
    first_real_time = float(station_times[int(real_station_idx[0])]) if len(real_station_idx) > 0 else float(station_times[0])
    connector_summary = {
        "rna": _draw_mist_ribbon_connector(
            ax,
            projected_day0_points=_project_to_card(np.asarray(rna_bg_by_time.get(float(first_real_time), np.zeros((0, 2))), dtype=float), rna_bounds, overview_rna_card),
            card=overview_rna_card,
            ribbon=rna_ribbon,
            modality="rna",
            zorder=0.94,
        ),
        "atac": _draw_mist_ribbon_connector(
            ax,
            projected_day0_points=_project_to_card(np.asarray(atac_bg_by_time.get(float(first_real_time), np.zeros((0, 2))), dtype=float), atac_bounds, overview_atac_card),
            card=overview_atac_card,
            ribbon=atac_ribbon,
            modality="atac",
            zorder=0.94,
        ),
    }

    station_cross_points_rna: Dict[int, int] = {}
    station_cross_points_atac: Dict[int, int] = {}
    for i, (stime, sx, is_real) in enumerate(
        zip(np.asarray(station_times, dtype=float).tolist(), np.asarray(station_x, dtype=float).tolist(), np.asarray(station_is_real, dtype=bool).tolist())
    ):
        tint_alpha = 0.075 if bool(is_real) else 0.032
        _draw_station_window_tint(
            ax,
            ribbon=rna_ribbon,
            station_x=float(sx),
            window_width=0.82,
            color=RNA_CARD_EDGE,
            alpha=tint_alpha,
            clip_patch=rna_ribbon_patch,
            zorder=1.22,
        )
        _draw_station_window_tint(
            ax,
            ribbon=atac_ribbon,
            station_x=float(sx),
            window_width=0.82,
            color=ATAC_CARD_EDGE,
            alpha=tint_alpha,
            clip_patch=atac_ribbon_patch,
            zorder=1.22,
        )
        if not bool(is_real):
            station_cross_points_rna[int(i)] = 0
            station_cross_points_atac[int(i)] = 0
            continue

        station_cross_points_rna[int(i)] = _draw_station_crosses(
            ax,
            points=np.asarray(rna_bg_by_time.get(float(stime), np.zeros((0, 2))), dtype=float),
            bounds=rna_bounds,
            ribbon=rna_ribbon,
            station_x=float(sx),
            clip_patch=rna_ribbon_patch,
            max_points=int(time_card_max_points),
            color=STATION_FOG_GRAY,
            alpha=0.9,
            zorder=1.34,
        )
        station_cross_points_atac[int(i)] = _draw_station_crosses(
            ax,
            points=np.asarray(atac_bg_by_time.get(float(stime), np.zeros((0, 2))), dtype=float),
            bounds=atac_bounds,
            ribbon=atac_ribbon,
            station_x=float(sx),
            clip_patch=atac_ribbon_patch,
            max_points=int(time_card_max_points),
            color=STATION_FOG_GRAY,
            alpha=0.9,
            zorder=1.34,
        )

    if len(station_x) > 1:
        interval_x = 0.5 * (np.asarray(station_x[:-1], dtype=float) + np.asarray(station_x[1:], dtype=float))
    else:
        interval_x = np.zeros(0, dtype=float)
    for sx in interval_x.tolist():
        _draw_station_separator(
            ax,
            ribbon=rna_ribbon,
            station_x=float(sx),
            color=RNA_CARD_EDGE,
            linewidth=0.85,
            alpha=0.26,
            clip_patch=rna_ribbon_patch,
            zorder=1.42,
        )
        _draw_station_separator(
            ax,
            ribbon=atac_ribbon,
            station_x=float(sx),
            color=ATAC_CARD_EDGE,
            linewidth=0.85,
            alpha=0.26,
            clip_patch=atac_ribbon_patch,
            zorder=1.42,
        )
    for sx, is_real, label in zip(np.asarray(station_x, dtype=float).tolist(), np.asarray(station_is_real, dtype=bool).tolist(), list(station_labels)):
        if (not bool(is_real)) or (not str(label).strip()):
            continue
        label_y = float(_ribbon_center_y(rna_ribbon, sx) + 0.5 * float(rna_ribbon.height) + 0.11)
        ax.text(
            float(sx),
            label_y,
            str(label),
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="normal",
            color=SUBTITLE_COLOR,
            zorder=3.1,
        )

    anchor_idx = np.flatnonzero(np.asarray(display_is_anchor, dtype=bool))

    for j, traj_id in enumerate(traj_ids):
        _ = traj_id
        corridor_rna_path = _project_path_to_ribbon(
            np.asarray(traj_rna_display[:, j, :], dtype=float),
            times=np.asarray(display_times, dtype=float),
            bounds=rna_bounds,
            ribbon=rna_ribbon,
            anchor_days=station_times,
            station_x=station_x,
        )
        corridor_atac_path = _project_path_to_ribbon(
            np.asarray(traj_atac_display[:, j, :], dtype=float),
            times=np.asarray(display_times, dtype=float),
            bounds=atac_bounds,
            ribbon=atac_ribbon,
            anchor_days=station_times,
            station_x=station_x,
        )
        _draw_time_gradient_line(
            ax,
            path_xy=corridor_rna_path,
            times=np.asarray(display_times, dtype=float),
            zorder=2.42,
            outer_linewidth=5.1,
            inner_linewidth=2.35,
            outer_alpha=0.20,
        )
        _draw_time_gradient_line(
            ax,
            path_xy=corridor_atac_path,
            times=np.asarray(display_times, dtype=float),
            zorder=2.42,
            outer_linewidth=5.1,
            inner_linewidth=2.35,
            outer_alpha=0.20,
        )
        _draw_time_markers(
            ax,
            anchor_points=np.asarray(corridor_rna_path[anchor_idx], dtype=float),
            anchor_times=np.asarray(display_times[anchor_idx], dtype=float),
            zorder=2.64,
        )
        _draw_time_markers(
            ax,
            anchor_points=np.asarray(corridor_atac_path[anchor_idx], dtype=float),
            anchor_times=np.asarray(display_times[anchor_idx], dtype=float),
            zorder=2.64,
        )

    bridge_summary: Dict[int, Dict[str, Any]] = {}
    n_st = int(len(station_times))
    for bridge_idx_raw in [int(x) for x in bridge_station_indices]:
        idx = int(bridge_idx_raw)
        if idx < 0:
            idx = int(n_st + idx)
        if idx < 0 or idx >= n_st:
            continue
        rna_cent = np.mean(np.asarray(traj_rna_anchor[idx], dtype=float), axis=0)
        atac_cent = np.mean(np.asarray(traj_atac_anchor[idx], dtype=float), axis=0)
        stime = float(station_times[idx])
        p_rna = _project_path_to_ribbon(
            np.asarray(rna_cent[None, :], dtype=float),
            times=np.asarray([stime], dtype=float),
            bounds=rna_bounds,
            ribbon=rna_ribbon,
            anchor_days=station_times,
            station_x=station_x,
        )[0]
        p_atac = _project_path_to_ribbon(
            np.asarray(atac_cent[None, :], dtype=float),
            times=np.asarray([stime], dtype=float),
            bounds=atac_bounds,
            ribbon=atac_ribbon,
            anchor_days=station_times,
            station_x=station_x,
        )[0]
        _draw_resonance_bridge(
            ax,
            p0=(float(p_rna[0]), float(p_rna[1])),
            p1=(float(p_atac[0]), float(p_atac[1])),
            color=SYNC_BRIDGE_COLOR,
            zorder=2.86,
        )
        bridge_summary[int(idx)] = {
            "station_x": float(station_x[idx]),
            "station_time": float(stime),
            "rna_centroid": [float(p_rna[0]), float(p_rna[1])],
            "atac_centroid": [float(p_atac[0]), float(p_atac[1])],
        }

    ax.text(-0.22, overview_rna_card.y + overview_rna_card.h + 0.58, "Synchronized RNA-ATAC Temporal Trajectories", fontsize=23, fontfamily="STIXGeneral", fontweight="bold", color=TITLE_COLOR, ha="left", va="bottom")
    ax.text(-0.22, overview_rna_card.y + overview_rna_card.h + 0.33, "State map flowing into all-time aligned local states", fontsize=12.3, color=SUBTITLE_COLOR, ha="left", va="bottom")
    ax.text(-0.34, float(_ribbon_center_y(rna_ribbon, rna_ribbon.x0)), "RNA space", fontsize=13.9, color=TEXT_COLOR, ha="right", va="center")
    ax.text(-0.34, float(_ribbon_center_y(atac_ribbon, atac_ribbon.x0)), "ATAC space", fontsize=13.9, color=TEXT_COLOR, ha="right", va="center")

    _add_time_gradient_legend(
        fig,
        start_label=str(station_labels[int(real_station_idx[0])] if len(real_station_idx) > 0 else "start"),
        end_label=str(station_labels[int(real_station_idx[-1])] if len(real_station_idx) > 0 else "end"),
    )
    style_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor=TIME_GRADIENT_HEX[0], markeredgewidth=1.1, markersize=7, label="start"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=TIME_GRADIENT_HEX[9], markeredgecolor="#1C2024", markeredgewidth=0.5, markersize=7.4, label="end"),
        Line2D([0], [0], color=SYNC_BRIDGE_COLOR, lw=2.3, label="sync bridge"),
    ]
    legend = fig.legend(
        handles=style_handles,
        loc="lower center",
        bbox_to_anchor=(0.69, 0.07),
        ncol=3,
        frameon=False,
        fontsize=9.4,
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.add_artist(legend)

    all_polys = np.vstack(
        [
            _card_polygon(overview_rna_card),
            _card_polygon(overview_atac_card),
            _ribbon_polygon(rna_ribbon),
            _ribbon_polygon(atac_ribbon),
        ]
    )
    x_min = float(np.min(all_polys[:, 0]) - 0.38)
    x_max = float(np.max(all_polys[:, 0]) + 0.58)
    y_min = float(np.min(all_polys[:, 1]) - 0.34)
    y_max = float(np.max(all_polys[:, 1]) + 0.85)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    png_preview = _figure_png_bytes(fig, dpi=300)
    html_title = "Synchronized RNA-ATAC Temporal Trajectories"
    html_subtitle = "State maps flow into all-time aligned RNA and ATAC ribbon corridors."
    html_path = _write_html_preview(output_html, png_preview, html_title, html_subtitle)

    saved = {"png": None, "svg": None, "pdf": None, "error": None}
    if export_static:
        try:
            output_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_png, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
            fig.savefig(output_svg, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
            fig.savefig(output_pdf, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
            saved["png"] = str(output_png)
            saved["svg"] = str(output_svg)
            saved["pdf"] = str(output_pdf)
        except Exception as exc:  # pragma: no cover
            saved["error"] = f"{type(exc).__name__}: {exc}"
    plt.close(fig)

    return {
        "html": html_path,
        "png": saved["png"],
        "svg": saved["svg"],
        "pdf": saved["pdf"],
        "static_export_error": saved["error"],
        "renderer": "all_time_ribbon_corridor",
        "overview_density_layers": overview_density_layers,
        "overview_points_drawn": overview_point_counts,
        "overview_streams": overview_stream_counts,
        "station_fog_layers_rna": {},
        "station_fog_layers_atac": {},
        "station_cross_points_rna": {str(k): int(v) for k, v in station_cross_points_rna.items()},
        "station_cross_points_atac": {str(k): int(v) for k, v in station_cross_points_atac.items()},
        "bridge_stations": [int(x) for x in bridge_summary.keys()],
        "bridge_summary": bridge_summary,
        "global_connectors": connector_summary,
        "overview_card": {
            "rna": {"x": overview_rna_card.x, "y": overview_rna_card.y, "w": overview_rna_card.w, "h": overview_rna_card.h},
            "atac": {"x": overview_atac_card.x, "y": overview_atac_card.y, "w": overview_atac_card.w, "h": overview_atac_card.h},
        },
        "ribbon_layout": {
            "rna": {"x0": rna_ribbon.x0, "y0": rna_ribbon.y0, "width": rna_ribbon.width, "height": rna_ribbon.height, "slope": rna_ribbon.slope},
            "atac": {"x0": atac_ribbon.x0, "y0": atac_ribbon.y0, "width": atac_ribbon.width, "height": atac_ribbon.height, "slope": atac_ribbon.slope},
            "station_x": [float(x) for x in station_x.tolist()],
            "station_times": [float(x) for x in np.asarray(station_times, dtype=float).tolist()],
            "station_is_real": [bool(x) for x in np.asarray(station_is_real, dtype=bool).tolist()],
            "station_labels": [str(x) for x in station_labels],
        },
        "style": {
            "rna_card_face": RNA_CARD_FACE,
            "rna_card_edge": RNA_CARD_EDGE,
            "atac_card_face": ATAC_CARD_FACE,
            "atac_card_edge": ATAC_CARD_EDGE,
            "sync_bridge_color": SYNC_BRIDGE_COLOR,
            "trajectory_color_mode": "time_gradient",
            "global_overview_mode": "real_data_only",
            "global_transition_mode": "mist_ribbon_bundle",
            "global_velocity_overlay": "stream",
        },
    }

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render synchronized multimodal trajectories as a generic all-time ribbon corridor."
)

    parser.add_argument("--dynamic-adata", type=str, default="paper_data/dynamics/hspc_31800/adata.h5ad", help="Path to dynamics adata (.h5ad).")
    parser.add_argument("--space1-processed", type=str, default="paper_data/processed/hspc_31800/rna.h5ad", help="Path to processed adata for space-1.")
    parser.add_argument("--space2-processed", type=str, default="paper_data/processed/hspc_31800/protein.h5ad", help="Path to processed adata for space-2.")
    parser.add_argument("--map-model", type=str, default="paper_data/maps/hspc_31800/best_model.pt", help="Path to map model checkpoint.")
    parser.add_argument(
        "--direction",
        type=str,
        default="2to1",
        choices=["1to2", "2to1"],
        help="Simulation/mapping direction.",
    )
    parser.add_argument("--time-key", type=str, default="time_point_processed", help="Obs column used as real-time key.")
    parser.add_argument("--dt", type=float, default=0.05, help="Resimulation dt.")
    parser.add_argument("--t-start", type=float, default=None, help="Optional simulation start time; default=min(real_times).")
    parser.add_argument("--t-end", type=float, default=None, help="Optional simulation end time; default=max(real_times).")
    parser.add_argument("--n-traj", type=int, default=3, help="Number of trajectories to render (default=3).")
    parser.add_argument("--source-index-file", type=str, default="", help="Optional file containing source cell indices.")
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=40,
        help="UMAP n_neighbors for the second-space embedding and auxiliary panel.",
    )
    parser.add_argument(
        "--space1-umap-model",
        type=str,
        default="",
        help="Optional prefit UMAP model for space-1. When set, it replaces corridor auto UMAP for space-1.",
    )
    parser.add_argument(
        "--space2-umap-model",
        type=str,
        default="",
        help="Optional prefit UMAP model for space-2. When set, it replaces corridor auto UMAP for space-2.",
    )
    parser.add_argument(
        "--midpoints-per-gap",
        type=int,
        default=30,
        help="Midpoint trajectory samples inserted between adjacent station windows.",
    )
    parser.add_argument(
        "--time-x-step",
        type=float,
        default=0.67,
        help="Base horizontal station step in corridor world units.",
    )
    parser.add_argument(
        "--time-rise-step",
        type=float,
        default=0.052,
        help="Vertical rise across the ribbon corridor.",
    )
    parser.add_argument(
        "--space-gap-factor",
        type=float,
        default=1.35,
        help="Relative upper-vs-lower row separation factor.",
    )
    parser.add_argument(
        "--bridge-stations",
        type=str,
        default="0,-1",
        help="Comma-separated station indices to connect with sync bridge arcs.",
    )
    parser.add_argument(
        "--mapper-type",
        type=str,
        default="ae",
        help="Transport mapper type passed to build_transport_map.",
    )
    parser.add_argument(
        "--mapper-kwargs-json",
        type=str,
        default="{}",
        help="JSON dict for mapper kwargs.",
    )
    parser.add_argument(
        "--overview-max-points",
        type=int,
        default=1500,
        help="Maximum sparse points drawn on each global overview card.",
    )
    parser.add_argument(
        "--time-card-max-points",
        type=int,
        default=520,
        help="Maximum real-data background points drawn per real station.",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu/cuda/auto")
    parser.add_argument("--output-dir", type=str, default="results/corridor_hspc_2to1", help="Output directory.")
    parser.add_argument(
        "--no-export-static",
        action="store_true",
        help="Skip PNG/SVG/PDF export and only write the embedded HTML preview.",
    )
    return parser.parse_args(argv)


def render(**overrides: Any) -> Path:
    """Render the corridor from Python without going through a CLI process."""

    args = _parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"Unknown corridor option: {key}")
        setattr(args, key, value)
    main(args)
    return Path(args.output_dir).resolve()


def main(args: Optional[argparse.Namespace] = None) -> None:
    seed_select=25
    args = _parse_args() if args is None else args
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dyn_adata = ad.read_h5ad(str(Path(args.dynamic_adata).resolve()))
    space1_proc = ad.read_h5ad(str(Path(args.space1_processed).resolve()))
    space2_proc = ad.read_h5ad(str(Path(args.space2_processed).resolve()))
    if "X_latent" not in space1_proc.obsm or "X_latent" not in space2_proc.obsm:
        raise ValueError("Both processed adata files must contain obsm['X_latent'].")

    time_key = str(args.time_key).strip()
    if time_key not in dyn_adata.obs.columns:
        raise ValueError(f"time_key='{time_key}' not found in dynamic_adata.obs")

    dynamic_times = np.asarray(dyn_adata.obs[time_key], dtype=float)
    real_times = np.asarray(sorted({float(x) for x in dynamic_times.tolist()}), dtype=float)
    if len(real_times) < 2:
        raise ValueError("At least two real time points are required.")

    _set_time_color_range(float(real_times[0]), float(real_times[-1]))
    station_times, station_is_real, extras_per_gap = _build_station_template(real_times)
    station_labels: List[str] = []
    real_counter = 0
    for is_real in station_is_real.tolist():
        if bool(is_real):
            station_labels.append(f"D{int(real_counter)}")
            real_counter += 1
        else:
            station_labels.append("")

    t_start = float(args.t_start) if args.t_start is not None else float(real_times[0])
    t_end = float(args.t_end) if args.t_end is not None else float(real_times[-1])
    if float(station_times[0]) < t_start - 1e-8 or float(station_times[-1]) > t_end + 1e-8:
        raise ValueError("Simulation range [t_start,t_end] must cover all station times.")

    device = _resolve_device(str(args.device), "cpu")
    model = load_model_from_adata(dyn_adata)
    hidden_dim = int(
        dyn_adata.uns.get("all_model", {})
        .get("model_config", {})
        .get("multi", {})
        .get("hidden_dim", 64)
    )

    mapper_kwargs_raw = json.loads(str(args.mapper_kwargs_json))
    if not isinstance(mapper_kwargs_raw, dict):
        raise ValueError("--mapper-kwargs-json must decode to a JSON object.")

    if str(args.direction) == "1to2":
        map_mode = (1, 2)
        source_latent = np.asarray(space1_proc.obsm["X_latent"], dtype=np.float32)
        source_times = (
            np.asarray(space1_proc.obs[time_key], dtype=float)
            if time_key in space1_proc.obs.columns
            else np.zeros(space1_proc.n_obs, dtype=float)
        )
    else:
        map_mode = (2, 1)
        source_latent = np.asarray(space2_proc.obsm["X_latent"], dtype=np.float32)
        source_times = (
            np.asarray(space2_proc.obs[time_key], dtype=float)
            if time_key in space2_proc.obs.columns
            else np.zeros(space2_proc.n_obs, dtype=float)
        )

    mapper = build_transport_map(
        map_model_path=str(Path(args.map_model).resolve()),
        input_dim1=int(space1_proc.obsm["X_latent"].shape[1]),
        input_dim2=int(space2_proc.obsm["X_latent"].shape[1]),
        mode=map_mode,
        hidden_dim=hidden_dim,
        device=device,
        mapper_type=str(args.mapper_type),
        mapper_kwargs=dict(mapper_kwargs_raw),
    )

    source_idx = _select_source_indices(
        source_latent,
        source_times,
        n_traj=int(args.n_traj),
        seed=seed_select,
        source_index_file=str(args.source_index_file),
    )
    traj_ids = [int(x) for x in source_idx.tolist()]
    color_by_traj = {int(tid): TRAJECTORY_PALETTE[i % len(TRAJECTORY_PALETTE)] for i, tid in enumerate(traj_ids)}

    x0 = np.asarray(source_latent[source_idx], dtype=np.float32)
    sim_time = _build_time_grid(t_start, t_end, float(args.dt))
    traj_src = _build_ode_trajectory(
        model=model,
        x0=x0,
        times=sim_time,
        adata=dyn_adata,
        dt=float(args.dt),
        device=device,
    )
    traj_mapped = _map_to_secondary(mapper, traj_src["main_latent"])["sub_latent"]

    space1_obs_times = (
        np.asarray(space1_proc.obs[time_key], dtype=float)
        if time_key in space1_proc.obs.columns
        else np.zeros(space1_proc.n_obs, dtype=float)
    )
    space2_obs_times = (
        np.asarray(space2_proc.obs[time_key], dtype=float)
        if time_key in space2_proc.obs.columns
        else np.zeros(space2_proc.n_obs, dtype=float)
    )

    if str(args.space1_umap_model).strip():
        space1_reducer, space1_bg_2d = _load_umap_transform(
            str(args.space1_umap_model),
            np.asarray(space1_proc.obsm["X_latent"], dtype=np.float32),
        )
        rna_umap_summary = {
            "selection_mode": "prefit_model",
            "model_path": str(Path(args.space1_umap_model).resolve()),
            "selected_n_neighbors": int(getattr(space1_reducer, "n_neighbors", 0)),
            "selected_min_dist": float(getattr(space1_reducer, "min_dist", 0.0)),
        }
    else:
        space1_reducer, space1_bg_2d, rna_umap_summary = _fit_umap_auto_by_time_geometry(
            np.asarray(space1_proc.obsm["X_latent"], dtype=np.float32),
            obs_times=space1_obs_times,
            seed=seed_select,
            n_neighbors_candidates=(100, 150, 200),
            min_dist_candidates=(0.70, 0.90),
            max_points_per_time=150,
        )
    if str(args.space2_umap_model).strip():
        space2_reducer, space2_bg_2d = _load_umap_transform(
            str(args.space2_umap_model),
            np.asarray(space2_proc.obsm["X_latent"], dtype=np.float32),
        )
    else:
        space2_reducer, space2_bg_2d = _fit_umap(
            np.asarray(space2_proc.obsm["X_latent"], dtype=np.float32),
            seed=seed_select,
            n_neighbors=int(args.umap_neighbors),
            min_dist=0.25,
        )

    if str(args.direction) == "1to2":
        traj_space1_2d = _transform_series_2d(np.asarray(traj_src["main_latent"], dtype=np.float32), space1_reducer)
        traj_space2_2d = _transform_series_2d(np.asarray(traj_mapped, dtype=np.float32), space2_reducer)
    else:
        traj_space2_2d = _transform_series_2d(np.asarray(traj_src["main_latent"], dtype=np.float32), space2_reducer)
        traj_space1_2d = _transform_series_2d(np.asarray(traj_mapped, dtype=np.float32), space1_reducer)

    display_times, display_is_anchor = _build_display_times(station_times, int(args.midpoints_per_gap))
    display_time_idx = _map_times_to_grid_indices(sim_time, display_times)
    station_time_idx = _map_times_to_grid_indices(sim_time, station_times)

    traj_rna_display = np.asarray(traj_space1_2d[display_time_idx], dtype=float)
    traj_atac_display = np.asarray(traj_space2_2d[display_time_idx], dtype=float)
    traj_rna_anchor = np.asarray(traj_space1_2d[station_time_idx], dtype=float)
    traj_atac_anchor = np.asarray(traj_space2_2d[station_time_idx], dtype=float)

    rna_bg_by_time: Dict[float, np.ndarray] = {}
    atac_bg_by_time: Dict[float, np.ndarray] = {}
    overlay_times_main: Dict[float, float] = {}
    overlay_times_atac: Dict[float, float] = {}
    for t in real_times.tolist():
        mask_main, time_main = _select_obs_time_mask(space1_obs_times, target_time=float(t), atol=1e-6)
        mask_atac, time_atac = _select_obs_time_mask(space2_obs_times, target_time=float(t), atol=1e-6)
        rna_bg_by_time[float(t)] = np.asarray(space1_bg_2d[mask_main], dtype=float)
        atac_bg_by_time[float(t)] = np.asarray(space2_bg_2d[mask_atac], dtype=float)
        overlay_times_main[float(t)] = float(time_main)
        overlay_times_atac[float(t)] = float(time_atac)

    overview_rna_card, overview_atac_card, rna_ribbon, atac_ribbon, station_x = _build_ribbon_layout(
        station_times,
        time_x_step=float(args.time_x_step),
        time_rise_step=float(args.time_rise_step),
        space_gap_factor=float(args.space_gap_factor),
    )
    bridge_stations = _parse_bridge_stations(str(args.bridge_stations), int(len(station_times)))

    out_html = output_dir / "sync_multimodal_time_corridor.html"
    out_corridor_png = output_dir / "sync_multimodal_time_corridor.png"
    out_corridor_svg = output_dir / "sync_multimodal_time_corridor.svg"
    out_corridor_pdf = output_dir / "sync_multimodal_time_corridor.pdf"
    out_aux_png = output_dir / "sync_multimodal_time_reference_2d_panel.png"

    render_main = _render_global_corridor_figure(
        rna_bg_all=np.asarray(space1_bg_2d, dtype=float),
        atac_bg_all=np.asarray(space2_bg_2d, dtype=float),
        rna_bg_by_time=rna_bg_by_time,
        atac_bg_by_time=atac_bg_by_time,
        rna_times_all=space1_obs_times,
        atac_times_all=space2_obs_times,
        traj_rna_display=traj_rna_display,
        traj_atac_display=traj_atac_display,
        traj_rna_anchor=traj_rna_anchor,
        traj_atac_anchor=traj_atac_anchor,
        display_times=display_times,
        display_is_anchor=display_is_anchor,
        station_times=station_times,
        station_is_real=station_is_real,
        station_labels=station_labels,
        traj_ids=traj_ids,
        color_by_traj=color_by_traj,
        bridge_station_indices=bridge_stations,
        overview_rna_card=overview_rna_card,
        overview_atac_card=overview_atac_card,
        rna_ribbon=rna_ribbon,
        atac_ribbon=atac_ribbon,
        station_x=station_x,
        output_html=out_html,
        output_png=out_corridor_png,
        output_svg=out_corridor_svg,
        output_pdf=out_corridor_pdf,
        export_static=(not bool(args.no_export_static)),
        overview_max_points=int(args.overview_max_points),
        time_card_max_points=int(args.time_card_max_points),
        seed=seed_select,
    )

    aux_panel = None
    if not bool(args.no_export_static):
        aux_panel = _render_reference_2d_panel(
            rna_bg=np.asarray(space1_bg_2d, dtype=float),
            atac_bg=np.asarray(space2_bg_2d, dtype=float),
            time_key_values=space1_obs_times,
            time_palette=_make_time_palette(np.unique(space1_obs_times)),
            traj_rna_2d=np.asarray(traj_space1_2d, dtype=float),
            traj_atac_2d=np.asarray(traj_space2_2d, dtype=float),
            traj_ids=traj_ids,
            color_by_traj=color_by_traj,
            output_png=out_aux_png,
        )

    run_summary = {
        "output_dir": str(output_dir),
        "input_dynamic_adata": str(Path(args.dynamic_adata).resolve()),
        "input_space1_processed": str(Path(args.space1_processed).resolve()),
        "input_space2_processed": str(Path(args.space2_processed).resolve()),
        "input_map_model": str(Path(args.map_model).resolve()),
        "direction": str(args.direction),
        "device_used": device,
        "renderer": str(render_main.get("renderer", "all_time_ribbon_corridor")),
        "design_name": "all_time_ribbon_corridor",
        "trajectory_color_mode": "time_gradient",
        "trajectory_identity_mode": "metadata_only",
        "global_overview_mode": "real_data_only",
        "global_time_color_source": "TIME_GRADIENT_HEX",
        "global_transition_mode": "mist_ribbon_bundle",
        "global_transition_origin": "global_right_outer_edge_aligned_to_day0_real_data",
        "global_transition_target": "d0_left_outer_edge",
        "global_transition_occlusion_policy": "outside_containers_only",
        "seed_rule": "random_3_default_and_manual_index_for_4",
        "t_start": float(t_start),
        "t_end": float(t_end),
        "dt": float(args.dt),
        "umap_neighbors": int(args.umap_neighbors),
        "space1_umap_model": str(Path(args.space1_umap_model).resolve()) if str(args.space1_umap_model).strip() else "",
        "space2_umap_model": str(Path(args.space2_umap_model).resolve()) if str(args.space2_umap_model).strip() else "",
        "rna_umap_selection_mode": str(rna_umap_summary.get("selection_mode", "auto_grid_search")),
        "rna_umap_selected_neighbors": int(rna_umap_summary.get("selected_n_neighbors", 0)),
        "rna_umap_selected_min_dist": float(rna_umap_summary.get("selected_min_dist", 0.0)),
        "rna_umap_best_max_step_over_median": float(rna_umap_summary.get("best_max_step_over_median", 0.0)),
        "rna_umap_best_max_centroid_dist": float(rna_umap_summary.get("best_max_centroid_dist", 0.0)),
        "rna_umap_candidate_scores": rna_umap_summary.get("candidate_scores", []),
        "time_x_step": float(args.time_x_step),
        "time_rise_step": float(args.time_rise_step),
        "space_gap_factor": float(args.space_gap_factor),
        "midpoints_per_gap": int(args.midpoints_per_gap),
        "bridge_stations_requested": [int(x) for x in bridge_stations],
        "bridge_stations_drawn": [int(x) for x in render_main.get("bridge_stations", [])],
        "overview_max_points": int(args.overview_max_points),
        "time_card_max_points": int(args.time_card_max_points),
        "real_times": [float(x) for x in real_times.tolist()],
        "station_times": [float(x) for x in station_times.tolist()],
        "station_is_real": [bool(x) for x in station_is_real.tolist()],
        "station_labels": [str(x) for x in station_labels],
        "extras_per_gap": [int(x) for x in extras_per_gap.tolist()],
        "time_points_displayed": [float(x) for x in station_times.tolist()],
        "time_grid": [float(x) for x in sim_time.tolist()],
        "station_time_indices_in_dt_grid": [int(x) for x in station_time_idx.tolist()],
        "display_times_for_corridor": [float(x) for x in display_times.tolist()],
        "display_time_indices_in_dt_grid": [int(x) for x in display_time_idx.tolist()],
        "display_is_anchor": [bool(x) for x in display_is_anchor.tolist()],
        "overlay_time_used_space1_by_time": {str(k): float(v) for k, v in overlay_times_main.items()},
        "overlay_time_used_space2_by_time": {str(k): float(v) for k, v in overlay_times_atac.items()},
        "time_key": time_key,
        "overview_density_layers": render_main.get("overview_density_layers", {}),
        "overview_points_drawn": render_main.get("overview_points_drawn", {}),
        "overview_streams": render_main.get("overview_streams", {}),
        "station_fog_layers_rna": render_main.get("station_fog_layers_rna", {}),
        "station_fog_layers_atac": render_main.get("station_fog_layers_atac", {}),
        "station_cross_points_rna": render_main.get("station_cross_points_rna", {}),
        "station_cross_points_atac": render_main.get("station_cross_points_atac", {}),
        "bridge_summary": render_main.get("bridge_summary", {}),
        "global_connectors": render_main.get("global_connectors", {}),
        "card_layout": {
            "overview": render_main.get("overview_card", {}),
            "ribbons": render_main.get("ribbon_layout", {}),
        },
        "style": render_main.get("style", {}),
        "source_cell_indices": [int(x) for x in source_idx.tolist()],
        "time_gradient_palette": {str(int(k)): str(v) for k, v in TIME_GRADIENT_HEX.items()},
        "aux_panel_traj_color_map": {str(k): str(v) for k, v in color_by_traj.items()},
        "outputs": {
            "corridor_html": render_main.get("html"),
            "corridor_png": render_main.get("png"),
            "corridor_svg": render_main.get("svg"),
            "corridor_pdf": render_main.get("pdf"),
            "tower_2d_png": aux_panel,
            "static_export_error": render_main.get("static_export_error"),
        },
    }
    summary_path = output_dir / "sync_multimodal_time_tower_manifest.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(f"[sync-corridor] selected trajectories={traj_ids}")
    print(f"[sync-corridor] output dir: {output_dir}")
    print(f"[sync-corridor] corridor html: {render_main.get('html')}")
    print(f"[sync-corridor] corridor png: {render_main.get('png')}")
    print(f"[sync-corridor] summary: {summary_path}")


if __name__ == "__main__":
    main()
