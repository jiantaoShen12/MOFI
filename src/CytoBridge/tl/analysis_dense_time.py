from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy import sparse
from scipy.stats import norm, ranksums
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from CytoBridge.Map.tl.transport_factory import build_transport_map
from CytoBridge.tl.analysis import simulate_trajectory
from CytoBridge.utils import load_model_from_adata

LOGGER = logging.getLogger(__name__)


@dataclass
class PCABundle:
    reconstruction: np.ndarray  # (k, g)
    forward: np.ndarray  # (g, k), pseudo-inverse of reconstruction
    mean: np.ndarray  # (g,)
    var_names: List[str]  # feature names in the reconstruction feature order
    x_before_pca: Optional[np.ndarray]  # (n, g), optional
    original_info_key: str  # uns key used to load original-space metadata


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _resolve_runtime_device(device: str) -> str:
    requested = str(device).strip() or "cpu"
    requested_lower = requested.lower()
    if requested_lower.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("Requested device '%s' but CUDA is unavailable. Falling back to CPU.", requested)
        return "cpu"
    return requested


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, ensure_ascii=False, indent=2)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _slugify_token(text: Any, *, fallback: str = "na") -> str:
    raw = str(text).strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    raw = raw.strip("-")
    return raw if raw else fallback


def _make_perturbation_suffix(groups: Sequence[str], cell_types: Sequence[str]) -> str:
    g = [_slugify_token(x) for x in groups if str(x).strip()]
    l = [_slugify_token(x) for x in cell_types if str(x).strip()]
    g_head = ("-".join(g[:2]) if g else "all")[:24].strip("-") or "all"
    l_head = ("-".join(l[:2]) if l else "all")[:24].strip("-") or "all"
    signature_obj = {"groups": g, "cell_types": l}
    signature = json.dumps(signature_obj, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
    return f"g-{g_head}__ct-{l_head}__h-{digest}"


def _safe_float_list(values: Sequence[float]) -> List[float]:
    out = sorted({float(v) for v in values})
    if len(out) == 0:
        raise ValueError("time grid is empty")
    return out


def _normalize_str_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = [values]
    elif isinstance(values, (list, tuple, np.ndarray, pd.Index)):
        raw = list(values)
    else:
        raw = [values]
    return [str(x).strip() for x in raw if str(x).strip()]


def _normalize_float_list(values: Any) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, np.ndarray, pd.Index)):
        raw = list(values)
    else:
        raw = [values]
    out: List[float] = []
    for item in raw:
        txt = str(item).strip()
        if not txt:
            continue
        try:
            out.append(float(txt))
        except (TypeError, ValueError):
            continue
    return out


def _require_str_list(cfg: Dict[str, Any], key: str, *, context: str) -> List[str]:
    out = _normalize_str_list(cfg.get(key, []))
    if not out:
        raise ValueError(f"{context}.{key} must be a non-empty list in config")
    return out


def _normalize_time_grid_presets(raw: Any) -> Dict[str, List[float]]:
    if not isinstance(raw, dict) or len(raw) == 0:
        raise ValueError("inference.time_grid_presets must be a non-empty mapping")
    out: Dict[str, List[float]] = {}
    for key, values in raw.items():
        key_norm = str(key).strip().lower()
        if not key_norm:
            continue
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError(f"inference.time_grid_presets.{key_norm} must be a list of numbers")
        out[key_norm] = _safe_float_list(values)
    if len(out) == 0:
        raise ValueError("inference.time_grid_presets must contain at least one valid mode")
    return out


def resolve_time_grid(
    observed_times: Sequence[float],
    mode: str = "coarse_dense",
    values: Optional[Sequence[float]] = None,
    time_grid_presets: Optional[Dict[str, List[float]]] = None,
) -> List[float]:
    # observed_times is kept for compatibility with older call signatures.
    _ = observed_times
    mode_norm = (mode or "coarse_dense").strip().lower()

    if not time_grid_presets:
        raise ValueError("inference.time_grid_presets is required")
    if mode_norm not in time_grid_presets:
        raise ValueError(
            f"Unsupported time_grid_mode={mode}. "
            f"Choose from {list(time_grid_presets.keys())}."
        )
    raw = _safe_float_list(values) if values else _safe_float_list(time_grid_presets[mode_norm])
    return _safe_float_list(raw)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return _pearson(xr, yr)


def _interp_at(times: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    return np.interp(t_query, times, values)


def _weighted_row_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"values must be 2D, got shape={arr.shape}")
    if weights is None:
        return arr.mean(axis=0)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != arr.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != row count {arr.shape[0]}")
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    if float(w.sum()) <= 1e-12:
        return arr.mean(axis=0)
    return np.asarray(np.average(arr, axis=0, weights=w), dtype=float)


def _weighted_quantile(values: np.ndarray, quantiles: Sequence[float], weights: Optional[np.ndarray]) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    q = np.asarray(list(quantiles), dtype=float)
    if x.size == 0 or q.size == 0:
        return np.zeros(q.size, dtype=float)
    q = np.clip(q, 0.0, 1.0)
    if weights is None:
        return np.asarray(np.quantile(x, q), dtype=float)

    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != x.size:
        raise ValueError(f"weights length {w.size} != values length {x.size}")

    finite = np.isfinite(x) & np.isfinite(w)
    if not np.any(finite):
        return np.asarray(np.quantile(x, q), dtype=float)
    x = x[finite]
    w = np.clip(w[finite], 0.0, None)
    if float(w.sum()) <= 1e-12:
        return np.asarray(np.quantile(x, q), dtype=float)

    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    cdf = cdf / cdf[-1]
    return np.asarray(np.interp(q, cdf, x_sorted), dtype=float)


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    m = len(p_values)
    if m == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p_values)
    ranked = np.asarray(p_values)[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _two_sided_p_from_z(stat: float) -> float:
    z = abs(float(stat))
    # Compute in log-space first to avoid underflow for extreme z.
    logp = np.log(2.0) + float(norm.logsf(z))
    return float(max(np.exp(logp), np.finfo(np.float64).tiny))


def _unwrap_maybe_object_scalar(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape == () and arr.dtype == object:
        return arr.item()
    return value


def _to_dense_2d(value: Any, *, name: str) -> np.ndarray:
    value = _unwrap_maybe_object_scalar(value)
    if value is None:
        raise ValueError(f"{name} is missing")
    if sparse.issparse(value):
        arr = value.toarray()
    else:
        arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def _to_dense_1d(value: Any, *, name: str) -> np.ndarray:
    value = _unwrap_maybe_object_scalar(value)
    if value is None:
        raise ValueError(f"{name} is missing")
    if sparse.issparse(value):
        arr = value.toarray()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    return arr


def _pick_original_info_key(proc: AnnData, *, space_name: str) -> Tuple[str, Dict[str, Any]]:
    key = "original_gene_info"
    payload = proc.uns["original_gene_info"]
    return key, payload


def _load_pca_bundle(proc: AnnData, *, space_name: str) -> PCABundle:
    pca_uns = proc.uns.get("pca", {})
    ogi_key, ogi = _pick_original_info_key(proc, space_name=space_name)
    if not isinstance(pca_uns, dict):
        raise ValueError(f"{space_name}: uns['pca'] not found or invalid")

    var_names_raw = ogi.get("var_names", None)
    if var_names_raw is None:
        raise ValueError(f"{space_name}: uns['{ogi_key}']['var_names'] is missing")
    var_names = [str(x) for x in list(var_names_raw)]

    x_before_pca_raw = _unwrap_maybe_object_scalar(ogi.get("X_before_pca", None))
    x_before_pca = None
    if x_before_pca_raw is not None:
        x_before_pca = _to_dense_2d(x_before_pca_raw, name=f"{space_name}.uns['{ogi_key}']['X_before_pca']")
    
    mean_raw = _unwrap_maybe_object_scalar(pca_uns.get("input_mean", None))
    if mean_raw is not None:
        mean = _to_dense_1d(mean_raw, name=f"{space_name}.uns['pca']['input_mean']")
    elif x_before_pca is not None:
        mean = np.asarray(x_before_pca.mean(axis=0), dtype=np.float32).reshape(-1)
        LOGGER.warning("%s: uns['pca']['input_mean'] missing, inferred from X_before_pca", space_name)
    else:
        raise ValueError(
            f"{space_name}: uns['pca']['input_mean'] missing and cannot infer without X_before_pca"
        )

    projection_raw = _unwrap_maybe_object_scalar(pca_uns.get("projection_matrix", None))
    reconstruction_raw = _unwrap_maybe_object_scalar(pca_uns.get("reconstruction_matrix", None))
    projection = None
    if projection_raw is not None:
        projection = _to_dense_2d(
            projection_raw,
            name=f"{space_name}.uns['pca']['projection_matrix']",
        )

    if reconstruction_raw is not None:
        reconstruction = _to_dense_2d(
            reconstruction_raw,
            name=f"{space_name}.uns['pca']['reconstruction_matrix']",
        )
    elif projection is not None:
        reconstruction = np.asarray(np.linalg.pinv(projection), dtype=np.float32)
        LOGGER.warning("%s: reconstruction_matrix missing, computed as pinv(projection_matrix)", space_name)
    elif "PCs" in proc.varm:
        pcs = _to_dense_2d(proc.varm["PCs"], name=f"{space_name}.varm['PCs']")
        reconstruction = np.asarray(pcs.T, dtype=np.float32)
        LOGGER.warning("%s: using varm['PCs'] to build reconstruction_matrix", space_name)
    else:
        latent_ref = None
        if "X_pca" in proc.obsm:
            latent_ref = np.asarray(proc.obsm["X_pca"], dtype=np.float32)
        elif "X_latent" in proc.obsm:
            latent_ref = np.asarray(proc.obsm["X_latent"], dtype=np.float32)
        if x_before_pca is None or latent_ref is None:
            raise ValueError(
                f"{space_name}: reconstruction_matrix missing and cannot be inferred "
                "(need X_before_pca and X_pca/X_latent)"
            )
        if x_before_pca.shape[0] != latent_ref.shape[0]:
            raise ValueError(
                f"{space_name}: cannot infer reconstruction_matrix, X_before_pca n={x_before_pca.shape[0]} "
                f"!= latent n={latent_ref.shape[0]}"
            )
        X_centered = np.asarray(x_before_pca - mean[None, :], dtype=np.float32)
        reconstruction = np.asarray(np.linalg.pinv(latent_ref) @ X_centered, dtype=np.float32)
        LOGGER.warning(
            "%s: inferred reconstruction_matrix from X_before_pca and X_pca/X_latent via least squares",
            space_name,
        )
    if projection is not None:
        forward = np.asarray(projection, dtype=np.float32)
    else:
        forward = np.linalg.pinv(reconstruction).astype(np.float32)
    bundle = PCABundle(
        reconstruction=np.asarray(reconstruction, dtype=np.float32),
        forward=np.asarray(forward, dtype=np.float32),
        mean=np.asarray(mean, dtype=np.float32),
        var_names=var_names,
        x_before_pca=x_before_pca,
        original_info_key=ogi_key,
    )
    _validate_pca_bundle(bundle, space_name=space_name)
    return bundle


def _validate_pca_bundle(bundle: PCABundle, *, space_name: str, expected_latent_dim: Optional[int] = None) -> None:
    k, g = bundle.reconstruction.shape
    if bundle.forward.shape != (g, k):
        raise ValueError(f"{space_name}: forward shape {bundle.forward.shape} != ({g}, {k})")
    if bundle.mean.shape[0] != g:
        raise ValueError(f"{space_name}: input_mean length {bundle.mean.shape[0]} != feature dim {g}")
    if len(bundle.var_names) != g:
        raise ValueError(f"{space_name}: len(var_names)={len(bundle.var_names)} != feature dim {g}")
    if expected_latent_dim is not None and k != int(expected_latent_dim):
        raise ValueError(f"{space_name}: latent dim mismatch, reconstruction has {k}, expected {expected_latent_dim}")
    if bundle.x_before_pca is not None and bundle.x_before_pca.shape[1] != g:
        raise ValueError(
            f"{space_name}: X_before_pca feature dim {bundle.x_before_pca.shape[1]} != reconstruction feature dim {g}"
        )


def _project_feature_to_latent(X_feat: np.ndarray, bundle: PCABundle) -> np.ndarray:
    X_feat = np.asarray(X_feat, dtype=np.float32)
    if X_feat.ndim != 2:
        raise ValueError(f"X_feat must be 2D, got shape={X_feat.shape}")
    return np.asarray((X_feat - bundle.mean[None, :]) @ bundle.forward, dtype=np.float32)


def _reconstruct_latent_to_feature(Z: np.ndarray, bundle: PCABundle) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape={Z.shape}")
    return np.asarray(Z @ bundle.reconstruction + bundle.mean[None, :], dtype=np.float32)


def _reconstruct_latent_series(latent_series: np.ndarray, bundle: PCABundle) -> np.ndarray:
    latent_series = np.asarray(latent_series, dtype=np.float32)
    if latent_series.ndim != 3:
        raise ValueError(f"latent_series must be 3D, got shape={latent_series.shape}")
    out = [_reconstruct_latent_to_feature(latent_series[i], bundle) for i in range(latent_series.shape[0])]
    return np.stack(out, axis=0)


def _feature_matrix_to_marker_profile(
    feature_series: np.ndarray,
    feature_names: Sequence[str],
    markers: Sequence[str],
    time_grid: Sequence[float],
    *,
    label: str,
    weights_by_time: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if len(markers) == 0:
        return pd.DataFrame(columns=["time", "marker", "mean_expr", "expr_p40", "expr_p40", "expr_p60", "n_cells", "space", "source"])
    lut = {str(n): i for i, n in enumerate(feature_names)}
    marker_idx = [lut[m] for m in markers if m in lut]
    marker_used = [m for m in markers if m in lut]
    if len(marker_idx) == 0:
        return pd.DataFrame(columns=["time", "marker", "mean_expr", "expr_p40", "expr_p40", "expr_p60", "n_cells", "space", "source"])

    weights_arr = None
    if weights_by_time is not None:
        weights_arr = np.asarray(weights_by_time, dtype=float)
        if weights_arr.shape != np.asarray(feature_series).shape[:2]:
            raise ValueError(
                f"weights_by_time shape {weights_arr.shape} mismatches feature_series shape {np.asarray(feature_series).shape}"
            )

    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(time_grid):
        vals = np.asarray(feature_series[i][:, marker_idx], dtype=np.float32)
        wt = None if weights_arr is None else weights_arr[i]
        mean_expr = _weighted_row_mean(vals, wt)
        for j, (marker, val) in enumerate(zip(marker_used, mean_expr)):
            q40, q50, q60 = _weighted_quantile(vals[:, j], [0.4, 0.5, 0.6], wt)
            rows.append(
                {
                    "time": float(t),
                    "marker": str(marker),
                    "mean_expr": float(val),
                    "expr_p40": float(q40),
                    "expr_p50": float(q50),
                    "expr_p60": float(q60),
                    "n_cells": int(vals.shape[0]),
                    "space": label,
                    "source": "pred",
                }
            )
    return pd.DataFrame(rows)


def _extract_columns_from_feature_matrix(
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    names: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    wanted = [str(x) for x in names]
    if len(wanted) == 0:
        return pd.DataFrame(), [], []
    lut = {str(n): i for i, n in enumerate(feature_names)}
    found = [m for m in wanted if m in lut]
    missing = [m for m in wanted if m not in lut]
    if len(found) == 0:
        return pd.DataFrame(), [], missing
    idx = [lut[m] for m in found]
    mat = np.asarray(feature_matrix[:, idx], dtype=np.float32)
    return pd.DataFrame(mat, columns=found), found, missing


def _batched_velocity_jacobian(
    model: torch.nn.Module,
    x_np: np.ndarray,
    t_val: float,
    *,
    device: str,
) -> np.ndarray:
    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"x_np must be 2D, got shape={x_np.shape}")
    n, d = x_np.shape
    out = np.zeros((n, d, d), dtype=np.float32)

    device_t = torch.device(device)
    model = model.to(device_t).eval()
    has_velocity = hasattr(model, "velocity_net")
    if not has_velocity:
        return out

    with torch.enable_grad():
        for i in range(n):
            xi = torch.tensor(x_np[i : i + 1], dtype=torch.float32, device=device_t, requires_grad=True)
            ti = torch.full((1, 1), float(t_val), dtype=torch.float32, device=device_t)
            vel = model.velocity_net(torch.cat([xi, ti], dim=1))
            rows = []
            for out_idx in range(d):
                grad = torch.autograd.grad(
                    vel[0, out_idx],
                    xi,
                    retain_graph=(out_idx < d - 1),
                    create_graph=False,
                    allow_unused=False,
                )[0][0]
                rows.append(grad.detach().cpu().numpy())
            out[i] = np.asarray(rows, dtype=np.float32)
    return out


def _batched_velocity(
    model: torch.nn.Module,
    x_np: np.ndarray,
    t_val: float,
    *,
    device: str,
) -> np.ndarray:
    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"x_np must be 2D, got shape={x_np.shape}")
    device_t = torch.device(device)
    model = model.to(device_t).eval()
    if not hasattr(model, "velocity_net"):
        return np.zeros_like(x_np, dtype=np.float32)
    with torch.no_grad():
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device_t)
        t_col = torch.full((x_t.shape[0], 1), float(t_val), dtype=torch.float32, device=device_t)
        vel = model.velocity_net(torch.cat([x_t, t_col], dim=1))
        return np.asarray(vel.detach().cpu().numpy(), dtype=np.float32)


def _select_initial_indices(
    adata: AnnData,
    *,
    time_key: str,
    init_time: Optional[float],
    sampling_mode: str,
    seed: int,
    cell_type_key: str = "cell_type",
    sampling_cell_type: Optional[str] = None,
    sampling_indices: Optional[Sequence[int]] = None,
    strict_index_init_time: bool = True,
    max_cells: Optional[int] = None,
) -> np.ndarray:
    if time_key not in adata.obs:
        raise KeyError(f"{time_key} not found in adata.obs")
    times = np.asarray(adata.obs[time_key], dtype=float)
    unique_times = np.sort(np.unique(times))
    t0 = float(unique_times[0]) if init_time is None else float(init_time)
    idx_all = np.where(np.isclose(times, t0))[0]
    if len(idx_all) == 0:
        raise ValueError(f"No cells found at init_time={t0}")

    rng = np.random.default_rng(seed)
    mode = (sampling_mode or "random").strip().lower()

    if mode in {"random", "sample"}:
        if max_cells is None:
            max_cells = min(5000, len(idx_all))
        take = min(len(idx_all), int(max_cells))
        selected = rng.choice(idx_all, size=take, replace=False)
    elif mode in {"cell_type_random", "celltype_random", "cell_type"}:
        if sampling_cell_type is None or str(sampling_cell_type).strip() == "":
            raise ValueError("sampling_mode='cell_type_random' requires sampling_cell_type.")
        if cell_type_key not in adata.obs:
            raise KeyError(f"{cell_type_key} not found in adata.obs")
        labels = np.asarray(adata.obs[cell_type_key]).astype(str)
        mask = np.isclose(times, t0) & (labels == str(sampling_cell_type))
        eligible = np.where(mask)[0]
        if len(eligible) == 0:
            raise ValueError(
                f"No cells found for init_time={t0} and {cell_type_key}='{sampling_cell_type}'."
            )
        if max_cells is None:
            max_cells = min(5000, len(eligible))
        take = min(len(eligible), int(max_cells))
        selected = rng.choice(eligible, size=take, replace=False)
    elif mode in {"index", "indices", "given_index"}:
        if sampling_indices is None:
            raise ValueError("sampling_mode='index' requires sampling_indices.")
        if isinstance(sampling_indices, str):
            raw = np.asarray([x.strip() for x in sampling_indices.split(",") if x.strip()], dtype=object)
        else:
            raw = np.asarray(sampling_indices, dtype=object).reshape(-1)
        idx_list: List[int] = []
        for value in raw:
            text = str(value).strip()
            if not text:
                continue
            num = float(text)
            if not np.isfinite(num) or abs(num - round(num)) > 1e-8:
                raise ValueError(f"Invalid index value '{value}' in sampling_indices.")
            idx_list.append(int(round(num)))
        if not idx_list:
            raise ValueError("sampling_indices is empty after parsing.")
        selected = np.asarray(idx_list, dtype=int)
        if int(np.min(selected)) < 0 or int(np.max(selected)) >= int(adata.n_obs):
            raise ValueError(
                f"sampling_indices out of range: min={int(np.min(selected))}, "
                f"max={int(np.max(selected))}, n_obs={int(adata.n_obs)}"
            )
        if strict_index_init_time:
            mask_t0 = np.isclose(times, t0)
            invalid = selected[~mask_t0[selected]]
            if invalid.size > 0:
                raise ValueError(
                    "sampling_mode='index' only allows init_time rows when "
                    f"strict_index_init_time=True. Invalid indices include {invalid[:10].tolist()}."
                )
    else:
        raise ValueError(
            "Unsupported sampling_mode=%s. Choose from random/sample, cell_type_random, index." % mode
        )

    if mode in {"random", "sample", "cell_type_random", "celltype_random", "cell_type"} and max_cells is not None and len(selected) > int(max_cells):
        selected = rng.choice(selected, size=int(max_cells), replace=False)
    return np.sort(np.asarray(selected, dtype=int))


def _build_ode_trajectory(
    model: torch.nn.Module,
    x0: np.ndarray,
    times: Sequence[float],
    *,
    adata: Optional[AnnData],
    dt: float,
    device: str,
) -> Dict[str, Any]:
    device_t = torch.device(device)
    model = model.to(device_t).eval()
    x0_t = torch.tensor(x0, dtype=torch.float32, device=device_t)
    t_eval = _safe_float_list(times)
    points, weights = simulate_trajectory(
        adata=adata,
        model=model,
        x0=x0_t,
        sigma=0.0,
        time=t_eval,
        dt=float(dt),
        device=device_t,
    )
    x_traj = np.asarray(points, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim == 3 and w.shape[-1] == 1:
        w = w[..., 0]
    if w.ndim != 2:
        raise ValueError(f"Unexpected weight shape from simulate_trajectory: {w.shape}")
    n_cells = x_traj.shape[1]
    weights_norm = w
    weights_abs = weights_norm * float(n_cells)
    lnw = np.log(np.maximum(weights_norm, 1e-12))

    velocities: List[np.ndarray] = []
    growths: List[np.ndarray] = []
    with torch.no_grad():
        for i, t_val in enumerate(t_eval):
            z = torch.tensor(x_traj[i], dtype=torch.float32, device=device_t)
            t_col = torch.full((z.shape[0], 1), float(t_val), device=device_t)
            net_input = torch.cat([z, t_col], dim=1)
            vel = model.velocity_net(net_input) if hasattr(model, "velocity_net") else torch.zeros_like(z)
            gro = model.growth_net(net_input) if hasattr(model, "growth_net") else torch.zeros((z.shape[0], 1), device=device_t)
            velocities.append(vel.cpu().numpy())
            growths.append(gro.cpu().numpy())

    return {
        "time_grid": np.asarray(t_eval, dtype=float),
        "main_latent": x_traj,
        "lnw": lnw,
        "weights_norm": weights_norm,
        "weights_abs": weights_abs,
        "velocity_latent": np.stack(velocities, axis=0),
        "growth_rate": np.stack(growths, axis=0).squeeze(-1),
    }


def _map_to_secondary(mapper: Any, main_latent: np.ndarray) -> Dict[str, np.ndarray]:
    t_num = main_latent.shape[0]
    mapped = []
    for i in range(t_num):
        main_i = main_latent[i]
        sub_i = mapper.T(main_i).detach().cpu().numpy()
        mapped.append(sub_i)
    return {
        "sub_latent": np.stack(mapped, axis=0),
    }


def _build_prediction_adata(
    latent: np.ndarray,
    weights: np.ndarray,
    source_indices: np.ndarray,
    source_cell_types: Optional[np.ndarray],
    time_grid: Sequence[float],
    velocity: Optional[np.ndarray] = None,
    growth: Optional[np.ndarray] = None,
) -> AnnData:
    t_num, n_cells, dim = latent.shape
    flat_latent = latent.reshape(t_num * n_cells, dim).astype(np.float32)
    flat_weights = weights.reshape(t_num * n_cells).astype(np.float32)
    time_rep = np.repeat(np.asarray(time_grid, dtype=np.float32), n_cells)
    source_rep = np.tile(source_indices.astype(int), t_num)
    traj_idx = np.tile(np.arange(n_cells, dtype=int), t_num)

    obs = pd.DataFrame(
        {
            "pred_time": time_rep,
            "source_cell_index": source_rep,
            "trajectory_index": traj_idx,
            "weight": flat_weights,
        }
    )
    if source_cell_types is not None:
        obs["source_cell_type"] = np.tile(source_cell_types.astype(str), t_num)

    var = pd.DataFrame(index=[f"latent_{i}" for i in range(dim)])
    pred = AnnData(X=flat_latent, obs=obs, var=var)
    pred.obsm["X_latent"] = flat_latent.copy()
    if velocity is not None:
        pred.obsm["velocity_latent"] = velocity.reshape(t_num * n_cells, dim).astype(np.float32)
    if growth is not None:
        pred.obs["growth_rate"] = growth.reshape(t_num * n_cells).astype(np.float32)
    pred.uns["dense_time"] = {
        "time_grid": [float(x) for x in time_grid],
        "n_time": int(t_num),
        "n_trajectory_cells": int(n_cells),
        "latent_dim": int(dim),
    }
    return pred


def _dense_time_output_files(out_dir: Path) -> Dict[str, Path]:
    return {
        "pred_main_h5ad": out_dir / "pred_main.h5ad",
        "pred_sub_h5ad": out_dir / "pred_sub.h5ad",
        "traj_main_npy": out_dir / "traj_main.npy",
        "traj_sub_npy": out_dir / "traj_sub.npy",
        "weights_npy": out_dir / "weights.npy",
        "velocity_main_npy": out_dir / "velocity_main.npy",
        "growth_main_npy": out_dir / "growth_main.npy",
        "qc_metrics_json": out_dir / "qc_metrics.json",
        "manifest_json": out_dir / "manifest.json",
    }


def _parse_multi_mode(raw_mode: Any) -> Tuple[int, int]:
    mode_use = raw_mode
    if isinstance(mode_use, str):
        txt = str(mode_use).strip()
        try:
            mode_use = eval(txt)
        except Exception:
            mode_use = txt
    if isinstance(mode_use, (list, tuple)) and len(mode_use) == 2:
        try:
            src = int(mode_use[0])
            dst = int(mode_use[1])
            if src in {1, 2} and dst in {1, 2}:
                return src, dst
        except Exception:
            pass
    return 1, 2


def _load_cached_inference_artifacts(
    out_dir: Path,
    *,
    expected_signature: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    files = _dense_time_output_files(out_dir)
    if not out_dir.exists():
        return None

    required = [
        "pred_main_h5ad",
        "pred_sub_h5ad",
        "traj_main_npy",
        "traj_sub_npy",
        "weights_npy",
        "velocity_main_npy",
        "growth_main_npy",
        "qc_metrics_json",
    ]
    missing = [name for name in required if not files[name].exists()]
    if missing:
        LOGGER.info("Dense-time cache incomplete in %s, missing=%s. Will run inference.", out_dir, missing)
        return None

    try:
        manifest = None
        if files["manifest_json"].exists():
            with files["manifest_json"].open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                manifest = None

        if expected_signature is not None:
            if manifest is None:
                LOGGER.info("Dense-time cache manifest missing/invalid in %s. Will run inference.", out_dir)
                return None
            cache_signature = manifest.get("cache_signature", None)
            if not isinstance(cache_signature, dict):
                LOGGER.info("Dense-time cache signature missing in %s. Will run inference.", out_dir)
                return None
            lhs = json.dumps(_to_serializable(cache_signature), ensure_ascii=False, sort_keys=True)
            rhs = json.dumps(_to_serializable(expected_signature), ensure_ascii=False, sort_keys=True)
            if lhs != rhs:
                LOGGER.info("Dense-time cache signature mismatch in %s. Will run inference.", out_dir)
                return None

        traj_main = np.asarray(np.load(files["traj_main_npy"], allow_pickle=False), dtype=np.float32)
        traj_sub = np.asarray(np.load(files["traj_sub_npy"], allow_pickle=False), dtype=np.float32)
        weights = np.asarray(np.load(files["weights_npy"], allow_pickle=False), dtype=np.float32)
        velocity_main = np.asarray(np.load(files["velocity_main_npy"], allow_pickle=False), dtype=np.float32)
        growth_main = np.asarray(np.load(files["growth_main_npy"], allow_pickle=False), dtype=np.float32)

        if traj_main.ndim != 3:
            raise ValueError(f"traj_main must be 3D, got shape={traj_main.shape}")
        if traj_sub.ndim != 3:
            raise ValueError(f"traj_sub must be 3D, got shape={traj_sub.shape}")
        if weights.ndim != 2:
            raise ValueError(f"weights must be 2D, got shape={weights.shape}")
        if velocity_main.ndim != 3:
            raise ValueError(f"velocity_main must be 3D, got shape={velocity_main.shape}")
        if growth_main.ndim != 2:
            raise ValueError(f"growth_main must be 2D, got shape={growth_main.shape}")
        if traj_sub.shape[:2] != traj_main.shape[:2]:
            raise ValueError(f"traj_sub shape {traj_sub.shape} mismatches traj_main shape {traj_main.shape}")
        if weights.shape != traj_main.shape[:2]:
            raise ValueError(f"weights shape {weights.shape} mismatches traj_main shape {traj_main.shape}")
        if velocity_main.shape != traj_main.shape:
            raise ValueError(f"velocity_main shape {velocity_main.shape} mismatches traj_main shape {traj_main.shape}")
        if growth_main.shape != traj_main.shape[:2]:
            raise ValueError(f"growth_main shape {growth_main.shape} mismatches traj_main shape {traj_main.shape}")

        with files["qc_metrics_json"].open("r", encoding="utf-8") as f:
            qc_metrics = json.load(f)
        if not isinstance(qc_metrics, dict):
            qc_metrics = {}

        main_pred = ad.read_h5ad(files["pred_main_h5ad"])
        dense_meta = main_pred.uns.get("dense_time", {}) if isinstance(main_pred.uns, dict) else {}
        time_grid_raw = dense_meta.get("time_grid", None) if isinstance(dense_meta, dict) else None

        def _empty_time_grid(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, np.ndarray):
                return value.size == 0
            if isinstance(value, (list, tuple)):
                return len(value) == 0
            return False

        if _empty_time_grid(time_grid_raw):
            time_grid_raw = qc_metrics.get("time_grid", None)
        if _empty_time_grid(time_grid_raw):
            if "pred_time" in main_pred.obs:
                time_grid_raw = sorted(set(float(x) for x in np.asarray(main_pred.obs["pred_time"], dtype=float)))
            else:
                time_grid_raw = [float(x) for x in range(traj_main.shape[0])]
        time_grid = [float(x) for x in list(time_grid_raw)]
        if len(time_grid) != traj_main.shape[0]:
            LOGGER.warning(
                "Cached time_grid length %d mismatches trajectory time dim %d in %s. Using sequential fallback.",
                len(time_grid),
                traj_main.shape[0],
                out_dir,
            )
            time_grid = [float(x) for x in range(traj_main.shape[0])]

        n_cells = int(traj_main.shape[1])
        obs = main_pred.obs
        init_idx: np.ndarray
        if "source_cell_index" in obs:
            if "pred_time" in obs and len(time_grid) > 0:
                pred_time = np.asarray(obs["pred_time"], dtype=float)
                first_mask = np.isclose(pred_time, float(time_grid[0]))
                source_idx = np.asarray(obs.loc[first_mask, "source_cell_index"], dtype=int)
                if source_idx.shape[0] >= n_cells:
                    init_idx = source_idx[:n_cells]
                else:
                    fallback = np.asarray(obs["source_cell_index"], dtype=int)
                    init_idx = fallback[:n_cells] if fallback.shape[0] >= n_cells else np.arange(n_cells, dtype=int)
            else:
                fallback = np.asarray(obs["source_cell_index"], dtype=int)
                init_idx = fallback[:n_cells] if fallback.shape[0] >= n_cells else np.arange(n_cells, dtype=int)
        else:
            init_idx = np.arange(n_cells, dtype=int)

        source_types = None
        if "source_cell_type" in obs:
            if "pred_time" in obs and len(time_grid) > 0:
                pred_time = np.asarray(obs["pred_time"], dtype=float)
                first_mask = np.isclose(pred_time, float(time_grid[0]))
                source_vals = np.asarray(obs.loc[first_mask, "source_cell_type"]).astype(str)
                if source_vals.shape[0] >= n_cells:
                    source_types = source_vals[:n_cells]
                else:
                    fallback = np.asarray(obs["source_cell_type"]).astype(str)
                    source_types = fallback[:n_cells] if fallback.shape[0] >= n_cells else None
            else:
                fallback = np.asarray(obs["source_cell_type"]).astype(str)
                source_types = fallback[:n_cells] if fallback.shape[0] >= n_cells else None

        cycle_rmse = np.asarray(qc_metrics.get("cycle_rmse_by_time", []), dtype=np.float32)
        if cycle_rmse.shape[0] != traj_main.shape[0]:
            cycle_rmse = np.zeros(traj_main.shape[0], dtype=np.float32)

        output_files = manifest.get("output_files", None) if isinstance(manifest, dict) else None
        if not isinstance(output_files, dict):
            output_files = {
                "pred_main_h5ad": str(files["pred_main_h5ad"]),
                "pred_sub_h5ad": str(files["pred_sub_h5ad"]),
                "traj_main_npy": str(files["traj_main_npy"]),
                "traj_sub_npy": str(files["traj_sub_npy"]),
                "weights_npy": str(files["weights_npy"]),
                "velocity_main_npy": str(files["velocity_main_npy"]),
                "growth_main_npy": str(files["growth_main_npy"]),
                "qc_metrics_json": str(files["qc_metrics_json"]),
            }

        primary_domain = 1
        secondary_domain = 2
        domain1_processed_path = ""
        domain2_processed_path = ""
        primary_processed_path = ""
        secondary_processed_path = ""
        if isinstance(manifest, dict):
            primary_domain = int(manifest.get("primary_domain", 1))
            secondary_domain = int(manifest.get("secondary_domain", 2))
            domain1_processed_path = str(manifest.get("domain1_processed_path", ""))
            domain2_processed_path = str(manifest.get("domain2_processed_path", ""))
            primary_processed_path = str(manifest.get("primary_processed_path", ""))
            secondary_processed_path = str(manifest.get("secondary_processed_path", ""))

        qc_metrics["time_grid"] = [float(x) for x in time_grid]
        qc_metrics["n_init_cells"] = int(len(init_idx))
        return {
            "time_grid": np.asarray(time_grid, dtype=float),
            "main_latent": traj_main,
            "sub_latent": traj_sub,
            "velocity_main": velocity_main,
            "growth_main": growth_main,
            "weights": weights,
            "cycle_rmse": cycle_rmse,
            "init_indices": np.asarray(init_idx, dtype=int),
            "source_cell_types": source_types,
            "qc_metrics": qc_metrics,
            "output_files": output_files,
            "primary_domain": int(primary_domain),
            "secondary_domain": int(secondary_domain),
            "domain1_processed_path": domain1_processed_path,
            "domain2_processed_path": domain2_processed_path,
            "primary_processed_path": primary_processed_path,
            "secondary_processed_path": secondary_processed_path,
        }
    except Exception as exc:
        LOGGER.warning("Failed to load cached dense-time outputs from %s (%s). Will run inference.", out_dir, exc)
        return None


def infer_dense_time_dual(
    *,
    dynamic_adata_path: str,
    main_processed_path: str,
    sub_processed_path: str,
    map_model_path: str,
    output_dir: str,
    time_grid: Sequence[float],
    time_key: str = "time_point_processed",
    cell_type_key: str = "cell_type",
    sampling_mode: str = "random",
    sampling_cell_type: Optional[str] = None,
    sampling_indices: Optional[Sequence[int]] = None,
    strict_index_init_time: bool = True,
    max_cells: Optional[int] = None,
    init_time: Optional[float] = None,
    dt: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
    mapper_type: str = "ae",
    mapper_kwargs: Optional[Dict[str, Any]] = None,
    reuse_cache: bool = True,
) -> Dict[str, Any]:
    set_random_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_runtime_device(device)

    domain1_proc = ad.read_h5ad(main_processed_path)
    domain2_proc = ad.read_h5ad(sub_processed_path)
    dyn_adata = ad.read_h5ad(dynamic_adata_path)

    multi_cfg = dyn_adata.uns.get("all_model", {}).get("model_config", {}).get("multi", {})
    mode_src, mode_dst = _parse_multi_mode(multi_cfg.get("mode", (1, 2)))
    primary_domain = int(mode_src)
    secondary_domain = int(mode_dst)

    if primary_domain == 1:
        main_proc = domain1_proc
        sub_proc = domain2_proc
        primary_processed_path = str(Path(main_processed_path).resolve())
        secondary_processed_path = str(Path(sub_processed_path).resolve())
    else:
        main_proc = domain2_proc
        sub_proc = domain1_proc
        primary_processed_path = str(Path(sub_processed_path).resolve())
        secondary_processed_path = str(Path(main_processed_path).resolve())

    observed_times = np.sort(np.asarray(main_proc.obs[time_key], dtype=float))
    t0 = float(np.min(observed_times)) if init_time is None else float(init_time)
    full_grid = _safe_float_list([t0] + [float(t) for t in time_grid])
    mapper_kwargs_use: Dict[str, Any] = dict(mapper_kwargs or {})
    cache_signature = {
        "dynamic_adata_path": str(Path(dynamic_adata_path).resolve()),
        "main_processed_path": str(Path(main_processed_path).resolve()),
        "sub_processed_path": str(Path(sub_processed_path).resolve()),
        "primary_processed_path": primary_processed_path,
        "secondary_processed_path": secondary_processed_path,
        "map_model_path": str(Path(map_model_path).resolve()),
        "time_key": str(time_key),
        "cell_type_key": str(cell_type_key),
        "sampling_mode": str(sampling_mode),
        "sampling_cell_type": (None if sampling_cell_type is None else str(sampling_cell_type)),
        "sampling_indices": _to_serializable(sampling_indices),
        "strict_index_init_time": bool(strict_index_init_time),
        "max_cells": (None if max_cells is None else int(max_cells)),
        "init_time": float(t0),
        "seed": int(seed),
        "dt": float(dt),
        "device": str(device),
        "mapper_type": str(mapper_type),
        "mapper_kwargs": _to_serializable(mapper_kwargs_use),
        "primary_domain": int(primary_domain),
        "secondary_domain": int(secondary_domain),
        "time_grid": [float(x) for x in full_grid],
    }

    init_idx = _select_initial_indices(
        main_proc,
        time_key=time_key,
        init_time=t0,
        sampling_mode=sampling_mode,
        seed=seed,
        cell_type_key=cell_type_key,
        sampling_cell_type=sampling_cell_type,
        sampling_indices=sampling_indices,
        strict_index_init_time=bool(strict_index_init_time),
        max_cells=max_cells,
    )

    model = load_model_from_adata(dyn_adata)
    hidden_dim = int(multi_cfg.get("hidden_dim", 64))
    mapper = build_transport_map(
        map_model_path=map_model_path,
        input_dim1=int(domain1_proc.obsm["X_latent"].shape[1]),
        input_dim2=int(domain2_proc.obsm["X_latent"].shape[1]),
        mode=(int(primary_domain), int(secondary_domain)),
        hidden_dim=hidden_dim,
        device=device,
        mapper_type=mapper_type,
        mapper_kwargs=mapper_kwargs_use,
    )

    cached = (
        _load_cached_inference_artifacts(out_dir, expected_signature=cache_signature)
        if bool(reuse_cache)
        else None
    )
    if cached is not None:
        LOGGER.info("Reusing cached dense-time inference outputs from %s", out_dir)
        return {
            "output_dir": str(out_dir),
            "time_grid": np.asarray(cached["time_grid"], dtype=float),
            "main_latent": np.asarray(cached["main_latent"], dtype=np.float32),
            "sub_latent": np.asarray(cached["sub_latent"], dtype=np.float32),
            "velocity_main": np.asarray(cached["velocity_main"], dtype=np.float32),
            "growth_main": np.asarray(cached["growth_main"], dtype=np.float32),
            "weights": np.asarray(cached["weights"], dtype=np.float32),
            "cycle_rmse": np.asarray(cached["cycle_rmse"], dtype=np.float32),
            "main_processed": main_proc,
            "sub_processed": sub_proc,
            "dynamic_adata": dyn_adata,
            "model": model,
            "mapper": mapper,
            "init_indices": np.asarray(cached["init_indices"], dtype=int),
            "source_cell_types": cached["source_cell_types"],
            "qc_metrics": dict(cached["qc_metrics"]),
            "output_files": dict(cached["output_files"]),
            "primary_domain": int(cached.get("primary_domain", primary_domain)),
            "secondary_domain": int(cached.get("secondary_domain", secondary_domain)),
            "domain1_processed": domain1_proc,
            "domain2_processed": domain2_proc,
            "domain1_processed_path": str(Path(main_processed_path).resolve()),
            "domain2_processed_path": str(Path(sub_processed_path).resolve()),
            "primary_processed_path": str(cached.get("primary_processed_path", primary_processed_path)),
            "secondary_processed_path": str(cached.get("secondary_processed_path", secondary_processed_path)),
            "inference_device": str(device),
        }

    x0 = np.asarray(main_proc.obsm["X_latent"])[init_idx]
    traj_main = _build_ode_trajectory(model, x0, full_grid, adata=dyn_adata, dt=dt, device=device)
    traj_sub = _map_to_secondary(mapper, traj_main["main_latent"])

    source_types = None
    if cell_type_key in main_proc.obs:
        source_types = np.asarray(main_proc.obs.iloc[init_idx][cell_type_key]).astype(str)

    main_pred = _build_prediction_adata(
        traj_main["main_latent"],
        traj_main["weights_abs"],
        init_idx,
        source_types,
        traj_main["time_grid"],
        velocity=traj_main["velocity_latent"],
        growth=traj_main["growth_rate"],
    )
    sub_pred = _build_prediction_adata(
        traj_sub["sub_latent"],
        traj_main["weights_abs"],
        init_idx,
        source_types,
        traj_main["time_grid"],
    )

    main_h5ad = out_dir / "pred_main.h5ad"
    sub_h5ad = out_dir / "pred_sub.h5ad"
    main_pred.write_h5ad(main_h5ad)
    sub_pred.write_h5ad(sub_h5ad)

    np.save(out_dir / "traj_main.npy", traj_main["main_latent"])
    np.save(out_dir / "traj_sub.npy", traj_sub["sub_latent"])
    np.save(out_dir / "weights.npy", traj_main["weights_abs"])
    np.save(out_dir / "velocity_main.npy", traj_main["velocity_latent"])
    np.save(out_dir / "growth_main.npy", traj_main["growth_rate"])

    qc_metrics = {
        "time_grid": [float(x) for x in traj_main["time_grid"]],
        "n_init_cells": int(len(init_idx)),
        "sampling_mode": sampling_mode,
        "sampling_cell_type": (None if sampling_cell_type is None else str(sampling_cell_type)),
        "sampling_indices": _to_serializable(sampling_indices),
        "strict_index_init_time": bool(strict_index_init_time),
        "weight_sum_by_time": np.sum(traj_main["weights_abs"], axis=1).tolist(),
        "weight_mean_by_time": np.mean(traj_main["weights_abs"], axis=1).tolist(),
    }
    _write_json(out_dir / "qc_metrics.json", qc_metrics)

    manifest = {
        "dynamic_adata_path": str(dynamic_adata_path),
        "main_processed_path": str(main_processed_path),
        "sub_processed_path": str(sub_processed_path),
        "domain1_processed_path": str(Path(main_processed_path).resolve()),
        "domain2_processed_path": str(Path(sub_processed_path).resolve()),
        "primary_processed_path": primary_processed_path,
        "secondary_processed_path": secondary_processed_path,
        "primary_domain": int(primary_domain),
        "secondary_domain": int(secondary_domain),
        "map_model_path": str(map_model_path),
        "time_key": str(time_key),
        "cell_type_key": str(cell_type_key),
        "sampling_mode": sampling_mode,
        "sampling_cell_type": (None if sampling_cell_type is None else str(sampling_cell_type)),
        "sampling_indices": _to_serializable(sampling_indices),
        "strict_index_init_time": bool(strict_index_init_time),
        "max_cells": (None if max_cells is None else int(max_cells)),
        "init_time": float(t0),
        "seed": int(seed),
        "dt": float(dt),
        "device": str(device),
        "mapper_type": str(mapper_type),
        "mapper_kwargs": _to_serializable(mapper_kwargs_use),
        "time_grid": [float(x) for x in traj_main["time_grid"]],
        "n_init_cells": int(len(init_idx)),
        "cache_signature": cache_signature,
        "output_files": {
            "pred_main_h5ad": str(main_h5ad),
            "pred_sub_h5ad": str(sub_h5ad),
            "traj_main_npy": str(out_dir / "traj_main.npy"),
            "traj_sub_npy": str(out_dir / "traj_sub.npy"),
            "weights_npy": str(out_dir / "weights.npy"),
            "velocity_main_npy": str(out_dir / "velocity_main.npy"),
            "growth_main_npy": str(out_dir / "growth_main.npy"),
            "qc_metrics_json": str(out_dir / "qc_metrics.json"),
        },
    }
    _write_json(out_dir / "manifest.json", manifest)

    return {
        "output_dir": str(out_dir),
        "time_grid": np.asarray(traj_main["time_grid"], dtype=float),
        "main_latent": traj_main["main_latent"],
        "sub_latent": traj_sub["sub_latent"],
        "velocity_main": traj_main["velocity_latent"],
        "growth_main": traj_main["growth_rate"],
        "weights": traj_main["weights_abs"],
        "main_processed": main_proc,
        "sub_processed": sub_proc,
        "dynamic_adata": dyn_adata,
        "model": model,
        "mapper": mapper,
        "init_indices": init_idx,
        "source_cell_types": source_types,
        "qc_metrics": qc_metrics,
        "output_files": manifest["output_files"],
        "primary_domain": int(primary_domain),
        "secondary_domain": int(secondary_domain),
        "domain1_processed": domain1_proc,
        "domain2_processed": domain2_proc,
        "domain1_processed_path": str(Path(main_processed_path).resolve()),
        "domain2_processed_path": str(Path(sub_processed_path).resolve()),
        "primary_processed_path": primary_processed_path,
        "secondary_processed_path": secondary_processed_path,
        "inference_device": str(device),
    }


@dataclass
class DenseTimeClassifierBundle:
    classifier: Any
    classes: np.ndarray
    train_space: str
    map_batch: int
    selected_cell_types: List[str]
    cell_type_filter: Dict[str, Any]

def _normalize_label_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = [values]
    elif isinstance(values, (list, tuple, np.ndarray)):
        raw = list(values)
    else:
        raw = [values]

    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        lab = str(item).strip()
        if not lab:
            continue
        key = lab.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(lab)
    return out


def _match_requested_labels(requested: Sequence[str], available: Sequence[str]) -> Tuple[List[str], List[str]]:
    avail = [str(x) for x in available]
    avail_ci = {x.lower(): x for x in avail}
    matched: List[str] = []
    missing: List[str] = []
    seen: set[str] = set()
    for req in requested:
        req_s = str(req).strip()
        if not req_s:
            continue
        key = req_s.lower()
        hit = avail_ci.get(key)
        if hit is None:
            fuzzy = [x for x in avail if key in x.lower()]
            if len(fuzzy) == 1:
                hit = fuzzy[0]
        if hit is None:
            missing.append(req_s)
            continue
        hit_key = hit.lower()
        if hit_key in seen:
            continue
        seen.add(hit_key)
        matched.append(hit)
    return matched, missing


def _resolve_cell_type_filter(available_classes: Sequence[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    available = [str(x) for x in available_classes]
    include_req = _normalize_label_list(cfg.get("include_cell_types", []))
    exclude_req = _normalize_label_list(cfg.get("exclude_cell_types", []))

    include_hit, include_missing = _match_requested_labels(include_req, available)
    if include_req:
        selected = include_hit.copy()
    else:
        selected = available.copy()

    exclude_pool = selected if include_req else available
    exclude_hit, exclude_missing = _match_requested_labels(exclude_req, exclude_pool)
    if exclude_hit:
        exclude_lut = {x.lower() for x in exclude_hit}
        selected = [x for x in selected if x.lower() not in exclude_lut]

    if not selected:
        raise ValueError(
            "No cell-type classes remain after include/exclude filtering. "
            f"available={available}, include={include_req}, exclude={exclude_req}"
        )

    return {
        "selection_active": bool(include_req or exclude_req),
        "available_classes": available,
        "include_requested": include_req,
        "exclude_requested": exclude_req,
        "include_matched": include_hit,
        "exclude_matched": exclude_hit,
        "include_missing": include_missing,
        "exclude_missing": exclude_missing,
        "selected_cell_types": selected,
    }


def _mask_for_classes(labels: np.ndarray, keep_classes: Sequence[str]) -> np.ndarray:
    keep = {str(x).lower() for x in keep_classes}
    y = np.asarray(labels).astype(str)
    return np.asarray([str(v).lower() in keep for v in y], dtype=bool)


def _select_output_classes(available_classes: Sequence[str], selected_cell_types: Sequence[str]) -> np.ndarray:
    available = [str(x) for x in available_classes]
    if not selected_cell_types:
        return np.asarray(available_classes)
    matched, _ = _match_requested_labels(selected_cell_types, available)
    if not matched:
        return np.asarray(available_classes)
    out: List[Any] = []
    for want in matched:
        for c in available_classes:
            if str(c).lower() == str(want).lower():
                out.append(c)
                break
    return np.asarray(out)


def _fit_classifier(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
    *,
    fit_classes: Optional[Sequence[str]] = None,
) -> Any:
    X_fit = np.asarray(X)
    y_fit = np.asarray(y).astype(str)
    if fit_classes:
        mask = _mask_for_classes(y_fit, fit_classes)
        if not np.any(mask):
            raise ValueError(f"No training rows found for fit_classes={list(fit_classes)}")
        X_fit = X_fit[mask]
        y_fit = y_fit[mask]
    if len(np.unique(y_fit)) < 2:
        raise ValueError("Classifier training requires at least 2 classes.")
    clf = LogisticRegression(
        max_iter=int(cfg.get("max_iter", 1200)),
        solver="lbfgs",
        multi_class="auto",
        class_weight="balanced",
    )
    clf.fit(X_fit, y_fit)
    return clf


def _predict_proba(clf: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        classes = np.asarray(clf.classes_)
        return proba, classes
    pred = clf.predict(X)
    classes = np.unique(pred)
    out = np.zeros((len(pred), len(classes)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, lab in enumerate(pred):
        out[i, class_to_idx[lab]] = 1.0
    return out, classes


def _predict_proba_for_classes(
    classifier: Any,
    X: np.ndarray,
    *,
    target_classes: Optional[Sequence[str]] = None,
    renormalize_subset: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    proba, classes = _predict_proba(classifier, X)
    base_classes = np.asarray(classes)
    if target_classes is None:
        use_classes = base_classes
    else:
        use_classes = np.asarray(list(target_classes))
    aligned = _align_proba_to_classes(np.asarray(proba, dtype=float), base_classes, use_classes)
    if renormalize_subset:
        denom = aligned.sum(axis=1, keepdims=True)
        denom = np.where(denom <= 1e-12, 1.0, denom)
        aligned = aligned / denom
    return np.asarray(aligned, dtype=float), np.asarray(use_classes)


def _save_classifier(path: Path, clf: Any, classes: np.ndarray, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"classifier": clf, "classes": np.asarray(classes), "meta": meta}
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _batched_mapper_forward(mapper: Any, x_np: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    out: List[np.ndarray] = []
    for i in range(0, x_np.shape[0], int(batch_size)):
        xb = x_np[i : i + int(batch_size)]
        yb = mapper.T(xb)
        if isinstance(yb, torch.Tensor):
            arr = yb.detach().cpu().numpy()
        else:
            arr = np.asarray(yb)
        out.append(np.asarray(arr, dtype=np.float32))
    return np.concatenate(out, axis=0)


def _resolve_classifier_train_space(classifier_cfg: Dict[str, Any]) -> str:
    train_space = str(classifier_cfg.get("train_space", "joint")).strip().lower()
    if train_space not in {"main", "sub", "joint"}:
        LOGGER.warning("unsupported classifier.train_space=%s, fallback to joint", train_space)
        train_space = "joint"
    return train_space


def _normalize_classifier_cfg(classifier_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(classifier_cfg or {})

    if "train_space" not in cfg and "primary_space" in cfg:
        cfg["train_space"] = cfg.get("primary_space")
        LOGGER.warning("classifier.primary_space is deprecated; use classifier.train_space")

    raw_type = str(cfg.get("type", "linear")).strip().lower()
    type_alias = {
        "linear": "linear",
        "logreg": "linear",
        "logistic": "linear",
        "logistic_regression": "linear",
    }
    if raw_type not in type_alias:
        LOGGER.warning("unsupported classifier.type=%s, forcing linear", raw_type)
    cfg["type"] = type_alias.get(raw_type, "linear")

    deprecated_keys = ["n_neighbors", "knn_weights", "class_weight_mode", "class_weight_power", "primary_space"]
    for key in deprecated_keys:
        if key in cfg:
            LOGGER.warning("classifier.%s is deprecated and ignored (linear classifier only).", key)
            cfg.pop(key, None)

    return cfg


def _map_main_to_sub_like(
    mapper: Any,
    main_latent: np.ndarray,
    *,
    map_batch: int,
) -> np.ndarray:
    x = np.asarray(main_latent, dtype=np.float32)
    if x.ndim == 2:
        return _batched_mapper_forward(mapper, x, batch_size=map_batch)
    if x.ndim == 3:
        t_num, n_cells, latent_dim = x.shape
        flat = x.reshape(t_num * n_cells, latent_dim)
        mapped = _batched_mapper_forward(mapper, flat, batch_size=map_batch)
        return np.asarray(mapped, dtype=np.float32).reshape(t_num, n_cells, -1)
    raise ValueError(f"main_latent must be 2D or 3D, got shape={x.shape}")


def _build_classifier_space_latent(
    *,
    train_space: str,
    main_latent: np.ndarray,
    sub_latent: Optional[np.ndarray],
    mapper: Any,
    map_batch: int,
) -> np.ndarray:
    main_arr = np.asarray(main_latent, dtype=np.float32)
    if train_space == "main":
        return main_arr

    if sub_latent is not None:
        sub_arr = np.asarray(sub_latent, dtype=np.float32)
        if sub_arr.shape[:-1] != main_arr.shape[:-1]:
            raise ValueError(
                f"sub_latent shape {sub_arr.shape} is incompatible with main_latent shape {main_arr.shape}"
            )
    else:
        sub_arr = _map_main_to_sub_like(mapper, main_arr, map_batch=map_batch)

    if train_space == "sub":
        return sub_arr
    if train_space == "joint":
        return np.concatenate([main_arr, sub_arr], axis=main_arr.ndim - 1)
    raise ValueError(f"Unsupported train_space={train_space}")


def build_classifier_bundle(
    *,
    main_proc: AnnData,
    sub_proc: AnnData,
    mapper: Any,
    cell_type_key: str,
    classifier_cfg: Dict[str, Any],
    reports_dir: Path,
) -> Tuple[DenseTimeClassifierBundle, Dict[str, Any]]:
    if cell_type_key not in main_proc.obs:
        raise KeyError(f"{cell_type_key} not found in main_processed.obs")

    X_main = np.asarray(main_proc.obsm["X_latent"], dtype=np.float32)
    y_main = np.asarray(main_proc.obs[cell_type_key]).astype(str)
    train_space = _resolve_classifier_train_space(classifier_cfg)
    map_batch = int(classifier_cfg.get("joint_map_batch", 2048))

    X_sub_obs: Optional[np.ndarray] = None
    if "X_latent" in sub_proc.obsm and len(sub_proc) == len(main_proc):
        X_sub_obs = np.asarray(sub_proc.obsm["X_latent"], dtype=np.float32)
    elif train_space in {"sub", "joint"}:
        LOGGER.warning(
            "sub space latent unavailable or unpaired (len(main)=%d, len(sub)=%d); fallback to mapper(main).",
            len(main_proc),
            len(sub_proc),
        )

    X_train = _build_classifier_space_latent(
        train_space=train_space,
        main_latent=X_main,
        sub_latent=X_sub_obs,
        mapper=mapper,
        map_batch=map_batch,
    )

    cell_type_filter = _resolve_cell_type_filter(np.unique(y_main), classifier_cfg)
    selected_cell_types = [str(x) for x in cell_type_filter["selected_cell_types"]]
    if cell_type_filter.get("include_missing"):
        LOGGER.warning("classifier.include_cell_types has unknown labels: %s", cell_type_filter["include_missing"])
    if cell_type_filter.get("exclude_missing"):
        LOGGER.warning("classifier.exclude_cell_types has unknown labels: %s", cell_type_filter["exclude_missing"])
    subset_training = bool(cell_type_filter["selection_active"])

    fit_classes = selected_cell_types if subset_training else None
    classifier = _fit_classifier(X_train, y_main, classifier_cfg, fit_classes=fit_classes)
    if subset_training:
        train_mask = _mask_for_classes(y_main, selected_cell_types)
        pred_train = classifier.predict(X_train[train_mask])
        train_acc = float(accuracy_score(y_main[train_mask], pred_train))
        train_n = int(np.sum(train_mask))
    else:
        pred_train = classifier.predict(X_train)
        train_acc = float(accuracy_score(y_main, pred_train))
        train_n = int(len(y_main))

    classes_out = _select_output_classes(np.asarray(classifier.classes_), selected_cell_types)

    bundle = DenseTimeClassifierBundle(
        classifier=classifier,
        classes=np.asarray(classes_out),
        train_space=train_space,
        map_batch=map_batch,
        selected_cell_types=selected_cell_types,
        cell_type_filter=cell_type_filter,
    )

    cls_dir = reports_dir / "classifiers"
    classifier_path = cls_dir / f"{train_space}_classifier.pkl"
    _save_classifier(
        classifier_path,
        classifier,
        np.asarray(classifier.classes_),
        {
            "space": train_space,
            "output_classes": [str(x) for x in bundle.classes],
            "class_weight": "balanced",
        },
    )

    eval_obj = {
        "train_space": train_space,
        "classifier_type": "logistic_regression",
        "class_weight": "balanced",
        "train_accuracy": train_acc,
        "train_samples": train_n,
        "feature_dim": int(X_train.shape[1]),
        "num_classes": int(len(bundle.classes)),
        "raw_num_classes": int(len(np.asarray(classifier.classes_))),
        "subset_training": bool(subset_training),
        "selected_cell_types": [str(x) for x in selected_cell_types],
        "cell_type_filter": cell_type_filter,
        "cell_type_output_classes": [str(x) for x in bundle.classes],
        "classifier_path": str(classifier_path),
    }
    _write_json(cls_dir / "classifier_manifest.json", eval_obj)
    return bundle, eval_obj


def _build_real_cell_type_profiles(adata: AnnData, time_key: str, cell_type_key: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    times = np.asarray(adata.obs[time_key], dtype=float)
    labels = np.asarray(adata.obs[cell_type_key]).astype(str)
    classes = np.unique(labels)
    for t in np.sort(np.unique(times)):
        mask = np.isclose(times, float(t))
        counts = pd.Series(labels[mask]).value_counts(normalize=True)
        n_cells = int(np.sum(mask))
        for cls in classes:
            y_bin = (labels[mask] == cls).astype(float)
            q40, q50, q60 = _weighted_quantile(y_bin, [0.4, 0.5, 0.6], None)
            rows.append(
                {
                    "time": float(t),
                    "cell_type": str(cls),
                    "prob": float(counts.get(cls, 0.0)),
                    "prob_p40": float(q40),
                    "prob_p50": float(q50),
                    "prob_p60": float(q60),
                    "n_cells": int(n_cells),
                    "weight_sum": float(n_cells),
                    "source": "real",
                }
            )
    return pd.DataFrame(rows)


def _build_pred_cell_type_profiles(
    latent: np.ndarray,
    time_grid: Sequence[float],
    classifier: Any,
    space: str,
    weights_by_time: Optional[np.ndarray] = None,
    target_classes: Optional[Sequence[str]] = None,
    renormalize_subset: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    classes = np.asarray(classifier.classes_) if target_classes is None else np.asarray(list(target_classes))
    weights_arr = None
    if weights_by_time is not None:
        weights_arr = np.asarray(weights_by_time, dtype=float)
        if weights_arr.shape != np.asarray(latent).shape[:2]:
            raise ValueError(
                f"weights_by_time shape {weights_arr.shape} mismatches latent shape {np.asarray(latent).shape}"
            )
    for i, t in enumerate(time_grid):
        proba, _ = _predict_proba_for_classes(
            classifier,
            latent[i],
            target_classes=classes,
            renormalize_subset=renormalize_subset,
        )
        wt = None if weights_arr is None else weights_arr[i]
        mean_probs = _weighted_row_mean(proba, wt)
        weight_sum = float(np.sum(wt)) if wt is not None else float(proba.shape[0])
        for j, (c, val) in enumerate(zip(classes, mean_probs)):
            q40, q50, q60 = _weighted_quantile(proba[:, j], [0.4, 0.5, 0.6], wt)
            rows.append(
                {
                    "time": float(t),
                    "cell_type": str(c),
                    "prob": float(val),
                    "prob_p40": float(q40),
                    "prob_p50": float(q50),
                    "prob_p60": float(q60),
                    "n_cells": int(proba.shape[0]),
                    "weight_sum": float(weight_sum),
                    "space": space,
                    "source": "pred",
                }
            )
    return pd.DataFrame(rows)


def _evaluate_series_against_anchors(
    pred_times: np.ndarray,
    pred_vals: np.ndarray,
    anchor_times: np.ndarray,
    anchor_vals: np.ndarray,
) -> Dict[str, Any]:
    if len(anchor_times) < 2:
        return {
            "anchor_pearson": 0.0,
            "anchor_spearman": 0.0,
            "direction_consistency": 0.0,
            "interval_pass_ratio": 0.0,
        }

    pred_at_anchor = _interp_at(pred_times, pred_vals, anchor_times)
    anchor_pearson = _pearson(pred_at_anchor, anchor_vals)
    anchor_spearman = _spearman(pred_at_anchor, anchor_vals)

    direction_hits = 0
    direction_total = 0
    interval_hits = 0
    interval_total = 0
    for i in range(len(anchor_times) - 1):
        a_t, b_t = float(anchor_times[i]), float(anchor_times[i + 1])
        a_v, b_v = float(anchor_vals[i]), float(anchor_vals[i + 1])
        pred_a = float(pred_at_anchor[i])
        pred_b = float(pred_at_anchor[i + 1])

        real_delta = b_v - a_v
        pred_delta = pred_b - pred_a
        if abs(real_delta) < 1e-8 and abs(pred_delta) < 1e-8:
            direction_hits += 1
        elif np.sign(real_delta) == np.sign(pred_delta):
            direction_hits += 1
        direction_total += 1

        in_seg = np.where((pred_times > a_t) & (pred_times < b_t))[0]
        if len(in_seg) == 0:
            continue
        lower = min(a_v, b_v)
        upper = max(a_v, b_v)
        tol = max(0.05 * (abs(lower) + abs(upper) + 1e-6), 0.2 * abs(upper - lower), 1e-4)
        vals = pred_vals[in_seg]
        ok = np.logical_and(vals >= lower - tol, vals <= upper + tol)
        interval_hits += int(np.sum(ok))
        interval_total += int(len(vals))

    return {
        "anchor_pearson": float(anchor_pearson),
        "anchor_spearman": float(anchor_spearman),
        "direction_consistency": float(direction_hits / max(direction_total, 1)),
        "interval_pass_ratio": float(interval_hits / max(interval_total, 1)),
    }


def evaluate_cell_type_prior_over_time(pred_df: pd.DataFrame, real_df: pd.DataFrame) -> Dict[str, Any]:
    cell_types = sorted(set(pred_df["cell_type"]).intersection(set(real_df["cell_type"])))
    details: Dict[str, Dict[str, Any]] = {}
    for cell_type in cell_types:
        p = pred_df[pred_df["cell_type"] == cell_type].sort_values("time")
        r = real_df[real_df["cell_type"] == cell_type].sort_values("time")
        eval_obj = _evaluate_series_against_anchors(
            p["time"].to_numpy(dtype=float),
            p["prob"].to_numpy(dtype=float),
            r["time"].to_numpy(dtype=float),
            r["prob"].to_numpy(dtype=float),
        )
        eval_obj["score"] = float(np.mean([max(0.0, eval_obj["anchor_spearman"]), eval_obj["direction_consistency"], eval_obj["interval_pass_ratio"]]))
        details[cell_type] = eval_obj

    return {
        "cell_type_details": details,
        "cell_type_score_median": float(np.median([x["score"] for x in details.values()])) if details else 0.0,
    }


def _build_real_profiles(marker_frame: pd.DataFrame, obs_times: np.ndarray, label: str) -> pd.DataFrame:
    if marker_frame.empty:
        return pd.DataFrame(columns=["time", "marker", "mean_expr", "expr_p40", "expr_p50", "expr_p60", "n_cells", "space", "source"])
    rows: List[Dict[str, Any]] = []
    for t in np.sort(np.unique(obs_times.astype(float))):
        mask = np.isclose(obs_times, float(t))
        if not np.any(mask):
            continue
        sub = marker_frame.loc[mask]
        mean_vals = sub.mean(axis=0)
        for marker, val in mean_vals.items():
            arr = np.asarray(sub[marker], dtype=float)
            q40, q50, q60 = _weighted_quantile(arr, [0.4, 0.5, 0.6], None)
            rows.append(
                {
                    "time": float(t),
                    "marker": str(marker),
                    "mean_expr": float(val),
                    "expr_p40": float(q40),
                    "expr_p50": float(q50),
                    "expr_p60": float(q60),
                    "n_cells": int(arr.shape[0]),
                    "space": label,
                    "source": "real",
                }
            )
    return pd.DataFrame(rows)


def _build_program_profiles(
    *,
    main_proc: AnnData,
    pred_time_grid: np.ndarray,
    time_key: str,
    programs: Dict[str, List[str]],
    main_feature_pred: np.ndarray,
    main_feature_names: Sequence[str],
    main_feature_real: np.ndarray,
    main_feature_real_names: Sequence[str],
    pred_weights: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    union_program_genes = sorted({g for genes in programs.values() for g in genes})
    pred_lut = {str(n): i for i, n in enumerate(main_feature_names)}
    real_lut = {str(n): i for i, n in enumerate(main_feature_real_names)}
    valid_genes = [g for g in union_program_genes if g in pred_lut and g in real_lut]
    missing = sorted(set(union_program_genes) - set(valid_genes))
    if not valid_genes:
        empty = pd.DataFrame(columns=["time", "program", "score", "score_p40", "score_p50", "score_p60", "n_cells", "source"])
        return empty, empty, missing

    pred_idx = [pred_lut[g] for g in valid_genes]
    real_idx = [real_lut[g] for g in valid_genes]
    real_full = np.asarray(main_feature_real[:, real_idx], dtype=np.float32)
    ref_means = np.asarray(real_full.mean(axis=0), dtype=np.float32)
    ref_stds = np.asarray(np.std(real_full, axis=0), dtype=np.float32)
    ref_stds = np.where(ref_stds <= 0.0, 1.0, ref_stds).astype(np.float32)
    gene_pos = {g: i for i, g in enumerate(valid_genes)}

    pred_weights_arr = None
    if pred_weights is not None:
        pred_weights_arr = np.asarray(pred_weights, dtype=float)
        if pred_weights_arr.shape != np.asarray(main_feature_pred).shape[:2]:
            raise ValueError(
                f"pred_weights shape {pred_weights_arr.shape} mismatches main_feature_pred shape {np.asarray(main_feature_pred).shape}"
            )

    pred_rows: List[Dict[str, Any]] = []
    for i, t in enumerate(pred_time_grid):
        pred_expr = np.asarray(main_feature_pred[i][:, pred_idx], dtype=np.float32)
        pred_z_cell = (pred_expr - ref_means[None, :]) / ref_stds[None, :]
        wt = None if pred_weights_arr is None else pred_weights_arr[i]
        for program_name, genes in programs.items():
            idx = [gene_pos[g] for g in genes if g in gene_pos]
            if len(idx) == 0:
                continue
            cell_scores = np.asarray(pred_z_cell[:, idx].mean(axis=1), dtype=float)
            mean_score = float(_weighted_row_mean(cell_scores[:, None], wt)[0])
            q40, q50, q60 = _weighted_quantile(cell_scores, [0.4, 0.5, 0.6], wt)
            pred_rows.append(
                {
                    "time": float(t),
                    "program": str(program_name),
                    "score": float(mean_score),
                    "score_p40": float(q40),
                    "score_p50": float(q50),
                    "score_p60": float(q60),
                    "n_cells": int(cell_scores.shape[0]),
                    "source": "pred",
                }
            )

    real_rows: List[Dict[str, Any]] = []
    obs_times = np.asarray(main_proc.obs[time_key], dtype=float)
    unique_obs_times = np.sort(np.unique(obs_times))
    for t in unique_obs_times:
        mask = np.isclose(obs_times, float(t))
        if not np.any(mask):
            continue
        real_expr = np.asarray(real_full[mask], dtype=np.float32)
        real_z_cell = (real_expr - ref_means[None, :]) / ref_stds[None, :]
        for program_name, genes in programs.items():
            idx = [gene_pos[g] for g in genes if g in gene_pos]
            if len(idx) == 0:
                continue
            cell_scores = np.asarray(real_z_cell[:, idx].mean(axis=1), dtype=float)
            mean_score = float(np.mean(cell_scores))
            q40, q50, q60 = _weighted_quantile(cell_scores, [0.4, 0.5, 0.6], None)
            real_rows.append(
                {
                    "time": float(t),
                    "program": str(program_name),
                    "score": float(mean_score),
                    "score_p40": float(q40),
                    "score_p50": float(q50),
                    "score_p60": float(q60),
                    "n_cells": int(cell_scores.shape[0]),
                    "source": "real",
                }
            )

    pred_df = pd.DataFrame(pred_rows, columns=["time", "program", "score", "score_p40", "score_p50", "score_p60", "n_cells", "source"])
    real_df = pd.DataFrame(real_rows, columns=["time", "program", "score", "score_p40", "score_p50", "score_p60", "n_cells", "source"])
    missing = sorted(set(union_program_genes) - set(valid_genes))
    return pred_df, real_df, missing


def _resolve_secondary_programs(
    programs_main: Dict[str, List[str]],
    programs_sec_raw: Any,
    sec_feature_names: Sequence[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    sec_name_set = {str(x) for x in sec_feature_names}

    requested: Dict[str, List[str]] = {}
    if isinstance(programs_sec_raw, dict) and len(programs_sec_raw) > 0:
        for name, genes in programs_sec_raw.items():
            gene_list = _normalize_str_list(genes)
            if gene_list:
                requested[str(name)] = gene_list
    else:
        for name, genes in programs_main.items():
            gene_list = _normalize_str_list(genes)
            if gene_list:
                requested[str(name)] = gene_list

    resolved: Dict[str, List[str]] = {}
    requested_union: List[str] = []
    resolved_union: List[str] = []
    for name, genes in requested.items():
        requested_union.extend([str(g) for g in genes])
        keep = [str(g) for g in genes if str(g) in sec_name_set]
        if keep:
            resolved[str(name)] = keep
            resolved_union.extend(keep)

    missing = sorted(set(requested_union) - set(resolved_union))
    return resolved, missing


def evaluate_mechanism_prior_over_time(program_pred_df: pd.DataFrame, program_real_df: pd.DataFrame) -> Dict[str, Any]:
    programs = sorted(set(program_pred_df["program"]).intersection(set(program_real_df["program"])))
    details: Dict[str, Dict[str, Any]] = {}
    for program in programs:
        p = program_pred_df[program_pred_df["program"] == program].sort_values("time")
        r = program_real_df[program_real_df["program"] == program].sort_values("time")
        eval_obj = _evaluate_series_against_anchors(
            p["time"].to_numpy(dtype=float),
            p["score"].to_numpy(dtype=float),
            r["time"].to_numpy(dtype=float),
            r["score"].to_numpy(dtype=float),
        )
        eval_obj["score"] = float(np.mean([max(0.0, eval_obj["anchor_spearman"]), eval_obj["direction_consistency"], eval_obj["interval_pass_ratio"]]))
        details[program] = eval_obj
    return {
        "program_details": details,
        "program_score_median": float(np.median([x["score"] for x in details.values()])) if details else 0.0,
    }


def _resolve_mapper_device(mapper: Any, fallback_device: str) -> torch.device:
    mapper_device = getattr(mapper, "device", None)
    if mapper_device is None:
        return torch.device(fallback_device)
    if isinstance(mapper_device, torch.device):
        return mapper_device
    return torch.device(str(mapper_device))


def _mapper_forward_with_graph(mapper: Any, x_t: torch.Tensor, *, reverse: bool) -> torch.Tensor:
    # TransportMap.T/T_rev wraps no_grad, so use model.forward_single directly when available.
    if hasattr(mapper, "model") and getattr(mapper, "model", None) is not None and hasattr(mapper.model, "forward_single"):
        if not hasattr(mapper, "mode"):
            raise AttributeError("mapper has model.forward_single but no mode attribute.")
        mode = tuple(mapper.mode)
        in_domain = int(mode[1] if reverse else mode[0])
        out_domain = int(mode[0] if reverse else mode[1])
        out = mapper.model.forward_single(x_t, in_domain=in_domain, out_domain=out_domain)[0]
        return out

    fn = mapper.T_rev if reverse else mapper.T
    out = fn(x_t)
    if isinstance(out, np.ndarray):
        return torch.as_tensor(out, dtype=torch.float32, device=x_t.device)
    if not isinstance(out, torch.Tensor):
        return torch.as_tensor(out, dtype=torch.float32, device=x_t.device)
    return out.to(device=x_t.device, dtype=torch.float32)


def _batched_main_to_sub_terms_autograd(
    mapper: Any,
    x_main: np.ndarray,
    xdot_main: np.ndarray,
    jac_z: np.ndarray,
    *,
    batch_size: int,
    include_hessian_term: bool,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_main, dtype=np.float32)
    xdot_arr = np.asarray(xdot_main, dtype=np.float32)
    jac_z_arr = np.asarray(jac_z, dtype=np.float32)
    if x_arr.ndim != 2:
        raise ValueError(f"x_main must be 2D, got shape={x_arr.shape}")
    if xdot_arr.shape != x_arr.shape:
        raise ValueError(f"xdot_main shape {xdot_arr.shape} mismatches x_main shape {x_arr.shape}")
    if jac_z_arr.ndim != 3 or jac_z_arr.shape[:2] != x_arr.shape:
        raise ValueError(f"jac_z shape {jac_z_arr.shape} mismatches x_main shape {x_arr.shape}")

    mapper_device = _resolve_mapper_device(mapper, device)
    jac_chunks: List[np.ndarray] = []
    term1_chunks: List[np.ndarray] = []
    term2_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []

    step = max(1, int(batch_size))
    for i in range(0, x_arr.shape[0], step):
        xb = torch.as_tensor(x_arr[i : i + step], dtype=torch.float32, device=mapper_device).clone().detach().requires_grad_(True)
        xdot_b = torch.as_tensor(xdot_arr[i : i + step], dtype=torch.float32, device=mapper_device)
        jac_z_b = torch.as_tensor(jac_z_arr[i : i + step], dtype=torch.float32, device=mapper_device)

        yb = _mapper_forward_with_graph(mapper, xb, reverse=False)
        if yb.ndim != 2:
            raise ValueError(f"mapper forward output must be 2D, got shape={tuple(yb.shape)}")
        out_dim = int(yb.shape[1])

        jac_rows: List[torch.Tensor] = []
        term1_rows: List[torch.Tensor] = []
        for out_idx in range(out_dim):
            retain_next = out_idx < (out_dim - 1)
            grad_row = torch.autograd.grad(
                outputs=yb[:, out_idx].sum(),
                inputs=xb,
                retain_graph=(True if include_hessian_term else retain_next),
                create_graph=include_hessian_term,
                allow_unused=False,
            )[0]
            jac_rows.append(grad_row.detach())

            if include_hessian_term:
                scalar = torch.sum(grad_row * xdot_b, dim=1).sum()
                hvp = torch.autograd.grad(
                    outputs=scalar,
                    inputs=xb,
                    retain_graph=retain_next,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                term1_rows.append(hvp.detach())

        jac_b = torch.stack(jac_rows, dim=1)  # (b, k_sub, k_main)
        term2_b = torch.einsum("bik,bkj->bij", jac_b, jac_z_b)  # (b, k_sub, k_main)
        if include_hessian_term:
            term1_b = torch.stack(term1_rows, dim=1)
        else:
            term1_b = torch.zeros_like(term2_b)

        jac_chunks.append(np.asarray(jac_b.cpu().numpy(), dtype=np.float32))
        term1_chunks.append(np.asarray(term1_b.cpu().numpy(), dtype=np.float32))
        term2_chunks.append(np.asarray(term2_b.cpu().numpy(), dtype=np.float32))
        y_chunks.append(np.asarray(yb.detach().cpu().numpy(), dtype=np.float32))

    jac_map = np.concatenate(jac_chunks, axis=0)
    term1 = np.concatenate(term1_chunks, axis=0)
    term2 = np.concatenate(term2_chunks, axis=0)
    y_sub = np.concatenate(y_chunks, axis=0)
    return jac_map, term1, term2, y_sub


def _batched_mapper_reverse_grad_autograd(
    mapper: Any,
    y_sub: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    y_arr = np.asarray(y_sub, dtype=np.float32)
    if y_arr.ndim != 2:
        raise ValueError(f"y_sub must be 2D, got shape={y_arr.shape}")

    mapper_device = _resolve_mapper_device(mapper, device)
    jac_chunks: List[np.ndarray] = []
    step = max(1, int(batch_size))
    for i in range(0, y_arr.shape[0], step):
        yb = torch.as_tensor(y_arr[i : i + step], dtype=torch.float32, device=mapper_device).clone().detach().requires_grad_(True)
        xb = _mapper_forward_with_graph(mapper, yb, reverse=True)
        if xb.ndim != 2:
            raise ValueError(f"mapper reverse output must be 2D, got shape={tuple(xb.shape)}")
        out_dim = int(xb.shape[1])

        jac_rows: List[torch.Tensor] = []
        for out_idx in range(out_dim):
            grad_row = torch.autograd.grad(
                outputs=xb[:, out_idx].sum(),
                inputs=yb,
                retain_graph=(out_idx < out_dim - 1),
                create_graph=False,
                allow_unused=False,
            )[0]
            jac_rows.append(grad_row.detach())
        jac_b = torch.stack(jac_rows, dim=1)  # (b, k_main, k_sub)
        jac_chunks.append(np.asarray(jac_b.cpu().numpy(), dtype=np.float32))
    return np.concatenate(jac_chunks, axis=0)



def _compute_grn_average_for_coordinates(
    *,
    x_main: np.ndarray,
    t_val: float,
    model: torch.nn.Module,
    mapper: Any,
    main_bundle: PCABundle,
    sub_bundle: PCABundle,
    main_var_idx: Sequence[int],
    sec_var_idx: Sequence[int],
    mapper_batch: int,
    include_hessian_term: bool,
    device: str,
) -> Dict[str, Any]:
    x_main = np.asarray(x_main, dtype=np.float32)
    if x_main.ndim != 2 or x_main.shape[0] == 0:
        raise ValueError(f"x_main must be non-empty 2D array, got shape={x_main.shape}")

    xdot_main = _batched_velocity(model, x_main, float(t_val), device=device)  # (n, k_main)
    jac_z = _batched_velocity_jacobian(model, x_main, float(t_val), device=device)  # (n, k_main, k_main)
    jac_map, term1, term2, y_sub = _batched_main_to_sub_terms_autograd(
        mapper=mapper,
        x_main=x_main,
        xdot_main=xdot_main,
        jac_z=jac_z,
        batch_size=mapper_batch,
        include_hessian_term=include_hessian_term,
        device=device,
    )
    # main->sub: d(ydot)/dx = term1 + term2, with term2 always enabled.
    j_main_to_sub_all = np.asarray(term1 + term2, dtype=np.float32)

    # sub->main uses explicit reverse map Jacobian dx/dy from the same mapper (not pseudo-inverse).
    jac_map_reverse = _batched_mapper_reverse_grad_autograd(
        mapper=mapper,
        y_sub=y_sub,
        batch_size=mapper_batch,
        device=device,
    )  # (n, k_main, k_sub)

    pair_rows_main_to_sub = []
    pair_rows_sub_to_main = []
    pair_rows_gene = []
    main_forward = np.asarray(main_bundle.forward, dtype=np.float32)
    main_recon = np.asarray(main_bundle.reconstruction, dtype=np.float32)
    sub_forward = np.asarray(sub_bundle.forward, dtype=np.float32)
    sub_recon = np.asarray(sub_bundle.reconstruction, dtype=np.float32)
    for cell_i in range(x_main.shape[0]):
        jz = np.asarray(jac_z[cell_i], dtype=np.float32)
        # Gene->gene dynamic coupling in main space.
        j_gene = np.asarray(
            main_forward @ jz.T @ main_recon,
            dtype=np.float32,
        )
        # main->sub coupling (output=sub velocity, input=main state):
        # d(ydot)/dx = (dJ_T/dx * xdot) + J_T * J_xdot
        j_main_to_sub = np.asarray(j_main_to_sub_all[cell_i], dtype=np.float32)
        j_gene_to_prot = np.asarray(
            main_forward @ j_main_to_sub.T @ sub_recon,
            dtype=np.float32,
        )
        # sub->main coupling via chain rule:
        # d(xdot)/dy = d(xdot)/dx * dx/dy, where dx/dy is from reverse mapper Jacobian.
        j_sub_to_main = np.asarray(
            np.einsum("ij,jk->ik", jz, jac_map_reverse[cell_i]),
            dtype=np.float32,
        )  # (k_main, k_sub)
        j_prot_to_gene = np.asarray(
            sub_forward @ j_sub_to_main.T @ main_recon,
            dtype=np.float32,
        )
        pair_rows_main_to_sub.append(j_gene_to_prot[np.ix_(main_var_idx, sec_var_idx)])
        pair_rows_sub_to_main.append(j_prot_to_gene[np.ix_(sec_var_idx, main_var_idx)])
        pair_rows_gene.append(j_gene[np.ix_(main_var_idx, main_var_idx)])

    cross_mean_main_to_sub = np.mean(np.stack(pair_rows_main_to_sub, axis=0), axis=0)
    cross_mean_sub_to_main = np.mean(np.stack(pair_rows_sub_to_main, axis=0), axis=0)
    gene_mean = np.mean(np.stack(pair_rows_gene, axis=0), axis=0)
    try:
        sing_vals = np.linalg.svd(np.asarray(jac_map, dtype=np.float64), compute_uv=False)
        cond_vals = sing_vals[:, 0] / np.maximum(sing_vals[:, -1], 1e-12)
        cond_median = float(np.median(cond_vals))
    except np.linalg.LinAlgError:
        cond_median = float("nan")
    return {
        "cross_mean": np.asarray(cross_mean_main_to_sub, dtype=np.float32),  # backward compatibility alias
        "cross_mean_main_to_sub": np.asarray(cross_mean_main_to_sub, dtype=np.float32),
        "cross_mean_sub_to_main": np.asarray(cross_mean_sub_to_main, dtype=np.float32),
        "gene_mean": np.asarray(gene_mean, dtype=np.float32),
        "term1_strength": float(np.mean(np.abs(term1))),
        "term2_strength": float(np.mean(np.abs(term2))),
        "jacobian_map_condition_median": cond_median,
    }


def run_jacobian_grn_analysis(
    *,
    main_proc: AnnData,
    mapper: Any,
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    main_bundle: PCABundle,
    sub_bundle: PCABundle,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str], List[str]]:
    jac_cfg = cfg.get("jacobian_grn", {})
    max_cells = int(jac_cfg.get("max_cells", 256))
    top_k = int(jac_cfg.get("top_k", 30))
    rng = np.random.default_rng(int(jac_cfg.get("seed", 42)))
    mapper_batch = int(jac_cfg.get("mapper_batch_size", 128))
    include_hessian_term = bool(jac_cfg.get("include_hessian_term", False))
    time_tol = float(jac_cfg.get("time_match_tolerance", 1e-6))
    if time_tol <= 0:
        raise ValueError("jacobian_grn.time_match_tolerance must be > 0")

    main_var_features = _require_str_list(jac_cfg, "main_var_features", context="analysis.jacobian_grn")
    sec_var_features = _require_str_list(jac_cfg, "sec_var_features", context="analysis.jacobian_grn")

    main_lut = {g: i for i, g in enumerate(main_bundle.var_names)}
    sub_lut = {g: i for i, g in enumerate(sub_bundle.var_names)}
    main_var_found = [g for g in main_var_features if g in main_lut]
    sec_var_found = [g for g in sec_var_features if g in sub_lut]
    main_var_missing = sorted(set(main_var_features) - set(main_var_found))
    sec_var_missing = sorted(set(sec_var_features) - set(sec_var_found))
    if not main_var_found or not sec_var_found:
        empty = pd.DataFrame(
            columns=[
                "time",
                "time_requested",
                "source",
                "scope",
                "cell_type",
                "n_cells",
                "main_var_feature",
                "sec_var_feature",
                "source_space",
                "target_space",
                "direction",
                "source_feature",
                "target_feature",
                "weight",
                "rank",
                "method",
            ]
        )
        return empty, {"message": "insufficient features"}, main_var_missing, sec_var_missing

    main_var_idx = [main_lut[g] for g in main_var_found]
    sec_var_idx = [sub_lut[g] for g in sec_var_found]

    requested_times_raw = jac_cfg.get("time_points", None)
    requested_times: List[float] = []
    if requested_times_raw is not None:
        if isinstance(requested_times_raw, (list, tuple, np.ndarray)):
            requested_times = sorted({float(x) for x in requested_times_raw})
        else:
            requested_times = [float(requested_times_raw)]

    mode_raw = str(jac_cfg.get("mode", "real_observed_only")).strip().lower()
    if mode_raw not in {"", "default", "real", "real_only", "real_observed", "real_observed_only"}:
        LOGGER.warning("jacobian_grn.mode=%s is not supported; forcing real_observed_only", mode_raw)
    mode = "real_observed_only"

    time_key = str(cfg.get("time_key", "time_point_processed"))
    obs_time = np.asarray(main_proc.obs[time_key], dtype=float)
    obs_unique = sorted({float(x) for x in obs_time.tolist()})

    if requested_times:
        eval_times = [float(x) for x in requested_times]
    else:
        eval_times = [float(x) for x in obs_unique]

    real_cell_mode_raw = str(jac_cfg.get("real_cell_mode", "all")).strip().lower()
    if real_cell_mode_raw not in {"", "all"}:
        LOGGER.warning("jacobian_grn.real_cell_mode=%s is not supported; forcing all", real_cell_mode_raw)
    real_cell_mode = "all"
    resolved_real_cell_type: Optional[str] = None

    real_latent = np.asarray(main_proc.obsm["X_latent"], dtype=np.float32)

    rows: List[Dict[str, Any]] = []
    gene_rows: List[Dict[str, Any]] = []
    term1_strength_by_time: List[float] = []
    term2_strength_by_time: List[float] = []
    jacobian_map_condition_median_by_time: List[float] = []
    evaluated_points: List[Dict[str, Any]] = []

    for t_req in eval_times:
        real_mask = np.isclose(obs_time, float(t_req), atol=time_tol, rtol=0.0)
        if not bool(np.any(real_mask)):
            continue
        source = "real"
        scope = "all_real_cells"
        used_cell_type = None
        x_pool = np.asarray(real_latent[real_mask], dtype=np.float32)
        t_used = float(t_req)

        if x_pool.shape[0] == 0:
            LOGGER.warning("jacobian_grn: no cells available at time=%.6f source=%s; skipping.", float(t_req), source)
            continue

        if x_pool.shape[0] > max_cells:
            keep = np.sort(rng.choice(np.arange(x_pool.shape[0]), size=max_cells, replace=False))
            x_main = np.asarray(x_pool[keep], dtype=np.float32)
        else:
            x_main = x_pool

        grn_obj = _compute_grn_average_for_coordinates(
            x_main=x_main,
            t_val=float(t_used),
            model=model,
            mapper=mapper,
            main_bundle=main_bundle,
            sub_bundle=sub_bundle,
            main_var_idx=main_var_idx,
            sec_var_idx=sec_var_idx,
            mapper_batch=mapper_batch,
            include_hessian_term=include_hessian_term,
            device=device,
        )
        cross_mean_main_to_sub = np.asarray(grn_obj["cross_mean_main_to_sub"], dtype=np.float32)
        cross_mean_sub_to_main = np.asarray(grn_obj["cross_mean_sub_to_main"], dtype=np.float32)
        gene_mean = np.asarray(grn_obj["gene_mean"], dtype=np.float32)
        term1_strength_by_time.append(float(grn_obj["term1_strength"]))
        term2_strength_by_time.append(float(grn_obj["term2_strength"]))
        jacobian_map_condition_median_by_time.append(float(grn_obj["jacobian_map_condition_median"]))

        evaluated_points.append(
            {
                "time_requested": float(t_req),
                "time_used": float(t_used),
                "source": str(source),
                "scope": str(scope),
                "cell_type": (None if used_cell_type is None else str(used_cell_type)),
                "n_cells": int(x_main.shape[0]),
            }
        )

        pair_flat_main_to_sub: List[Tuple[str, str, float]] = []
        for r_i, r_name in enumerate(main_var_found):
            for p_i, p_name in enumerate(sec_var_found):
                pair_flat_main_to_sub.append((str(r_name), str(p_name), float(cross_mean_main_to_sub[r_i, p_i])))
        pair_flat_main_to_sub.sort(key=lambda x: abs(x[2]), reverse=True)
        for rank, (r_name, p_name, w) in enumerate(pair_flat_main_to_sub[:top_k], start=1):
            rows.append(
                {
                    "time": float(t_used),
                    "time_requested": float(t_req),
                    "source": str(source),
                    "scope": str(scope),
                    "cell_type": (None if used_cell_type is None else str(used_cell_type)),
                    "n_cells": int(x_main.shape[0]),
                    "main_var_feature": str(r_name),
                    "sec_var_feature": str(p_name),
                    "source_space": "main",
                    "target_space": "sub",
                    "direction": "main_to_sub",
                    "source_feature": str(r_name),
                    "target_feature": str(p_name),
                    "weight": float(w),
                    "rank": int(rank),
                    "method": "timepoint_jacobian_pca_projection",
                }
            )

        pair_flat_sub_to_main: List[Tuple[str, str, float]] = []
        for s_i, s_name in enumerate(sec_var_found):
            for m_i, m_name in enumerate(main_var_found):
                pair_flat_sub_to_main.append((str(s_name), str(m_name), float(cross_mean_sub_to_main[s_i, m_i])))
        pair_flat_sub_to_main.sort(key=lambda x: abs(x[2]), reverse=True)
        for rank, (s_name, m_name, w) in enumerate(pair_flat_sub_to_main[:top_k], start=1):
            rows.append(
                {
                    "time": float(t_used),
                    "time_requested": float(t_req),
                    "source": str(source),
                    "scope": str(scope),
                    "cell_type": (None if used_cell_type is None else str(used_cell_type)),
                    "n_cells": int(x_main.shape[0]),
                    # Keep legacy columns for compatibility; use direction/source/target to disambiguate.
                    "main_var_feature": str(m_name),
                    "sec_var_feature": str(s_name),
                    "source_space": "sub",
                    "target_space": "main",
                    "direction": "sub_to_main",
                    "source_feature": str(s_name),
                    "target_feature": str(m_name),
                    "weight": float(w),
                    "rank": int(rank),
                    "method": "timepoint_jacobian_pca_projection_reverse_mapper",
                }
            )

        gene_flat: List[Tuple[str, str, float]] = []
        for src_i, src_name in enumerate(main_var_found):
            for dst_i, dst_name in enumerate(main_var_found):
                gene_flat.append((str(src_name), str(dst_name), float(gene_mean[src_i, dst_i])))
        gene_flat.sort(key=lambda x: abs(x[2]), reverse=True)
        for rank, (src_name, dst_name, w) in enumerate(gene_flat[:top_k], start=1):
            gene_rows.append(
                {
                    "time": float(t_used),
                    "time_requested": float(t_req),
                    "source": str(source),
                    "scope": str(scope),
                    "cell_type": (None if used_cell_type is None else str(used_cell_type)),
                    "n_cells": int(x_main.shape[0]),
                    "source_gene": str(src_name),
                    "target_gene": str(dst_name),
                    "weight": float(w),
                    "rank": int(rank),
                }
            )

    edges_df = pd.DataFrame(rows)
    gene_df = pd.DataFrame(gene_rows)
    if edges_df.empty:
        summary = {
            "top_pairs": [],
            "top_pairs_main_to_sub": [],
            "top_pairs_sub_to_main": [],
            "time_points": [float(x) for x in eval_times],
            "n_evaluated_time_points": 0,
            "evaluated_points": [],
            "mode": mode,
            "real_cell_mode": real_cell_mode,
            "real_cell_type": (None if resolved_real_cell_type is None else str(resolved_real_cell_type)),
            "method": "timepoint_jacobian_pca_projection",
            "include_hessian_term": include_hessian_term,
            "sub_to_main_mapping": "reverse_mapper_jacobian",
            "term1_strength_by_time": term1_strength_by_time,
            "term2_strength_by_time": term2_strength_by_time,
            "jacobian_map_condition_median_by_time": jacobian_map_condition_median_by_time,
        }
    else:
        edges_main_to_sub = edges_df[edges_df["direction"].astype(str) == "main_to_sub"].copy()
        edges_sub_to_main = edges_df[edges_df["direction"].astype(str) == "sub_to_main"].copy()

        agg_main_to_sub = (
            edges_main_to_sub.groupby(["main_var_feature", "sec_var_feature"], as_index=False)["weight"]
            .mean()
            .assign(abs_weight=lambda d: d["weight"].abs())
            .sort_values("abs_weight", ascending=False)
        ) if not edges_main_to_sub.empty else pd.DataFrame(columns=["main_var_feature", "sec_var_feature", "weight", "abs_weight"])

        agg_sub_to_main = (
            edges_sub_to_main.groupby(["source_feature", "target_feature"], as_index=False)["weight"]
            .mean()
            .assign(abs_weight=lambda d: d["weight"].abs())
            .sort_values("abs_weight", ascending=False)
        ) if not edges_sub_to_main.empty else pd.DataFrame(columns=["source_feature", "target_feature", "weight", "abs_weight"])
        if gene_df.empty:
            top_gene_pairs: List[Dict[str, Any]] = []
        else:
            top_gene_pairs = (
                gene_df.groupby(["source_gene", "target_gene"], as_index=False)["weight"]
                .mean()
                .assign(abs_weight=lambda d: d["weight"].abs())
                .sort_values("abs_weight", ascending=False)
                .head(top_k)
                .to_dict(orient="records")
            )
        summary = {
            # Keep top_pairs backward-compatible as the main->sub direction.
            "top_pairs": agg_main_to_sub.head(top_k).to_dict(orient="records"),
            "top_pairs_main_to_sub": agg_main_to_sub.head(top_k).to_dict(orient="records"),
            "top_pairs_sub_to_main": agg_sub_to_main.head(top_k).to_dict(orient="records"),
            "top_gene_pairs": top_gene_pairs,
            "time_points": sorted([float(x) for x in edges_df["time"].unique().tolist()]),
            "n_evaluated_time_points": int(edges_df[["source", "time"]].drop_duplicates().shape[0]),
            "evaluated_points": evaluated_points,
            "n_pairs": int(len(edges_df)),
            "n_pairs_main_to_sub": int(len(edges_main_to_sub)),
            "n_pairs_sub_to_main": int(len(edges_sub_to_main)),
            "mode": mode,
            "real_cell_mode": real_cell_mode,
            "real_cell_type": (None if resolved_real_cell_type is None else str(resolved_real_cell_type)),
            "method": "timepoint_jacobian_pca_projection",
            "include_hessian_term": include_hessian_term,
            "sub_to_main_mapping": "reverse_mapper_jacobian",
            "term1_strength_by_time": term1_strength_by_time,
            "term2_strength_by_time": term2_strength_by_time,
            "jacobian_map_condition_median_by_time": jacobian_map_condition_median_by_time,
        }
    return edges_df, summary, main_var_missing, sec_var_missing


def _align_proba_to_classes(proba: np.ndarray, classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    for i, c in enumerate(target_classes):
        if c in idx:
            aligned[:, i] = proba[:, idx[c]]
    return aligned


def _fate_distribution(
    classifier: Any,
    X: np.ndarray,
    target_classes: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    renormalize_subset: bool = False,
) -> np.ndarray:
    aligned, _ = _predict_proba_for_classes(
        classifier,
        X,
        target_classes=target_classes,
        renormalize_subset=renormalize_subset,
    )
    return _weighted_row_mean(aligned, weights)


def _resolve_target_cell_type(classes: Sequence[str], target_hint: Optional[str]) -> str:
    cls = [str(x) for x in classes]
    if target_hint:
        for c in cls:
            if c.lower() == str(target_hint).lower():
                return c
        for c in cls:
            if str(target_hint).lower() in c.lower():
                return c
    for key in ["mk", "mpc", "megakary"]:
        for c in cls:
            if key in c.lower():
                return c
    return cls[0]


def _normalize_perturbation_groups(
    *,
    targets: Sequence[str],
    target_groups_raw: Any,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []

    def _mk_group(genes_raw: Any, name_raw: Optional[str] = None) -> Optional[Dict[str, Any]]:
        genes = _normalize_label_list(genes_raw)
        if not genes:
            return None
        name = str(name_raw).strip() if name_raw is not None else ""
        if not name:
            name = "+".join(genes)
        return {"name": name, "genes": genes}

    if isinstance(target_groups_raw, (list, tuple)):
        for item in target_groups_raw:
            if isinstance(item, dict):
                group = _mk_group(item.get("genes", []), item.get("name"))
            else:
                group = _mk_group(item)
            if group is not None:
                groups.append(group)
    elif isinstance(target_groups_raw, dict):
        group = _mk_group(target_groups_raw.get("genes", []), target_groups_raw.get("name"))
        if group is not None:
            groups.append(group)

    if not groups:
        for gene in _normalize_label_list(targets):
            groups.append({"name": gene, "genes": [gene]})

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for g in groups:
        key = str(g["name"]).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"name": str(g["name"]).strip(), "genes": [str(x) for x in g["genes"]]})
    return out


def run_multispace_perturbation_analysis(
    *,
    inference_output: Dict[str, Any],
    classifier_bundle: DenseTimeClassifierBundle,
    main_bundle: PCABundle,
    perturb_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    model = inference_output["model"]
    dyn_adata = inference_output["dynamic_adata"]
    mapper = inference_output["mapper"]
    main_proc: AnnData = inference_output["main_processed"]

    time_grid = np.asarray(inference_output["time_grid"], dtype=float)
    main_base = np.asarray(inference_output["main_latent"], dtype=np.float32)
    sub_base = np.asarray(inference_output["sub_latent"], dtype=np.float32)
    weights_base = np.asarray(inference_output["weights"], dtype=np.float32)
    if weights_base.shape != main_base.shape[:2]:
        raise ValueError(
            f"inference_output['weights'] shape {weights_base.shape} mismatches main_latent shape {main_base.shape}"
        )

    if "init_indices" not in inference_output:
        raise KeyError(
            "inference_output missing init_indices; perturbation must reuse _select_initial_indices outputs."
        )
    init_indices = np.asarray(inference_output["init_indices"], dtype=int)
    if len(init_indices) != main_base.shape[1]:
        raise ValueError(
            f"init_indices length {len(init_indices)} != trajectory cell count {main_base.shape[1]}"
        )
    if init_indices.size == 0:
        raise ValueError("init_indices is empty; cannot run perturbation.")
    if np.min(init_indices) < 0:
        raise ValueError(f"init_indices contains negative values: min={int(np.min(init_indices))}")
    X_main_latent = np.asarray(main_proc.obsm["X_latent"], dtype=np.float32)
    if np.max(init_indices) >= X_main_latent.shape[0]:
        raise ValueError(
            f"init_indices out of range for main_processed.obsm['X_latent']: max={int(np.max(init_indices))}, "
            f"n_obs={X_main_latent.shape[0]}"
        )
    # Strictly use the sampled initial cells from _select_initial_indices:
    # latent -> original feature space reconstruction -> perturb -> project back to latent.
    x0 = np.asarray(X_main_latent[init_indices], dtype=np.float32)
    x0_base_feat = _reconstruct_latent_to_feature(x0, main_bundle).astype(np.float32, copy=True)

    dt = float(perturb_cfg.get("dt", 0.05))
    device = str(perturb_cfg.get("device", "cpu"))

    targets = _normalize_str_list(perturb_cfg.get("targets", []))
    target_groups = _normalize_perturbation_groups(
        targets=targets,
        target_groups_raw=perturb_cfg.get("target_groups", []),
    )
    if not target_groups:
        raise ValueError("analysis.perturbation requires non-empty targets or target_groups in config")
    z_scores = [float(x) for x in perturb_cfg.get("z_scores", [-1.0, 1.0])]
    z_scores = sorted({float(x) for x in z_scores})
    missing_target_policy = str(perturb_cfg.get("missing_target_policy", "zero")).strip().lower()
    if missing_target_policy not in {"zero", "skip"}:
        LOGGER.warning("invalid perturbation.missing_target_policy=%s, fallback to zero", missing_target_policy)
        missing_target_policy = "zero"
    feat_lut = {g: i for i, g in enumerate(main_bundle.var_names)}
    feat_std = np.asarray(np.std(x0_base_feat, axis=0), dtype=np.float32)

    classes = np.asarray(classifier_bundle.classes)
    classifier_space = str(classifier_bundle.train_space)
    target_cell_type_hint = str(perturb_cfg.get("target_cell_type", "")).strip()
    target_cell_type = _resolve_target_cell_type(classes, target_cell_type_hint)
    baseline_input = _build_classifier_space_latent(
        train_space=classifier_space,
        main_latent=main_base[-1],
        sub_latent=sub_base[-1],
        mapper=mapper,
        map_batch=int(classifier_bundle.map_batch),
    )
    baseline_dist = _fate_distribution(
        classifier_bundle.classifier,
        baseline_input,
        classes,
        weights=weights_base[-1],
        renormalize_subset=False,
    )

    def _fate_distribution_by_time(
        *,
        main_latent_series: np.ndarray,
        sub_latent_series: np.ndarray,
        weights_series: np.ndarray,
    ) -> np.ndarray:
        latent_series = _build_classifier_space_latent(
            train_space=classifier_space,
            main_latent=main_latent_series,
            sub_latent=sub_latent_series,
            mapper=mapper,
            map_batch=int(classifier_bundle.map_batch),
        )
        if latent_series.ndim != 3:
            raise ValueError(f"classifier latent series must be 3D, got shape={latent_series.shape}")
        out: List[np.ndarray] = []
        for t_idx in range(latent_series.shape[0]):
            dist_t = _fate_distribution(
                classifier_bundle.classifier,
                latent_series[t_idx],
                classes,
                weights=np.asarray(weights_series[t_idx], dtype=np.float32),
                renormalize_subset=False,
            )
            out.append(np.asarray(dist_t, dtype=float))
        return np.asarray(out, dtype=float)

    baseline_dist_by_time = _fate_distribution_by_time(
        main_latent_series=main_base,
        sub_latent_series=sub_base,
        weights_series=weights_base,
    )

    fate_rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []
    focus_curve_rows: List[Dict[str, Any]] = []
    perturb_log: List[Dict[str, Any]] = []

    sweep_space = classifier_space
    sweep_cell_types_req = _normalize_label_list(perturb_cfg.get("sweep_cell_types", []))
    sweep_classes_all = np.asarray(classes)
    sweep_cell_types, sweep_cell_types_missing = _match_requested_labels(
        sweep_cell_types_req,
        [str(x) for x in sweep_classes_all],
    )
    if sweep_cell_types_req and not sweep_cell_types:
        LOGGER.warning(
            "No perturbation.sweep_cell_types matched for sweep_space=%s. Requested=%s, available=%s",
            sweep_space,
            sweep_cell_types_req,
            [str(x) for x in sweep_classes_all],
        )
    sweep_active_set = {x.lower() for x in (sweep_cell_types if sweep_cell_types else [str(x) for x in sweep_classes_all])}

    focus_curve_cell_types_req = _normalize_label_list(
        perturb_cfg.get("focus_curve_cell_types", ["EryP", "MasP", "MkP", "MoP", "NeuP"])
    )
    focus_curve_cell_types, focus_curve_cell_types_missing = _match_requested_labels(
        focus_curve_cell_types_req,
        [str(x) for x in sweep_classes_all],
    )
    if focus_curve_cell_types_req and not focus_curve_cell_types:
        LOGGER.warning(
            "No perturbation.focus_curve_cell_types matched. Requested=%s, available=%s",
            focus_curve_cell_types_req,
            [str(x) for x in sweep_classes_all],
        )
    if focus_curve_cell_types:
        focus_curve_active_types = focus_curve_cell_types
    elif sweep_cell_types:
        focus_curve_active_types = sweep_cell_types
    else:
        focus_curve_active_types = [str(x) for x in sweep_classes_all]
    focus_curve_active_set = {x.lower() for x in focus_curve_active_types}

    focus_curve_scores_req = sorted(
        {
            float(x)
            for x in perturb_cfg.get("focus_curve_scores", [-8.0, 0.0, 8.0])
        }
    )
    focus_curve_scores_used: List[float] = []
    focus_curve_scores_missing: List[float] = []
    for req in focus_curve_scores_req:
        hit = None
        for z in z_scores:
            if abs(float(z) - float(req)) <= 1e-8:
                hit = float(z)
                break
        if hit is None:
            focus_curve_scores_missing.append(float(req))
            continue
        if not any(abs(hit - x) <= 1e-8 for x in focus_curve_scores_used):
            focus_curve_scores_used.append(float(hit))
    focus_curve_scores_used = sorted(focus_curve_scores_used)
    if focus_curve_scores_req and not focus_curve_scores_used:
        LOGGER.warning(
            "No perturbation.focus_curve_scores matched z_scores. requested=%s, z_scores=%s",
            focus_curve_scores_req,
            z_scores,
        )
    focus_curve_score_active = (
        set([float(x) for x in focus_curve_scores_used])
        if focus_curve_scores_used
        else set([float(x) for x in z_scores])
    )

    def _append_sweep_rows(
        *,
        group_name: str,
        group_genes: Sequence[str],
        z: float,
        baseline_prob: np.ndarray,
        pert_dist: np.ndarray,
        classes: np.ndarray,
    ) -> None:
        for c, base_p, pert_p in zip(classes, baseline_prob, pert_dist):
            base_val = float(base_p)
            pert_val = float(pert_p)
            delta = pert_val - base_val
            sweep_rows.append(
                {
                    "target_group": str(group_name),
                    "target_genes": "|".join([str(x) for x in group_genes]),
                    "n_genes": int(len(group_genes)),
                    "z_score": float(z),
                    "cell_type": str(c),
                    "baseline_prob": base_val,
                    "perturbed_prob": pert_val,
                    "delta_prob": float(delta),
                    "delta_rate": float(delta / max(abs(base_val), 1e-8)),
                    "space": classifier_space,
                    "sweep_space": sweep_space,
                    "is_selected_cell_type": bool(str(c).lower() in sweep_active_set),
                }
            )

    def _append_focus_curve_rows(
        *,
        group_name: str,
        group_genes: Sequence[str],
        z: float,
        baseline_time_dist: np.ndarray,
        pert_time_dist: np.ndarray,
        status: str,
    ) -> None:
        if baseline_time_dist.shape != pert_time_dist.shape:
            raise ValueError(
                f"timecourse dist shape mismatch: baseline={baseline_time_dist.shape}, perturbed={pert_time_dist.shape}"
            )
        if baseline_time_dist.ndim != 2:
            raise ValueError(f"timecourse dist must be 2D, got shape={baseline_time_dist.shape}")

        for t_idx, t in enumerate(time_grid):
            for c_idx, c in enumerate(classes):
                base_val = float(baseline_time_dist[t_idx, c_idx])
                # By design, t=0 is always copied from the unperturbed baseline.
                if t_idx == 0:
                    pert_val = float(baseline_time_dist[t_idx, c_idx])
                else:
                    pert_val = float(pert_time_dist[t_idx, c_idx])
                if str(c).lower() not in focus_curve_active_set:
                    continue
                delta = pert_val - base_val
                focus_curve_rows.append(
                    {
                        "target_group": str(group_name),
                        "target_genes": "|".join([str(x) for x in group_genes]),
                        "n_genes": int(len(group_genes)),
                        "z_score": float(z),
                        "time": float(t),
                        "cell_type": str(c),
                        "baseline_prob": float(base_val),
                        "perturbed_prob": float(pert_val),
                        "delta_prob": float(delta),
                        "delta_rate": float(delta / max(abs(base_val), 1e-8)),
                        "space": classifier_space,
                        "sweep_space": sweep_space,
                        "is_selected_cell_type": bool(str(c).lower() in focus_curve_active_set),
                        "status": str(status),
                    }
                )

    def _append_zero_rows(
        *,
        group_name: str,
        group_genes: Sequence[str],
        gene_stats: Sequence[Dict[str, Any]],
        z: float,
        status: str,
    ) -> None:
        tag = f"{group_name}@z{z:g}"
        target_str = str(group_genes[0]) if len(group_genes) == 1 else str(group_name)
        gene_index_repr = int(gene_stats[0]["gene_index"]) if len(gene_stats) == 1 else -1
        gene_std_repr = float(gene_stats[0]["feature_std"]) if len(gene_stats) == 1 else float("nan")
        perturb_log.append(
            {
                "target_group": str(group_name),
                "target_genes": [str(x) for x in group_genes],
                "status": status,
                "z_score": float(z),
                "gene_stats": [dict(x) for x in gene_stats],
            }
        )
        for c, base_p in zip(classes, baseline_dist):
            fate_rows.append(
                {
                    "target": target_str,
                    "target_group": str(group_name),
                    "target_genes": "|".join([str(x) for x in group_genes]),
                    "n_genes": int(len(group_genes)),
                    "gene_index": int(gene_index_repr),
                    "feature_std": float(gene_std_repr),
                    "delta_feature": 0.0,
                    "z_score": float(z),
                    "tag": tag,
                    "cell_type": str(c),
                    "baseline_prob": float(base_p),
                    "perturbed_prob": float(base_p),
                    "delta_prob": 0.0,
                    "space": classifier_space,
                }
            )
        _append_sweep_rows(
            group_name=group_name,
            group_genes=group_genes,
            z=z,
            baseline_prob=baseline_dist,
            pert_dist=baseline_dist,
            classes=classes,
        )
        if any(abs(float(z) - float(x)) <= 1e-8 for x in focus_curve_score_active):
            _append_focus_curve_rows(
                group_name=group_name,
                group_genes=group_genes,
                z=z,
                baseline_time_dist=baseline_dist_by_time,
                pert_time_dist=baseline_dist_by_time,
                status=status,
            )

    for group in target_groups:
        group_name = str(group["name"])
        group_genes = [str(x) for x in group["genes"]]
        gene_stats: List[Dict[str, Any]] = []
        valid_genes: List[Dict[str, Any]] = []
        for gene in group_genes:
            if gene not in feat_lut:
                gene_stats.append(
                    {
                        "gene": gene,
                        "status": "missing_in_var_names",
                        "gene_index": -1,
                        "feature_std": 0.0,
                    }
                )
                continue
            gene_idx = int(feat_lut[gene])
            gene_std = float(feat_std[gene_idx])
            if gene_std < 1e-8:
                gene_stats.append(
                    {
                        "gene": gene,
                        "status": "near_zero_std",
                        "gene_index": gene_idx,
                        "feature_std": gene_std,
                    }
                )
                continue
            item = {
                "gene": gene,
                "status": "ok",
                "gene_index": gene_idx,
                "feature_std": gene_std,
            }
            gene_stats.append(item)
            valid_genes.append(item)

        if not valid_genes:
            if missing_target_policy == "skip":
                perturb_log.append(
                    {
                        "target_group": group_name,
                        "target_genes": group_genes,
                        "status": "skip_no_valid_genes",
                        "gene_stats": gene_stats,
                    }
                )
                continue
            for z in z_scores:
                _append_zero_rows(
                    group_name=group_name,
                    group_genes=group_genes,
                    gene_stats=gene_stats,
                    z=z,
                    status="zero_fallback_no_valid_genes",
                )
            continue

        for z in z_scores:
            x0_pert_feat = x0_base_feat.copy()
            per_gene_delta: List[Dict[str, Any]] = []
            for item in valid_genes:
                delta_feature = float(z) * float(item["feature_std"])
                idx = int(item["gene_index"])
                x0_pert_feat[:, idx] = x0_pert_feat[:, idx] + delta_feature
                per_gene_delta.append(
                    {
                        "gene": str(item["gene"]),
                        "gene_index": idx,
                        "feature_std": float(item["feature_std"]),
                        "delta_feature": float(delta_feature),
                    }
                )

            z0_pert = _project_feature_to_latent(x0_pert_feat, main_bundle)
            traj_main = _build_ode_trajectory(model, z0_pert, time_grid, adata=dyn_adata, dt=dt, device=device)
            traj_sub = _map_to_secondary(mapper, traj_main["main_latent"])
            pert_dist_by_time = _fate_distribution_by_time(
                main_latent_series=np.asarray(traj_main["main_latent"], dtype=np.float32),
                sub_latent_series=np.asarray(traj_sub["sub_latent"], dtype=np.float32),
                weights_series=np.asarray(traj_main["weights_abs"], dtype=np.float32),
            )

            pert_input = _build_classifier_space_latent(
                train_space=classifier_space,
                main_latent=traj_main["main_latent"][-1],
                sub_latent=traj_sub["sub_latent"][-1],
                mapper=mapper,
                map_batch=int(classifier_bundle.map_batch),
            )
            pert_dist = _fate_distribution(
                classifier_bundle.classifier,
                pert_input,
                classes,
                weights=np.asarray(traj_main["weights_abs"][-1], dtype=np.float32),
                renormalize_subset=False,
            )

            tag = f"{group_name}@z{z:g}"
            target_str = str(group_genes[0]) if len(group_genes) == 1 else str(group_name)
            if len(valid_genes) == 1:
                gene_index_repr = int(valid_genes[0]["gene_index"])
                gene_std_repr = float(valid_genes[0]["feature_std"])
                delta_repr = float(per_gene_delta[0]["delta_feature"])
            else:
                gene_index_repr = -1
                gene_std_repr = float("nan")
                delta_repr = float(np.sum([x["delta_feature"] for x in per_gene_delta]))
            perturb_log.append(
                {
                    "target_group": group_name,
                    "target_genes": group_genes,
                    "status": "ok" if len(valid_genes) == len(group_genes) else "ok_partial_genes",
                    "z_score": float(z),
                    "gene_stats": gene_stats,
                    "per_gene_delta": per_gene_delta,
                }
            )
            for c, base_p, pert_p in zip(classes, baseline_dist, pert_dist):
                fate_rows.append(
                    {
                        "target": target_str,
                        "target_group": group_name,
                        "target_genes": "|".join(group_genes),
                        "n_genes": int(len(group_genes)),
                        "gene_index": int(gene_index_repr),
                        "feature_std": float(gene_std_repr),
                        "delta_feature": float(delta_repr),
                        "z_score": float(z),
                        "tag": tag,
                        "cell_type": str(c),
                        "baseline_prob": float(base_p),
                        "perturbed_prob": float(pert_p),
                        "delta_prob": float(pert_p - base_p),
                        "space": classifier_space,
                    }
                )

            _append_sweep_rows(
                group_name=group_name,
                group_genes=group_genes,
                z=z,
                baseline_prob=baseline_dist,
                pert_dist=pert_dist,
                classes=classes,
            )
            if any(abs(float(z) - float(x)) <= 1e-8 for x in focus_curve_score_active):
                _append_focus_curve_rows(
                    group_name=group_name,
                    group_genes=group_genes,
                    z=z,
                    baseline_time_dist=baseline_dist_by_time,
                    pert_time_dist=pert_dist_by_time,
                    status="ok" if len(valid_genes) == len(group_genes) else "ok_partial_genes",
                )

    fate_df = pd.DataFrame(fate_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    focus_curve_df = pd.DataFrame(focus_curve_rows)

    target_shift = fate_df[fate_df["cell_type"] == target_cell_type] if not fate_df.empty else pd.DataFrame()
    summary = {
        "targets": sorted(target_shift["target"].unique().tolist()) if not target_shift.empty else [],
        "target_cell_type": str(target_cell_type),
        "target_groups": sorted([str(x["name"]) for x in target_groups]),
        "max_positive_shift": float(target_shift["delta_prob"].max()) if not target_shift.empty else 0.0,
        "max_negative_shift": float(target_shift["delta_prob"].min()) if not target_shift.empty else 0.0,
        "num_runs": int(target_shift[["target", "z_score"]].drop_duplicates().shape[0]) if not target_shift.empty else 0,
        "max_abs_shift": float(fate_df["delta_prob"].abs().max()) if not fate_df.empty else 0.0,
        "method": "feature_space_perturb_then_project_to_latent",
        "missing_target_policy": missing_target_policy,
        "z_scores": [float(x) for x in z_scores],
        "classifier_space": classifier_space,
        "sweep_space": sweep_space,
        "sweep_cell_types_requested": sweep_cell_types_req,
        "sweep_cell_types_used": sweep_cell_types if sweep_cell_types else [str(x) for x in sweep_classes_all],
        "sweep_cell_types_missing": sweep_cell_types_missing,
        "focus_curve_cell_types_requested": focus_curve_cell_types_req,
        "focus_curve_cell_types_used": focus_curve_active_types,
        "focus_curve_cell_types_missing": focus_curve_cell_types_missing,
        "focus_curve_scores_requested": [float(x) for x in focus_curve_scores_req],
        "focus_curve_scores_used": [float(x) for x in focus_curve_scores_used] if focus_curve_scores_used else [float(x) for x in z_scores],
        "focus_curve_scores_missing": [float(x) for x in focus_curve_scores_missing],
        "logs": perturb_log,
    }
    return fate_df, sweep_df, focus_curve_df, summary


def _build_path_mask(
    *,
    main_latent: np.ndarray,
    time_grid: np.ndarray,
    classifier: Any,
    target_classes: Optional[Sequence[str]],
    target_cell_type: str,
    quantile: float,
    renormalize_subset: bool = False,
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    classes = np.asarray(classifier.classes_) if target_classes is None else np.asarray(list(target_classes))
    target = _resolve_target_cell_type(classes, target_cell_type)
    target_idx = int(np.where(classes == target)[0][0])

    rows: List[Dict[str, Any]] = []
    masks: List[np.ndarray] = []
    for i, t in enumerate(time_grid):
        aligned, _ = _predict_proba_for_classes(
            classifier,
            main_latent[i],
            target_classes=classes,
            renormalize_subset=renormalize_subset,
        )
        score = aligned[:, target_idx]
        thr = float(np.quantile(score, quantile))
        mask = score >= thr
        masks.append(mask)
        for j in range(len(score)):
            rows.append(
                {
                    "time": float(t),
                    "trajectory_index": int(j),
                    "path_score": float(score[j]),
                    "is_path": bool(mask[j]),
                    "target_cell_type": target,
                }
            )

    return pd.DataFrame(rows), masks


def _run_wilcoxon_de(
    *,
    decoded_by_time: List[np.ndarray],
    features: Sequence[str],
    masks: List[np.ndarray],
    time_grid: np.ndarray,
    modality: str,
    fdr_alpha: float,
    min_abs_log2fc: float,
    min_pct: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(time_grid):
        X = decoded_by_time[i]
        mask = masks[i]
        if np.sum(mask) < 3 or np.sum(~mask) < 3:
            continue

        p_vals = []
        stats = []
        log2fcs = []
        pct_path = []
        pct_bg = []
        for j in range(X.shape[1]):
            a = X[mask, j]
            b = X[~mask, j]
            stat, _ = ranksums(a, b)
            p = _two_sided_p_from_z(stat)
            stats.append(float(stat))
            p_vals.append(float(p))
            # Reconstructed features can have negative means; clip to keep log2FC finite.
            m1 = float(max(np.mean(a), 0.0))
            m0 = float(max(np.mean(b), 0.0))
            log2fc = float(np.log2(m1 + 1e-3) - np.log2(m0 + 1e-3))
            log2fcs.append(log2fc)
            pct_path.append(float(np.mean(a > 0)))
            pct_bg.append(float(np.mean(b > 0)))

        p_adj = _bh_adjust(np.asarray(p_vals, dtype=float))
        for j, feat in enumerate(features):
            sig = bool((p_adj[j] < fdr_alpha) and (abs(log2fcs[j]) >= min_abs_log2fc) and (max(pct_path[j], pct_bg[j]) >= min_pct))
            rows.append(
                {
                    "time": float(t),
                    "modality": modality,
                    "feature": str(feat),
                    "stat": float(stats[j]),
                    "p_value": float(p_vals[j]),
                    "p_adj": float(p_adj[j]),
                    "log2fc": float(log2fcs[j]),
                    "pct_path": float(pct_path[j]),
                    "pct_bg": float(pct_bg[j]),
                    "direction": "up_in_path" if log2fcs[j] >= 0 else "down_in_path",
                    "is_significant": sig,
                }
            )

    return pd.DataFrame(rows)


def _run_wilcoxon_de_last_vs_early(
    *,
    decoded_by_time: List[np.ndarray],
    features: Sequence[str],
    masks: List[np.ndarray],
    time_grid: np.ndarray,
    modality: str,
    fdr_alpha: float,
    min_abs_log2fc: float,
    min_pct: float,
    target_times: Sequence[float],
    reference_times: Sequence[float],
) -> pd.DataFrame:
    if len(features) == 0 or len(decoded_by_time) == 0:
        return pd.DataFrame()

    grid = np.asarray(time_grid, dtype=float)
    target_arr = np.asarray([float(x) for x in target_times], dtype=float)
    ref_arr = np.asarray([float(x) for x in reference_times], dtype=float)

    target_idx = [i for i, t in enumerate(grid) if bool(np.any(np.isclose(float(t), target_arr, atol=1e-6)))]
    ref_idx = [i for i, t in enumerate(grid) if bool(np.any(np.isclose(float(t), ref_arr, atol=1e-6)))]

    if not target_idx or not ref_idx:
        return pd.DataFrame()

    target_blocks: List[np.ndarray] = []
    ref_blocks: List[np.ndarray] = []
    for i in target_idx:
        X = decoded_by_time[i]
        mask = np.asarray(masks[i], dtype=bool)
        if int(np.sum(mask)) > 0:
            target_blocks.append(np.asarray(X[mask], dtype=np.float32))
    for i in ref_idx:
        X = decoded_by_time[i]
        mask = np.asarray(masks[i], dtype=bool)
        if int(np.sum(mask)) > 0:
            ref_blocks.append(np.asarray(X[mask], dtype=np.float32))

    if not target_blocks or not ref_blocks:
        return pd.DataFrame()

    target_mat = np.concatenate(target_blocks, axis=0)
    ref_mat = np.concatenate(ref_blocks, axis=0)
    n_target = int(target_mat.shape[0])
    n_reference = int(ref_mat.shape[0])
    if n_target < 3 or n_reference < 3:
        return pd.DataFrame()

    p_vals: List[float] = []
    stats: List[float] = []
    log2fcs: List[float] = []
    pct_target: List[float] = []
    pct_reference: List[float] = []
    for j in range(target_mat.shape[1]):
        a = target_mat[:, j]
        b = ref_mat[:, j]
        stat, _ = ranksums(a, b)
        p = _two_sided_p_from_z(stat)
        stats.append(float(stat))
        p_vals.append(float(p))
        # Reconstructed features can have negative means; clip to keep log2FC finite.
        m1 = float(max(np.mean(a), 0.0))
        m0 = float(max(np.mean(b), 0.0))
        log2fcs.append(float(np.log2(m1 + 1e-3) - np.log2(m0 + 1e-3)))
        pct_target.append(float(np.mean(a > 0)))
        pct_reference.append(float(np.mean(b > 0)))

    p_adj = _bh_adjust(np.asarray(p_vals, dtype=float))
    target_label = ",".join([f"{float(x):g}" for x in sorted(set(target_arr.tolist()))])
    reference_label = ",".join([f"{float(x):g}" for x in sorted(set(ref_arr.tolist()))])
    contrast = f"t{target_label}_vs_t{reference_label}"
    anchor_time = float(np.max(target_arr))

    rows: List[Dict[str, Any]] = []
    for j, feat in enumerate(features):
        sig = bool(
            (p_adj[j] < fdr_alpha)
            and (abs(log2fcs[j]) >= min_abs_log2fc)
            and (max(pct_target[j], pct_reference[j]) >= min_pct)
        )
        rows.append(
            {
                "time": anchor_time,
                "modality": modality,
                "feature": str(feat),
                "stat": float(stats[j]),
                "p_value": float(p_vals[j]),
                "p_adj": float(p_adj[j]),
                "log2fc": float(log2fcs[j]),
                "pct_path": float(pct_target[j]),
                "pct_bg": float(pct_reference[j]),
                "direction": "up_in_path" if log2fcs[j] >= 0 else "down_in_path",
                "is_significant": sig,
                "contrast": contrast,
                "n_target": n_target,
                "n_reference": n_reference,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_key_molecules(
    de_main_var_df: pd.DataFrame,
    de_sec_var_df: pd.DataFrame,
    top_k: int,
    min_time_hits: int,
    prior_targets: Sequence[str],
) -> pd.DataFrame:
    all_df = pd.concat([de_main_var_df, de_sec_var_df], axis=0, ignore_index=True)
    if all_df.empty:
        return pd.DataFrame(columns=["modality", "feature", "direction", "time_hits", "mean_log2fc", "best_p_adj", "stable", "prior_hit"])

    sig = all_df[all_df["is_significant"]].copy()
    if sig.empty:
        return pd.DataFrame(columns=["modality", "feature", "direction", "time_hits", "mean_log2fc", "best_p_adj", "stable", "prior_hit"])

    agg = (
        sig.groupby(["modality", "feature", "direction"], as_index=False)
        .agg(time_hits=("time", "nunique"), mean_log2fc=("log2fc", "mean"), best_p_adj=("p_adj", "min"))
    )
    agg["stable"] = agg["time_hits"] >= int(min_time_hits)
    prior_lut = {str(x).upper() for x in prior_targets}
    agg["prior_hit"] = agg["feature"].map(lambda x: str(x).upper() in prior_lut)
    agg = agg.sort_values(["stable", "time_hits", "best_p_adj", "mean_log2fc"], ascending=[False, False, True, False])
    return agg.head(int(top_k)).reset_index(drop=True)


def run_key_molecule_analysis(
    *,
    path_latent: np.ndarray,
    time_grid: np.ndarray,
    path_classifier: Any,
    path_classifier_classes: np.ndarray,
    path_classifier_space: str,
    main_bundle: PCABundle,
    sub_bundle: PCABundle,
    main_feature_pred: np.ndarray,
    sub_feature_pred: np.ndarray,
    cfg: Dict[str, Any],
    programs: Dict[str, List[str]],
    prior_targets: Sequence[str],
    main_var_markers: Sequence[str],
    sec_var_markers: Sequence[str],
    target_cell_type_hint: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str], List[str]]:
    km_cfg = cfg.get("key_molecule", {})
    quantile = float(km_cfg.get("path_quantile", 0.75))
    fdr_alpha = float(km_cfg.get("fdr_alpha", 0.05))
    min_abs_log2fc = float(km_cfg.get("min_abs_log2fc", 0.25))
    min_pct = float(km_cfg.get("min_pct", 0.1))
    top_k = int(km_cfg.get("top_k", 50))
    min_time_hits = int(km_cfg.get("min_time_hits", 2))
    panel_mode = str(km_cfg.get("panel_mode", "configured")).strip().lower()
    if panel_mode not in {"configured", "full_space"}:
        panel_mode = "configured"
    de_contrast_cfg = km_cfg.get("de_contrast", {})
    if not isinstance(de_contrast_cfg, dict):
        de_contrast_cfg = {}
    de_mode = str(de_contrast_cfg.get("mode", "path_vs_bg_each_time")).strip().lower()

    if panel_mode == "full_space":
        main_var_panel = [str(x) for x in main_bundle.var_names]
        sec_var_panel = [str(x) for x in sub_bundle.var_names]
    else:
        main_var_panel = [
            str(x)
            for x in km_cfg.get(
                "main_var_panel",
                sorted(set(list(main_var_markers) + [g for genes in programs.values() for g in genes] + [str(x) for x in prior_targets])),
            )
        ]
        sec_var_panel = [str(x) for x in km_cfg.get("sec_var_panel", [str(x) for x in sec_var_markers])]

    path_df, path_masks = _build_path_mask(
        main_latent=path_latent,
        time_grid=time_grid,
        classifier=path_classifier,
        target_classes=path_classifier_classes,
        target_cell_type=target_cell_type_hint or "",
        quantile=quantile,
        renormalize_subset=False,
    )
    target_cell_type = str(path_df["target_cell_type"].iloc[0]) if not path_df.empty else _resolve_target_cell_type(path_classifier_classes, target_cell_type_hint)
    
    main_lut = {g: i for i, g in enumerate(main_bundle.var_names)}
    sub_lut = {g: i for i, g in enumerate(sub_bundle.var_names)}
    main_var_found = [g for g in main_var_panel if g in main_lut]
    sec_var_found = [g for g in sec_var_panel if g in sub_lut]
    main_var_missing = sorted(set(main_var_panel) - set(main_var_found))
    sec_var_missing = sorted(set(sec_var_panel) - set(sec_var_found))

    main_var_decoded = []
    if main_var_found:
        main_var_idx = [main_lut[g] for g in main_var_found]
        main_var_decoded = [np.asarray(main_feature_pred[i][:, main_var_idx], dtype=np.float32) for i in range(main_feature_pred.shape[0])]

    sec_var_decoded = []
    if sec_var_found:
        sec_var_idx = [sub_lut[g] for g in sec_var_found]
        sec_var_decoded = [np.asarray(sub_feature_pred[i][:, sec_var_idx], dtype=np.float32) for i in range(sub_feature_pred.shape[0])]

    if de_mode == "last_vs_early":
        target_times = _normalize_float_list(de_contrast_cfg.get("target_times", []))
        if not target_times:
            if "last_time" in de_contrast_cfg:
                target_times = [float(de_contrast_cfg.get("last_time"))]
            else:
                target_times = [float(np.max(np.asarray(time_grid, dtype=float)))]
        reference_times = _normalize_float_list(de_contrast_cfg.get("reference_times", []))
        if not reference_times:
            reference_times = _normalize_float_list(de_contrast_cfg.get("early_times", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    else:
        target_times = []
        reference_times = []

    if main_var_decoded:
        if de_mode == "last_vs_early":
            de_main_var_df = _run_wilcoxon_de_last_vs_early(
                decoded_by_time=main_var_decoded,
                features=main_var_found,
                masks=path_masks,
                time_grid=time_grid,
                modality="main_var",
                fdr_alpha=fdr_alpha,
                min_abs_log2fc=min_abs_log2fc,
                min_pct=min_pct,
                target_times=target_times,
                reference_times=reference_times,
            )
        else:
            de_main_var_df = _run_wilcoxon_de(
                decoded_by_time=main_var_decoded,
                features=main_var_found,
                masks=path_masks,
                time_grid=time_grid,
                modality="main_var",
                fdr_alpha=fdr_alpha,
                min_abs_log2fc=min_abs_log2fc,
                min_pct=min_pct,
            )
    else:
        de_main_var_df = pd.DataFrame(columns=["time", "modality", "feature", "stat", "p_value", "p_adj", "log2fc", "pct_path", "pct_bg", "direction", "is_significant"])

    if sec_var_decoded:
        if de_mode == "last_vs_early":
            de_sec_var_df = _run_wilcoxon_de_last_vs_early(
                decoded_by_time=sec_var_decoded,
                features=sec_var_found,
                masks=path_masks,
                time_grid=time_grid,
                modality="sec_var",
                fdr_alpha=fdr_alpha,
                min_abs_log2fc=min_abs_log2fc,
                min_pct=min_pct,
                target_times=target_times,
                reference_times=reference_times,
            )
        else:
            de_sec_var_df = _run_wilcoxon_de(
                decoded_by_time=sec_var_decoded,
                features=sec_var_found,
                masks=path_masks,
                time_grid=time_grid,
                modality="sec_var",
                fdr_alpha=fdr_alpha,
                min_abs_log2fc=min_abs_log2fc,
                min_pct=min_pct,
            )
    else:
        de_sec_var_df = pd.DataFrame(columns=["time", "modality", "feature", "stat", "p_value", "p_adj", "log2fc", "pct_path", "pct_bg", "direction", "is_significant"])

    union_df = _aggregate_key_molecules(
        de_main_var_df,
        de_sec_var_df,
        top_k=top_k,
        min_time_hits=min_time_hits,
        prior_targets=prior_targets,
    )

    summary = {
        "target_cell_type": target_cell_type,
        "path_classifier_space": str(path_classifier_space),
        "de_mode": de_mode,
        "panel_mode": panel_mode,
        "num_sig_main_vars": int(de_main_var_df[de_main_var_df["is_significant"]].shape[0]) if not de_main_var_df.empty else 0,
        "num_sig_sec_vars": int(de_sec_var_df[de_sec_var_df["is_significant"]].shape[0]) if not de_sec_var_df.empty else 0,
        "num_union_key_molecules": int(len(union_df)),
        "prior_hits": union_df[union_df["prior_hit"]]["feature"].tolist() if not union_df.empty else [],
        "method": "pca_reconstruction_decode",
    }

    return path_df, de_main_var_df, de_sec_var_df, union_df, summary, main_var_missing, sec_var_missing



def analyze_dense_time_predictions(
    *,
    inference_output: Dict[str, Any],
    output_dir: str,
    analysis_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = analysis_cfg or {}
    out_dir = Path(output_dir)
    tables_dir = out_dir / "tables"
    reports_dir = out_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    for stale_table in tables_dir.glob("*.csv"):
        try:
            stale_table.unlink()
        except OSError as exc:
            LOGGER.warning("Failed to remove stale table %s (%s)", stale_table, exc)

    main_proc: AnnData = inference_output["main_processed"]
    sub_proc: AnnData = inference_output["sub_processed"]
    main_latent = np.asarray(inference_output["main_latent"], dtype=np.float32)
    sub_latent = np.asarray(inference_output["sub_latent"], dtype=np.float32)
    pred_weights = np.asarray(inference_output["weights"], dtype=np.float32)
    if pred_weights.shape != main_latent.shape[:2]:
        raise ValueError(
            f"inference_output['weights'] shape {pred_weights.shape} mismatches main_latent shape {main_latent.shape}"
        )
    time_grid = np.asarray(inference_output["time_grid"], dtype=float)

    time_key = str(cfg.get("time_key", "time_point_processed"))
    cell_type_key = str(cfg.get("cell_type_key", "cell_type"))
    decode_mode = str(cfg.get("decode_mode", "pca_projection")).strip().lower()
    analysis_device = str(cfg.get("device", inference_output.get("inference_device", "cpu")))
    if decode_mode != "pca_projection":
        raise ValueError(f"Unsupported decode_mode={decode_mode}. This workflow now supports only pca_projection.")
    main_var_markers = _require_str_list(cfg, "main_var_markers", context="analysis")
    sec_var_markers = _require_str_list(cfg, "sec_var_markers", context="analysis")
    prior_targets = _require_str_list(cfg, "prior_targets", context="analysis")
    programs_raw = cfg.get("programs", {})
    if not isinstance(programs_raw, dict) or len(programs_raw) == 0:
        raise ValueError("analysis.programs must be a non-empty mapping in config")
    programs: Dict[str, List[str]] = {}
    for name, genes in programs_raw.items():
        gene_list = _normalize_str_list(genes)
        if gene_list:
            programs[str(name)] = gene_list
    if len(programs) == 0:
        raise ValueError("analysis.programs must contain at least one non-empty gene list")

    main_bundle = _load_pca_bundle(main_proc, space_name="main_processed")
    sub_bundle = _load_pca_bundle(sub_proc, space_name="sub_processed")
    _validate_pca_bundle(main_bundle, space_name="main_processed", expected_latent_dim=main_latent.shape[2])
    _validate_pca_bundle(sub_bundle, space_name="sub_processed", expected_latent_dim=sub_latent.shape[2])
    programs_sec, programs_sec_missing_requested = _resolve_secondary_programs(
        programs,
        cfg.get("programs_sec", {}),
        sub_bundle.var_names,
    )

    main_feature_pred = _reconstruct_latent_series(main_latent, main_bundle)
    sub_feature_pred = _reconstruct_latent_series(sub_latent, sub_bundle)
    if main_bundle.x_before_pca is not None and int(main_bundle.x_before_pca.shape[0]) == int(main_proc.n_obs):
        main_feature_real = np.asarray(main_bundle.x_before_pca, dtype=np.float32)
    else:
        if main_bundle.x_before_pca is not None:
            LOGGER.warning(
                "main_processed: X_before_pca n_obs=%d mismatches adata.n_obs=%d; fallback to latent reconstruction.",
                int(main_bundle.x_before_pca.shape[0]),
                int(main_proc.n_obs),
            )
        main_feature_real = _reconstruct_latent_to_feature(np.asarray(main_proc.obsm["X_latent"], dtype=np.float32), main_bundle)
    if sub_bundle.x_before_pca is not None and int(sub_bundle.x_before_pca.shape[0]) == int(sub_proc.n_obs):
        sub_feature_real = np.asarray(sub_bundle.x_before_pca, dtype=np.float32)
    else:
        if sub_bundle.x_before_pca is not None:
            LOGGER.warning(
                "sub_processed: X_before_pca n_obs=%d mismatches adata.n_obs=%d; fallback to latent reconstruction.",
                int(sub_bundle.x_before_pca.shape[0]),
                int(sub_proc.n_obs),
            )
        sub_feature_real = _reconstruct_latent_to_feature(np.asarray(sub_proc.obsm["X_latent"], dtype=np.float32), sub_bundle)

    classifier_cfg = _normalize_classifier_cfg(dict(cfg.get("classifier", {})))
    if "train_space" not in classifier_cfg:
        classifier_cfg["train_space"] = "joint"

    classifier_bundle, classifier_eval = build_classifier_bundle(
        main_proc=main_proc,
        sub_proc=sub_proc,
        mapper=inference_output["mapper"],
        cell_type_key=cell_type_key,
        classifier_cfg=classifier_cfg,
        reports_dir=reports_dir,
    )
    classifier_space = str(classifier_bundle.train_space)
    classifier_latent = _build_classifier_space_latent(
        train_space=classifier_space,
        main_latent=main_latent,
        sub_latent=sub_latent,
        mapper=inference_output["mapper"],
        map_batch=int(classifier_bundle.map_batch),
    )

    cell_type_pred_main_df = _build_pred_cell_type_profiles(
        classifier_latent,
        time_grid,
        classifier_bundle.classifier,
        space=classifier_space,
        weights_by_time=pred_weights,
        target_classes=classifier_bundle.classes,
        renormalize_subset=False,
    )
    cell_type_real_main_df = _build_real_cell_type_profiles(main_proc, time_key=time_key, cell_type_key=cell_type_key)
    cell_type_eval_main = evaluate_cell_type_prior_over_time(cell_type_pred_main_df, cell_type_real_main_df)
    primary_classifier = classifier_bundle.classifier
    primary_classes = np.asarray(classifier_bundle.classes)
    primary_latent = classifier_latent

    main_var_real_df_raw, main_var_found, main_var_missing = _extract_columns_from_feature_matrix(
        main_feature_real,
        main_bundle.var_names,
        main_var_markers,
    )
    sec_var_real_df_raw, sec_var_found, sec_var_missing = _extract_columns_from_feature_matrix(
        sub_feature_real,
        sub_bundle.var_names,
        sec_var_markers,
    )

    main_var_pred_df = _feature_matrix_to_marker_profile(
        main_feature_pred,
        main_bundle.var_names,
        main_var_found,
        time_grid,
        label="main_var",
        weights_by_time=pred_weights,
    )
    sec_var_pred_df = _feature_matrix_to_marker_profile(
        sub_feature_pred,
        sub_bundle.var_names,
        sec_var_found,
        time_grid,
        label="sec_var",
        weights_by_time=pred_weights,
    )
    main_var_real_df = _build_real_profiles(
        main_var_real_df_raw[main_var_found] if main_var_found else pd.DataFrame(),
        np.asarray(main_proc.obs[time_key], dtype=float),
        "main_var",
    )
    sec_var_real_df = _build_real_profiles(
        sec_var_real_df_raw[sec_var_found] if sec_var_found else pd.DataFrame(),
        np.asarray(sub_proc.obs[time_key], dtype=float),
        "sec_var",
    )

    program_pred_df, program_real_df, program_missing = _build_program_profiles(
        main_proc=main_proc,
        pred_time_grid=time_grid,
        time_key=time_key,
        programs=programs,
        main_feature_pred=main_feature_pred,
        main_feature_names=main_bundle.var_names,
        main_feature_real=main_feature_real,
        main_feature_real_names=main_bundle.var_names,
        pred_weights=pred_weights,
    )
    mechanism_eval = evaluate_mechanism_prior_over_time(program_pred_df, program_real_df)
    if programs_sec:
        program_sec_pred_df, program_sec_real_df, program_sec_missing = _build_program_profiles(
            main_proc=sub_proc,
            pred_time_grid=time_grid,
            time_key=time_key,
            programs=programs_sec,
            main_feature_pred=sub_feature_pred,
            main_feature_names=sub_bundle.var_names,
            main_feature_real=sub_feature_real,
            main_feature_real_names=sub_bundle.var_names,
            pred_weights=pred_weights,
        )
        program_sec_missing = sorted(set(programs_sec_missing_requested).union(set(program_sec_missing)))
    else:
        program_sec_pred_df = pd.DataFrame(columns=["time", "program", "score", "score_p40", "score_p50", "score_p60", "n_cells", "source"])
        program_sec_real_df = pd.DataFrame(columns=["time", "program", "score", "score_p40", "score_p50", "score_p60", "n_cells", "source"])
        program_sec_missing = list(programs_sec_missing_requested)



    jac_edges_df, jac_summary, jac_main_var_missing, jac_sec_var_missing = run_jacobian_grn_analysis(
        main_proc=main_proc,
        mapper=inference_output["mapper"],
        cfg=cfg,
        model=inference_output["model"],
        main_bundle=main_bundle,
        sub_bundle=sub_bundle,
        device=analysis_device,
    )

    target_cell_type_hint = str(cfg.get("path_definition", {}).get("target_cell_type", "")) if isinstance(cfg.get("path_definition"), dict) else ""
    target_cell_type = _resolve_target_cell_type(primary_classes, target_cell_type_hint)
    perturb_cfg = dict(cfg.get("perturbation", {}))
    if not str(perturb_cfg.get("target_cell_type", "")).strip():
        perturb_cfg["target_cell_type"] = target_cell_type

    perturb_fate_df, perturb_sweep_df, perturb_focus_curve_df, perturb_summary = run_multispace_perturbation_analysis(
        inference_output=inference_output,
        classifier_bundle=classifier_bundle,
        main_bundle=main_bundle,
        perturb_cfg=perturb_cfg,
    )

    key_path_df, de_main_var_df, de_sec_var_df, key_union_df, key_summary, key_main_var_missing, key_sec_var_missing = run_key_molecule_analysis(
        path_latent=primary_latent,
        time_grid=time_grid,
        path_classifier=primary_classifier,
        path_classifier_classes=primary_classes,
        path_classifier_space=classifier_space,
        main_bundle=main_bundle,
        sub_bundle=sub_bundle,
        main_feature_pred=main_feature_pred,
        sub_feature_pred=sub_feature_pred,
        cfg=cfg,
        programs=programs,
        prior_targets=prior_targets,
        main_var_markers=main_var_markers,
        sec_var_markers=sec_var_markers,
        target_cell_type_hint=target_cell_type,
    )


    _save_table(main_var_pred_df, tables_dir / "main_var_marker_pred.csv")
    _save_table(sec_var_pred_df, tables_dir / "sec_var_marker_pred.csv")
    _save_table(main_var_real_df, tables_dir / "main_var_marker_real.csv")
    _save_table(sec_var_real_df, tables_dir / "sec_var_marker_real.csv")
    _save_table(cell_type_pred_main_df, tables_dir / "cell_type_pred_main.csv")
    _save_table(cell_type_real_main_df, tables_dir / "cell_type_real_main.csv")
    _save_table(program_pred_df, tables_dir / "program_pred.csv")
    _save_table(program_real_df, tables_dir / "program_real.csv")
    _save_table(program_sec_pred_df, tables_dir / "program_sec_pred.csv")
    _save_table(program_sec_real_df, tables_dir / "program_sec_real.csv")
    _save_table(jac_edges_df, tables_dir / "jacobian_grn_edges.csv")
    perturb_groups_for_name = [str(x) for x in perturb_summary.get("target_groups", [])]
    perturb_cell_types_for_name = [str(x) for x in perturb_summary.get("sweep_cell_types_used", [])]
    perturb_name_suffix = _make_perturbation_suffix(perturb_groups_for_name, perturb_cell_types_for_name)
    perturb_sweep_dynamic_path = tables_dir / f"perturbation_zscore_sweep__{perturb_name_suffix}.csv"
    perturb_focus_curve_dynamic_path = tables_dir / f"perturbation_timecurve_focus__{perturb_name_suffix}.csv"

    _save_table(perturb_fate_df, tables_dir / "perturbation_fate_shift_main.csv")
    _save_table(perturb_sweep_df, perturb_sweep_dynamic_path)
    _save_table(perturb_focus_curve_df, perturb_focus_curve_dynamic_path)
    _save_table(key_path_df, tables_dir / "key_path_cells.csv")
    _save_table(de_main_var_df, tables_dir / "de_main_var_path_wilcoxon.csv")
    _save_table(de_sec_var_df, tables_dir / "de_sec_var_path_wilcoxon.csv")
    _save_table(key_union_df, tables_dir / "key_molecules_union.csv")

    jacobian_grn_eval = jac_summary
    perturbation_eval = perturb_summary
    key_molecule_eval = key_summary

    analysis_summary = {
        "schema_version": "dense_time_v2",
        "time_grid": [float(x) for x in time_grid],
        "decode_mode": decode_mode,
        "analysis_device": analysis_device,
        "space_orientation": {
            "primary_domain": int(inference_output.get("primary_domain", 1)),
            "secondary_domain": int(inference_output.get("secondary_domain", 2)),
            "domain1_processed_path": str(inference_output.get("domain1_processed_path", "")),
            "domain2_processed_path": str(inference_output.get("domain2_processed_path", "")),
            "primary_processed_path": str(inference_output.get("primary_processed_path", "")),
            "secondary_processed_path": str(inference_output.get("secondary_processed_path", "")),
        },
        "tables_schema": {
            "cell_type_pred_main": [
                "time",
                "cell_type",
                "prob",
                "prob_p40",
                "prob_p50",
                "prob_p60",
                "n_cells",
                "weight_sum",
                "space",
                "source",
            ],
            "marker_profile": [
                "time",
                "marker",
                "mean_expr",
                "expr_p40",
                "expr_p50",
                "expr_p60",
                "n_cells",
                "space",
                "source",
            ],
            "program_profile": [
                "time",
                "program",
                "score",
                "score_p40",
                "score_p50",
                "score_p60",
                "n_cells",
                "source",
            ],
        },
        "pca_bundle_validation": {
            "main": {
                "latent_dim": int(main_bundle.reconstruction.shape[0]),
                "feature_dim": int(main_bundle.reconstruction.shape[1]),
                "num_var_names": int(len(main_bundle.var_names)),
                "has_x_before_pca": bool(main_bundle.x_before_pca is not None),
                "original_info_key": str(main_bundle.original_info_key),
            },
            "sub": {
                "latent_dim": int(sub_bundle.reconstruction.shape[0]),
                "feature_dim": int(sub_bundle.reconstruction.shape[1]),
                "num_var_names": int(len(sub_bundle.var_names)),
                "has_x_before_pca": bool(sub_bundle.x_before_pca is not None),
                "original_info_key": str(sub_bundle.original_info_key),
            },
        },
        "missing_markers": {
            "main_var": main_var_missing,
            "sec_var": sec_var_missing,
            "program_missing": program_missing,
            "program_sec_missing": program_sec_missing,
            "jacobian_main_var": jac_main_var_missing,
            "jacobian_sec_var": jac_sec_var_missing,
            "key_main_var_missing": key_main_var_missing,
            "key_sec_var_missing": key_sec_var_missing,
        },
        "classifier_eval": classifier_eval,
        "cell_type_eval": {
            "train_space": classifier_space,
            "metrics": cell_type_eval_main,
        },
        "mechanism_eval": mechanism_eval,
        "jacobian_grn_eval": jacobian_grn_eval,
        "perturbation_eval": perturbation_eval,
        "key_molecule_eval": key_molecule_eval,
        "qc_metrics": inference_output.get("qc_metrics", {}),
    }

    _write_json(reports_dir / "analysis_summary.json", analysis_summary)
    _write_json(reports_dir / "classifier_eval.json", classifier_eval)
    _write_json(reports_dir / "jacobian_grn_eval.json", jacobian_grn_eval)
    _write_json(reports_dir / "perturbation_eval.json", perturbation_eval)
    _write_json(reports_dir / "key_molecule_eval.json", key_molecule_eval)

    return {
        "tables_dir": str(tables_dir),
        "reports_dir": str(reports_dir),
        "analysis_summary": analysis_summary,
        "table_paths": {
            "main_var_marker_pred": str(tables_dir / "main_var_marker_pred.csv"),
            "sec_var_marker_pred": str(tables_dir / "sec_var_marker_pred.csv"),
            "cell_type_pred_main": str(tables_dir / "cell_type_pred_main.csv"),
            "program_pred": str(tables_dir / "program_pred.csv"),
            "program_sec_pred": str(tables_dir / "program_sec_pred.csv"),
            "jacobian_grn_edges": str(tables_dir / "jacobian_grn_edges.csv"),
            "perturbation_fate_main": str(tables_dir / "perturbation_fate_shift_main.csv"),
            "perturbation_zscore_sweep": str(perturb_sweep_dynamic_path),
            "perturbation_timecurve_focus": str(perturb_focus_curve_dynamic_path),
            "de_main_var_path_wilcoxon": str(tables_dir / "de_main_var_path_wilcoxon.csv"),
            "de_sec_var_path_wilcoxon": str(tables_dir / "de_sec_var_path_wilcoxon.csv"),
            "key_molecules_union": str(tables_dir / "key_molecules_union.csv"),
        },
    }


def run_dense_time_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config.get("paths", {})
    inference_cfg = config.get("inference", {})
    analysis_cfg = config.get("analysis", {})
    output_cfg = config.get("output", {})

    output_root = Path(output_cfg.get("root_dir", "results/31800_dense_time")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = _resolve_runtime_device(str(inference_cfg.get("device", "cpu")))
    seed = int(inference_cfg.get("seed", 42))
    dt = float(inference_cfg.get("dt", 0.05))
    time_key = str(analysis_cfg.get("time_key", "time_point_processed"))
    cell_type_key = str(analysis_cfg.get("cell_type_key", "cell_type"))

    main_processed = ad.read_h5ad(paths["main_processed"])
    observed_times = np.asarray(main_processed.obs[time_key], dtype=float)
    time_grid_presets = _normalize_time_grid_presets(inference_cfg.get("time_grid_presets", {}))
    default_time_grid_mode = str(inference_cfg.get("time_grid_mode", next(iter(time_grid_presets.keys()))))

    run_grids = inference_cfg.get("run_grids")
    if not run_grids:
        run_grids = [
            {
                "mode": default_time_grid_mode,
                "values": inference_cfg.get("time_grid_values"),
            }
        ]

    passes_raw = inference_cfg.get("passes", ["sample"])
    if isinstance(passes_raw, str):
        passes = [passes_raw]
    elif isinstance(passes_raw, (list, tuple, np.ndarray)):
        passes = [str(x) for x in passes_raw]
    else:
        raise ValueError("inference.passes must be a string or list of strings")
    normalized_passes: List[str] = []
    for p in passes:
        p_norm = str(p).strip().lower()
        if not p_norm:
            continue
        if p_norm != "sample":
            LOGGER.warning("unknown inference pass '%s' ignored; only sample is supported.", p_norm)
            continue
        normalized_passes.append(p_norm)
    passes = normalized_passes if normalized_passes else ["sample"]
    sample_max = inference_cfg.get("sample_max_cells")
    sample_mode_default = str(inference_cfg.get("sampling_mode", "random")).strip().lower()
    sampling_cell_type_default = inference_cfg.get("sampling_cell_type", None)
    sampling_indices_default = inference_cfg.get("sampling_indices", None)
    strict_index_init_time_default = bool(inference_cfg.get("strict_index_init_time", True))
    if sample_mode_default not in {"random", "sample", "cell_type_random", "celltype_random", "cell_type", "index", "indices", "given_index"}:
        LOGGER.warning("unsupported inference.sampling_mode=%s, fallback to random", sample_mode_default)
        sample_mode_default = "random"

    reuse_cache = bool(inference_cfg.get("reuse_cache", True))

    workflow_runs: List[Dict[str, Any]] = []
    for grid_cfg in run_grids:
        mode = str(grid_cfg.get("mode", default_time_grid_mode))
        values = grid_cfg.get("values")
        grid_abs = resolve_time_grid(
            observed_times,
            mode=mode,
            values=values,
            time_grid_presets=time_grid_presets,
        )

        for pass_name in passes:
            pass_norm = str(pass_name).strip().lower()
            sampling_mode = sample_mode_default
            max_cells = sample_max

            run_dir = output_root / mode / pass_norm
            infer_res = infer_dense_time_dual(
                dynamic_adata_path=str(paths["dynamic_adata"]),
                main_processed_path=str(paths["main_processed"]),
                sub_processed_path=str(paths["sub_processed"]),
                map_model_path=str(paths["map_model_path"]),
                output_dir=str(run_dir / "assets"),
                time_grid=grid_abs,
                time_key=time_key,
                cell_type_key=cell_type_key,
                sampling_mode=sampling_mode,
                sampling_cell_type=sampling_cell_type_default,
                sampling_indices=sampling_indices_default,
                strict_index_init_time=strict_index_init_time_default,
                max_cells=max_cells,
                init_time=inference_cfg.get("init_time"),
                dt=dt,
                seed=seed,
                device=device,
                mapper_type=str(inference_cfg.get("mapper_type", "ae")),
                mapper_kwargs=inference_cfg.get("mapper_kwargs", {}),
                reuse_cache=reuse_cache,
            )

            analysis_res = analyze_dense_time_predictions(
                inference_output=infer_res,
                output_dir=str(run_dir),
                analysis_cfg=analysis_cfg,
            )
            summary_obj = analysis_res["analysis_summary"]
            workflow_runs.append(
                {
                    "grid_mode": mode,
                    "grid_values_abs": [float(x) for x in infer_res["time_grid"]],
                    "pass": pass_norm,
                    "run_dir": str(run_dir),
                    "assets_dir": str(run_dir / "assets"),
                    "tables_dir": analysis_res["tables_dir"],
                    "reports_dir": analysis_res["reports_dir"],
                    "primary_domain": int(infer_res.get("primary_domain", 1)),
                    "secondary_domain": int(infer_res.get("secondary_domain", 2)),
                    "primary_processed_path": str(infer_res.get("primary_processed_path", "")),
                    "secondary_processed_path": str(infer_res.get("secondary_processed_path", "")),
                    "domain1_processed_path": str(infer_res.get("domain1_processed_path", "")),
                    "domain2_processed_path": str(infer_res.get("domain2_processed_path", "")),
                    "classifier_eval": summary_obj.get("classifier_eval", {}),
                    "cell_type_eval": summary_obj.get("cell_type_eval", {}),
                    "mechanism_eval": summary_obj.get("mechanism_eval", {}),
                    "jacobian_grn_eval": summary_obj.get("jacobian_grn_eval", {}),
                    "perturbation_eval": summary_obj.get("perturbation_eval", {}),
                    "key_molecule_eval": summary_obj.get("key_molecule_eval", {}),
                }
            )

    summary = {
        "output_root": str(output_root),
        "num_runs": len(workflow_runs),
        "runs": workflow_runs,
    }
    _write_json(output_root / "workflow_summary.json", summary)
    return summary
