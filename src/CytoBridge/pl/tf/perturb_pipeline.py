from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
except Exception:  # pragma: no cover
    leaves_list = None
    linkage = None

from CytoBridge.tl.analysis_dense_time import (
    _build_ode_trajectory,
    _map_to_secondary,
    _project_feature_to_latent,
    _reconstruct_latent_to_feature,
)

from .context import TFRunContext
from .perturb import _build_indices_cache_settings, extract_perturbation_indices
from .regulation_umap import build_regulation_umap_outputs

DEFAULT_BATCH_PERTURB_SCORES: List[float] = [0.0, 25.0, 50.0, -25.0, -50.0]

# Normalized aliases for matching human-facing marker names to dataset var_names.
PROTEIN_NAME_ALIASES: Dict[str, List[str]] = {
    "CD41A": ["CD41a", "CD41"],
    "CD235A": ["CD235a", "CD235", "GLYA"],
    "FCER1A": ["FceRIa", "FCER1A"],
}

# A/B/C tiers derived from the user-provided biological relation list.
BIOLIST_TASKS: List[Dict[str, Any]] = [
    {"task_id": "a_itga2b_cd41", "tier": "A", "label": "ITGA2B->CD41", "genes": ["ITGA2B"], "proteins": ["CD41"]},
    {"task_id": "a_tfrc_cd71", "tier": "A", "label": "TFRC->CD71", "genes": ["TFRC"], "proteins": ["CD71"]},
    {"task_id": "b_klf1_ery_markers", "tier": "B", "label": "KLF1->(CD71,CD235a)", "genes": ["KLF1"], "proteins": ["CD71", "CD235a"]},
    {"task_id": "b_gata1_ery_markers", "tier": "B", "label": "GATA1->(CD71,CD36,CD235a)", "genes": ["GATA1"], "proteins": ["CD71", "CD36", "CD235a"]},
    {"task_id": "b_gata1_mk_markers", "tier": "B", "label": "GATA1->(CD41,CD42b)", "genes": ["GATA1"], "proteins": ["CD41", "CD42b"]},
    {"task_id": "b_gp1ba_cd42b", "tier": "B", "label": "GP1BA->CD42b", "genes": ["GP1BA"], "proteins": ["CD42b"]},
    {
        "task_id": "b_itga2b_gp1ba_mk_phenotype",
        "tier": "B",
        "label": "(ITGA2B,GP1BA)->(CD41,CD42b)",
        "genes": ["ITGA2B", "GP1BA"],
        "proteins": ["CD41", "CD42b"],
    },
    {"task_id": "c_mpo_mkery", "tier": "C", "label": "MPO->(CD71,CD41,CD42b)", "genes": ["MPO"], "proteins": ["CD71", "CD41", "CD42b"]},
    {"task_id": "c_lyz_mkery", "tier": "C", "label": "LYZ->(CD71,CD41,CD42b)", "genes": ["LYZ"], "proteins": ["CD71", "CD41", "CD42b"]},
    {"task_id": "c_mpo_lyz_mkery", "tier": "C", "label": "(MPO,LYZ)->(CD71,CD41,CD42b)", "genes": ["MPO", "LYZ"], "proteins": ["CD71", "CD41", "CD42b"]},
]


def _slug(text: Any) -> str:
    raw = str(text).strip().lower()
    out = []
    for ch in raw:
        out.append(ch if ch.isalnum() else "-")
    token = "".join(out).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "na"


def _signed_float_token(value: float) -> str:
    txt = f"{float(value):g}"
    txt = txt.replace("-", "m").replace("+", "p").replace(".", "d")
    return _slug(txt)


def _zscore(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    std = float(np.nanstd(x))
    if (not np.isfinite(std)) or std < 1e-12:
        return np.zeros_like(x, dtype=float)
    mean = float(np.nanmean(x))
    return (x - mean) / std


def _zscore_with_stats(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = np.asarray(arr, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    if (not np.isfinite(mean)) or (not np.isfinite(std)) or float(std) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - float(mean)) / float(std)


def _joint_normalize_pair_curves(
    sim_by_score: Dict[float, pd.DataFrame],
    real_df: pd.DataFrame,
) -> Tuple[Dict[float, pd.DataFrame], pd.DataFrame, Dict[str, Dict[str, float]]]:
    primary_all: List[np.ndarray] = [real_df["primary_value"].to_numpy(dtype=float)]
    secondary_all: List[np.ndarray] = [real_df["secondary_value"].to_numpy(dtype=float)]
    for z in sorted(sim_by_score.keys(), key=lambda x: float(x)):
        df = sim_by_score[float(z)]
        primary_all.append(df["primary_value"].to_numpy(dtype=float))
        secondary_all.append(df["secondary_value"].to_numpy(dtype=float))

    p_concat = np.concatenate(primary_all, axis=0) if primary_all else np.zeros((0,), dtype=float)
    s_concat = np.concatenate(secondary_all, axis=0) if secondary_all else np.zeros((0,), dtype=float)
    p_mean = float(np.nanmean(p_concat)) if p_concat.size > 0 else 0.0
    s_mean = float(np.nanmean(s_concat)) if s_concat.size > 0 else 0.0
    p_std = float(np.nanstd(p_concat)) if p_concat.size > 0 else 0.0
    s_std = float(np.nanstd(s_concat)) if s_concat.size > 0 else 0.0

    real_out = real_df.copy()
    real_out["primary_value"] = _zscore_with_stats(real_out["primary_value"].to_numpy(dtype=float), p_mean, p_std)
    real_out["secondary_value"] = _zscore_with_stats(real_out["secondary_value"].to_numpy(dtype=float), s_mean, s_std)

    sim_out: Dict[float, pd.DataFrame] = {}
    for z in sorted(sim_by_score.keys(), key=lambda x: float(x)):
        src = sim_by_score[float(z)]
        dst = src.copy()
        dst["primary_value"] = _zscore_with_stats(dst["primary_value"].to_numpy(dtype=float), p_mean, p_std)
        dst["secondary_value"] = _zscore_with_stats(dst["secondary_value"].to_numpy(dtype=float), s_mean, s_std)
        sim_out[float(z)] = dst

    stats = {
        "primary": {"mean": float(p_mean), "std": float(p_std)},
        "secondary": {"mean": float(s_mean), "std": float(s_std)},
    }
    return sim_out, real_out, stats


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    if weights is None:
        return float(np.mean(x))
    w = np.asarray(weights, dtype=float).reshape(-1)
    if x.shape[0] != w.shape[0]:
        raise ValueError(f"weights length mismatch: values={x.shape[0]} vs weights={w.shape[0]}")
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    if float(w.sum()) <= 1e-12:
        return float(np.mean(x))
    return float(np.average(x, weights=w))


def _auto_sample_n_from_time0(primary_proc, time_key: str, source_pool_n: int) -> Tuple[int, str, int, float]:
    if str(time_key) not in primary_proc.obs.columns:
        return int(max(1, source_pool_n)), "fallback_source_pool", int(max(1, source_pool_n)), 0.0
    t = np.asarray(primary_proc.obs[str(time_key)], dtype=float).reshape(-1)
    if t.size == 0:
        return int(max(1, source_pool_n)), "fallback_source_pool", int(max(1, source_pool_n)), 0.0
    if np.any(np.isclose(t, 0.0, atol=1e-8)):
        t0 = 0.0
    else:
        t0 = float(np.nanmin(t))
    n_t0 = int(np.sum(np.isclose(t, float(t0), atol=1e-8)))
    target = int(min(2500, max(1, int(n_t0 // 2))))
    return int(target), "min(2500,time0_real_n/2)", int(n_t0), float(t0)


def _source_indices_all_t0(primary_proc, time_key: str) -> Tuple[np.ndarray, float]:
    if str(time_key) not in primary_proc.obs.columns:
        n = int(primary_proc.n_obs)
        return np.arange(n, dtype=int), 0.0
    t = np.asarray(primary_proc.obs[str(time_key)], dtype=float).reshape(-1)
    if t.size == 0:
        n = int(primary_proc.n_obs)
        return np.arange(n, dtype=int), 0.0
    if np.any(np.isclose(t, 0.0, atol=1e-8)):
        t0 = 0.0
    else:
        t0 = float(np.nanmin(t))
    idx = np.where(np.isclose(t, float(t0), atol=1e-8))[0]
    if idx.size == 0:
        idx = np.arange(int(primary_proc.n_obs), dtype=int)
    return np.asarray(idx, dtype=int), float(t0)


def _build_dense_time_grid(
    native_time: Sequence[float],
    line_dt: float,
    points_between: int = 2,
    target_total_points: int = 0,
) -> np.ndarray:
    t = np.asarray(native_time, dtype=float).reshape(-1)
    t = np.asarray(np.unique(t), dtype=float)
    if t.size <= 1:
        return t
    n_intervals = int(t.size - 1)
    target_n = int(max(target_total_points, 0))
    if target_n > int(t.size):
        extra = int(target_n - int(t.size))
        base = int(extra // n_intervals)
        rem = int(extra % n_intervals)
        out: List[float] = []
        for i in range(n_intervals):
            t0 = float(t[i])
            t1 = float(t[i + 1])
            out.append(t0)
            k_i = base + (1 if i < rem else 0)
            if k_i > 0:
                gap = (t1 - t0) / float(k_i + 1)
                for j in range(1, k_i + 1):
                    out.append(t0 + float(j) * gap)
        out.append(float(t[-1]))
        return np.asarray(out, dtype=float)

    k = int(max(points_between, 0))
    if k > 0:
        out: List[float] = []
        for i in range(n_intervals):
            t0 = float(t[i])
            t1 = float(t[i + 1])
            out.append(t0)
            gap = (t1 - t0) / float(k + 1)
            for j in range(1, k + 1):
                out.append(t0 + float(j) * gap)
        out.append(float(t[-1]))
        return np.asarray(out, dtype=float)
    if (not np.isfinite(line_dt)) or line_dt <= 0:
        return t
    t0 = float(t.min())
    t1 = float(t.max())
    if t1 <= t0:
        return t
    dense = np.arange(t0, t1 + 0.5 * float(line_dt), float(line_dt), dtype=float)
    dense = np.clip(dense, t0, t1)
    if dense.size == 0 or (not np.isclose(dense[-1], t1)):
        dense = np.append(dense, t1)
    return np.asarray(np.unique(dense), dtype=float)


def _find_latest_indices_csv(ctx: TFRunContext) -> Optional[Path]:
    candidates = sorted(
        ctx.output_table_dir.glob("perturbation_selected_indices__*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return None


def _normalize_source_pool_mode(raw: Any) -> str:
    return str(raw).strip().lower() or "time0_random"


def _indices_csv_is_authoritative(cfg: Dict[str, Any]) -> bool:
    if bool(cfg.get("use_indices_csv", False)):
        return True
    return _is_indices_source_pool_mode(_normalize_source_pool_mode(cfg.get("source_pool_mode", "time0_random")))


def _normalize_optional_selector_token(raw: Any) -> str:
    token = str(raw).strip()
    if token.lower() in {"", "auto"}:
        return ""
    return token


def _normalize_optional_selector_float(raw: Any) -> Optional[float]:
    token = _normalize_optional_selector_token(raw)
    if not token:
        return None
    try:
        return float(token)
    except Exception:
        return None


def _build_indices_request_signature(
    ctx: TFRunContext,
    cfg: Dict[str, Any],
    *,
    label_rule: str,
    cell_type_merge_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "label_rule": str(label_rule).strip().lower(),
        "target_group": _normalize_optional_selector_token(cfg.get("target_group", "")),
        "z_score": _normalize_optional_selector_float(cfg.get("z_score", "")),
        "time_point": _normalize_optional_selector_float(cfg.get("time_point", "")),
        "cell_type": _normalize_optional_selector_token(cfg.get("cell_type", "")),
        "source_settings": _build_indices_cache_settings(
            default_cell_type_key=str(ctx.cell_type_key),
            source_cell_type_key=str(cfg.get("source_cell_type_key", "")),
            merge_cell_types=cell_type_merge_cfg,
        ),
    }


def _indices_meta_matches_request(meta: Dict[str, Any], request_sig: Dict[str, Any]) -> bool:
    current_rule = str(meta.get("label_rule", "")).strip().lower()
    requested_rule = str(request_sig.get("label_rule", "")).strip().lower()
    if (not current_rule) or (current_rule != requested_rule):
        return False

    requested_target_group = str(request_sig.get("target_group", "")).strip()
    if requested_target_group:
        current_target_group = str(meta.get("target_group", "")).strip()
        if current_target_group != requested_target_group:
            return False

    requested_z = request_sig.get("z_score", None)
    if requested_z is not None:
        try:
            current_z = float(meta.get("z_score"))
        except Exception:
            return False
        if not np.isclose(current_z, float(requested_z), atol=1e-8):
            return False

    requested_time = request_sig.get("time_point", None)
    if requested_time is not None:
        try:
            current_time = float(meta.get("time_requested"))
        except Exception:
            return False
        if not np.isclose(current_time, float(requested_time), atol=1e-8):
            return False

    requested_cell_type = str(request_sig.get("cell_type", "")).strip()
    if requested_cell_type:
        current_cell_type = str(meta.get("cell_type_requested", "")).strip()
        if current_cell_type != requested_cell_type:
            return False

    if ("source_cell_type_key" not in meta) or ("cell_type_merge" not in meta):
        return False

    cached_source_settings = _build_indices_cache_settings(
        default_cell_type_key=str(request_sig.get("source_settings", {}).get("source_cell_type_key", "cell_type")),
        source_cell_type_key=str(meta.get("source_cell_type_key", "")),
        merge_cell_types=meta.get("cell_type_merge", {}),
    )
    return cached_source_settings == dict(request_sig.get("source_settings", {}))


def _select_cached_indices_path(
    ctx: TFRunContext,
    *,
    explicit_path: Optional[Path],
    allow_explicit: bool,
    request_sig: Dict[str, Any],
) -> Optional[Path]:
    candidates: List[Path] = []
    if allow_explicit and explicit_path is not None:
        candidates.append(explicit_path)
    for path in sorted(ctx.output_table_dir.glob("perturbation_selected_indices__*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        if explicit_path is not None and path.resolve() == explicit_path.resolve():
            continue
        candidates.append(path)

    for path in candidates:
        if not path.exists():
            continue
        meta_json = Path(path).with_suffix(".json")
        if not meta_json.exists():
            continue
        try:
            with meta_json.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if isinstance(meta, dict) and _indices_meta_matches_request(meta, request_sig):
            return path.resolve()
    return None


def _perturb_manifest_filename(*, batch_mode: bool) -> str:
    return f"perturb_pipeline_manifest__{'batch' if bool(batch_mode) else 'single'}.json"


def _figure_manifest_filename(*, batch_mode: bool) -> str:
    return f"figure_manifest__{'batch' if bool(batch_mode) else 'single'}.json"


def _resolve_existing_manifest_path(
    out_dir: Path,
    *,
    batch_mode: bool,
    kind: str,
) -> Path:
    if str(kind) == "figure":
        preferred = out_dir / _figure_manifest_filename(batch_mode=bool(batch_mode))
        legacy = out_dir / "figure_manifest.json"
        alt = out_dir / _figure_manifest_filename(batch_mode=(not bool(batch_mode)))
    else:
        preferred = out_dir / _perturb_manifest_filename(batch_mode=bool(batch_mode))
        legacy = out_dir / "perturb_pipeline_manifest.json"
        alt = out_dir / _perturb_manifest_filename(batch_mode=(not bool(batch_mode)))

    for path in [preferred, legacy, alt]:
        if path.exists():
            return path
    return preferred


def _auto_points_between(n_real_points: int) -> int:
    n = int(max(n_real_points, 0))
    if n <= 3:
        return 3
    if n == 4:
        return 2
    if n < 10:
        return 1
    return 0


def _normalize_symbol(text: Any) -> str:
    raw = str(text).strip().upper()
    return "".join(ch for ch in raw if ch.isalnum())


def _unique_float_list(values: Sequence[Any]) -> List[float]:
    out: List[float] = []
    seen = set()
    for v in values:
        fv = float(v)
        key = f"{fv:.12g}"
        if key in seen:
            continue
        seen.add(key)
        out.append(float(fv))
    return out


def _is_batch_enabled(cfg_obj: Dict[str, Any]) -> bool:
    if bool(cfg_obj.get("batch_mode", False)):
        return True
    if str(cfg_obj.get("task_template", "")).strip():
        return True
    tasks = cfg_obj.get("perturb_tasks", None)
    return isinstance(tasks, list) and len(tasks) > 0


def _parse_perturb_scores(cfg_obj: Dict[str, Any]) -> List[float]:
    raw = cfg_obj.get("perturb_scores", None)
    if raw is None:
        return list(DEFAULT_BATCH_PERTURB_SCORES)
    if isinstance(raw, str):
        vals = [x.strip() for x in raw.split(",") if x.strip()]
        return _unique_float_list([float(x) for x in vals]) if vals else list(DEFAULT_BATCH_PERTURB_SCORES)
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) == 0:
            return list(DEFAULT_BATCH_PERTURB_SCORES)
        return _unique_float_list([float(x) for x in raw])
    return list(DEFAULT_BATCH_PERTURB_SCORES)


def _extract_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if str(x).strip()]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _build_batch_tasks(cfg_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_tasks = cfg_obj.get("perturb_tasks", None)
    tasks: List[Dict[str, Any]] = []
    if isinstance(raw_tasks, list) and raw_tasks:
        for i, row in enumerate(raw_tasks):
            if not isinstance(row, dict):
                continue
            primary_vars = _extract_list(row.get("primary_vars", row.get("genes", row.get("gene", []))))
            secondary_vars = _extract_list(row.get("secondary_vars", row.get("proteins", row.get("protein", []))))
            if (not primary_vars) and (not secondary_vars):
                continue
            task_id = str(row.get("task_id", f"task_{i+1}")).strip() or f"task_{i+1}"
            tasks.append(
                {
                    "task_id": task_id,
                    "tier": str(row.get("tier", "")).strip(),
                    "label": str(row.get("label", task_id)).strip() or task_id,
                    "primary_vars": primary_vars,
                    "secondary_vars": secondary_vars,
                    "genes": list(primary_vars),
                    "proteins": list(secondary_vars),
                }
            )
        return tasks

    template = str(cfg_obj.get("task_template", "")).strip().lower()
    if template in {"biolist", "bio", "biological", "default", "ho_regulome"}:
        out: List[Dict[str, Any]] = []
        for row in BIOLIST_TASKS:
            item = dict(row)
            item["primary_vars"] = _extract_list(item.get("genes", []))
            item["secondary_vars"] = _extract_list(item.get("proteins", []))
            out.append(item)
        return out
    return []


def _normalize_perturb_domain(raw: Any) -> str:
    token = str(raw).strip().lower() if raw is not None else ""
    if token in {"", "auto", "default"}:
        return "primary"
    if token in {"primary", "pri", "p", "1", "main", "domain1"}:
        return "primary"
    if token in {"secondary", "sec", "sub", "s", "2", "domain2"}:
        return "secondary"
    if token in {"both", "all", "dual", "primary+secondary", "secondary+primary", "1+2", "2+1"}:
        return "both"
    return "primary"


def _map_with_mapper(mapper: Any, latent: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    method_name = "T_rev" if bool(reverse) else "T"
    fn = getattr(mapper, method_name, None)
    if fn is None:
        raise ValueError(f"mapper does not provide {method_name}")
    out = fn(np.asarray(latent, dtype=np.float32))
    if hasattr(out, "detach"):
        arr = out.detach().cpu().numpy()
    else:
        arr = np.asarray(out)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"unexpected mapper output shape for {method_name}: {arr.shape}")
    return arr


def _is_indices_source_pool_mode(source_pool_mode: str) -> bool:
    mode = str(source_pool_mode).strip().lower()
    return mode in {"indices", "selected_indices", "filtered_indices"}


def _resolve_source_pool(
    *,
    source_proc,
    idx_df: pd.DataFrame,
    time_key: str,
    source_pool_mode: str,
    domain: str,
) -> Tuple[np.ndarray, str, float]:
    if _is_indices_source_pool_mode(source_pool_mode):
        source_idx_raw = np.unique(np.asarray(idx_df["source_cell_index"], dtype=int))
        if source_idx_raw.size == 0:
            raise ValueError("No source_cell_index in perturbation indices csv")
        source_idx_all = source_idx_raw[(source_idx_raw >= 0) & (source_idx_raw < int(source_proc.n_obs))]
        if source_idx_all.size == 0:
            raise ValueError(
                f"No valid source indices for perturb_domain={domain}. "
                f"source_pool={int(source_proc.n_obs)}, raw_size={int(source_idx_raw.size)}"
            )
        source_pool_filter = "indices_csv(target_group/time_used/cell_type_used)"
        source_pool_time_value = 0.0
        if "time_used" in idx_df.columns and len(idx_df) > 0:
            try:
                source_pool_time_value = float(idx_df["time_used"].iloc[0])
            except Exception:
                source_pool_time_value = 0.0
        return np.asarray(source_idx_all, dtype=int), str(source_pool_filter), float(source_pool_time_value)

    source_idx_all, source_pool_time_value = _source_indices_all_t0(source_proc, time_key)
    return np.asarray(source_idx_all, dtype=int), "time0_only", float(source_pool_time_value)


def _resolve_target_group(cfg_obj: Dict[str, Any], idx_df: pd.DataFrame, source_pool_mode: str) -> str:
    target_group = str(cfg_obj.get("target_group", "")).strip()
    if target_group:
        return target_group
    if _is_indices_source_pool_mode(source_pool_mode) and "target_group" in idx_df.columns:
        return str(idx_df["target_group"].iloc[0])
    return "all_t0_random"


def _sample_source_indices(
    *,
    source_proc,
    time_key: str,
    source_idx_all: np.ndarray,
    sample_n_cfg: Any,
    seed: int,
) -> Tuple[np.ndarray, int, int, str, int, float]:
    auto_sample_n, _, time0_real_n, time0_value = _auto_sample_n_from_time0(
        source_proc, time_key, int(source_idx_all.size)
    )
    if isinstance(sample_n_cfg, str) and str(sample_n_cfg).strip().lower() in {"", "auto"}:
        source_idx = np.asarray(np.unique(source_idx_all), dtype=int)
        return (
            np.asarray(source_idx, dtype=int),
            int(source_idx.size),
            int(source_idx.size),
            "source_pool_full",
            int(time0_real_n),
            float(time0_value),
        )
    else:
        try:
            cfg_n = int(sample_n_cfg)
        except Exception:
            cfg_n = int(auto_sample_n)
        target_sample_n = int(auto_sample_n) if cfg_n <= 0 else int(min(cfg_n, auto_sample_n))
        sample_policy = f"random_without_replacement(n={int(target_sample_n)})"

    rng = np.random.default_rng(int(seed))
    n_take = int(min(max(int(target_sample_n), 1), source_idx_all.size))
    source_idx = np.asarray(rng.choice(source_idx_all, size=n_take, replace=False), dtype=int)
    return source_idx, int(n_take), int(auto_sample_n), str(sample_policy), int(time0_real_n), float(time0_value)


def _resolve_dense_time_plan(
    *,
    primary_proc,
    time_key: str,
    ctx_time_grid: Sequence[float],
    line_dt: float,
    dense_points_between: int,
    dense_target_points: int,
    dense_keep_real_threshold: int,
) -> Tuple[np.ndarray, List[float], int]:
    real_time_points = sorted({float(x) for x in np.asarray(primary_proc.obs[time_key], dtype=float).tolist()})
    dense_source_time = real_time_points if len(real_time_points) >= 2 else [float(x) for x in ctx_time_grid]
    resolved_points_between = 0
    if len(real_time_points) >= int(max(dense_keep_real_threshold, 2)):
        dense_time = np.asarray(dense_source_time, dtype=float)
    else:
        if int(dense_target_points) <= 0:
            resolved_points_between = int(dense_points_between)
            if resolved_points_between <= 0:
                resolved_points_between = _auto_points_between(len(real_time_points))
        dense_time = _build_dense_time_grid(
            dense_source_time,
            float(line_dt),
            points_between=int(resolved_points_between),
            target_total_points=int(dense_target_points),
        )
    return np.asarray(dense_time, dtype=float), [float(x) for x in real_time_points], int(resolved_points_between)


def _simulate_perturbed_trajectories(
    *,
    domain: str,
    x0_feat: np.ndarray,
    primary_bundle,
    secondary_bundle,
    model,
    mapper,
    dyn_adata,
    dense_time: np.ndarray,
    ode_dt: float,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    if str(domain) == "primary":
        z0 = _project_feature_to_latent(x0_feat, primary_bundle)
        mapping_chain = "primary_feature->primary_latent->ode->secondary_latent"
    else:
        z0_secondary = _project_feature_to_latent(x0_feat, secondary_bundle)
        z0 = _map_with_mapper(mapper, z0_secondary, reverse=True)
        mapping_chain = "secondary_feature->secondary_latent->primary_latent(T_rev)->ode->secondary_latent(T)"

    traj_primary = _build_ode_trajectory(
        model=model,
        x0=z0,
        times=dense_time,
        adata=dyn_adata,
        dt=ode_dt,
        device=device,
    )
    traj_secondary = _map_to_secondary(mapper, traj_primary["main_latent"])
    return traj_primary, traj_secondary, str(mapping_chain)


def _build_source_files_meta(ctx: TFRunContext, idx_path: Path) -> Tuple[List[str], Dict[str, bool]]:
    source_files_read = [
        str(idx_path),
        str(ctx.assets_dir / "manifest.json"),
        str(ctx.assets_dir / "qc_metrics.json"),
        str(ctx.output_table_dir / "tf_pairs_final.csv"),
    ]
    source_files_exists = {p: bool(Path(p).exists()) for p in source_files_read}
    return source_files_read, source_files_exists


def _write_perturb_manifest(out_fig_dir: Path, payload: Dict[str, Any], *, batch_mode: bool) -> Path:
    manifest_json = out_fig_dir / _perturb_manifest_filename(batch_mode=bool(batch_mode))
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return manifest_json


def _safe_z_order(raw: Any, fallback: Sequence[float]) -> List[float]:
    if raw is None:
        return [float(x) for x in fallback]
    if isinstance(raw, str):
        vals = [x.strip() for x in raw.split(",") if x.strip()]
        if not vals:
            return [float(x) for x in fallback]
        return [float(x) for x in vals]
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) == 0:
            return [float(x) for x in fallback]
        return [float(x) for x in raw]
    return [float(x) for x in fallback]


def _latest_table_by_pattern(ctx: TFRunContext, pattern: str) -> Optional[Path]:
    cands = sorted(ctx.tables_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _table_candidates_by_pattern(ctx: TFRunContext, pattern: str) -> List[Path]:
    return sorted(ctx.tables_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def _group_matches(lhs: Any, rhs: Any) -> bool:
    left = str(lhs).strip()
    right = str(rhs).strip()
    if (not left) or (not right):
        return False
    return left.lower() == right.lower() or _slug(left) == _slug(right)


def _resolve_fate_heatmap_sweep(
    ctx: TFRunContext,
    cfg_obj: Dict[str, Any],
    fate_cfg: Dict[str, Any],
    *,
    z_order: Sequence[float],
) -> Dict[str, Any]:
    explicit_path_raw = str(fate_cfg.get("sweep_csv", "")).strip()
    if explicit_path_raw:
        requested_group = str(fate_cfg.get("target_group", "")).strip()
    else:
        requested_group = str(fate_cfg.get("target_group", cfg_obj.get("target_group", ""))).strip()
    allow_partial_z = bool(fate_cfg.get("allow_partial_z", False))
    requested_z = [float(x) for x in z_order]
    need = {"cell_type", "z_score", "delta_rate", "perturbed_prob"}

    if explicit_path_raw:
        candidates = [Path(explicit_path_raw).resolve()]
    else:
        candidates = _table_candidates_by_pattern(ctx, "perturbation_zscore_sweep__*.csv")
    if not candidates:
        return {
            "status": "skipped",
            "reason": "no sweep csv found",
            "requested_target_group": requested_group,
            "requested_z_order": requested_z,
            "allow_partial_z": allow_partial_z,
            "checked_candidates": [],
        }

    checked: List[Dict[str, Any]] = []
    for path in candidates:
        entry: Dict[str, Any] = {"path": str(path)}
        if not path.exists():
            entry["status"] = "missing"
            checked.append(entry)
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            entry["status"] = "read_error"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
            checked.append(entry)
            continue

        cols = set(df.columns)
        if not need.issubset(cols):
            entry["status"] = "missing_columns"
            entry["reason"] = f"missing columns: {sorted(list(need - cols))}"
            checked.append(entry)
            continue

        work = df.copy()
        if requested_group and ("target_group" in cols):
            mask = work["target_group"].astype(str).map(lambda x: _group_matches(x, requested_group))
            if not bool(mask.any()):
                entry["status"] = "target_group_mismatch"
                entry["present_target_groups"] = sorted(work["target_group"].astype(str).dropna().unique().tolist())
                checked.append(entry)
                continue
            work = work.loc[mask].copy()

        present_z = sorted(
            {float(x) for x in pd.to_numeric(work["z_score"], errors="coerce").dropna().astype(float).tolist()}
        )
        entry["status"] = "ok"
        entry["present_z"] = present_z
        missing_z = [float(z) for z in requested_z if not any(np.isclose(float(z), float(pz)) for pz in present_z)]
        if requested_z and missing_z and (not allow_partial_z):
            entry["status"] = "z_mismatch"
            entry["missing_z"] = missing_z
            checked.append(entry)
            continue

        checked.append(entry)
        return {
            "status": "success",
            "sweep_csv": str(path),
            "work_df": work,
            "requested_target_group": requested_group,
            "requested_z_order": requested_z,
            "allow_partial_z": allow_partial_z,
            "checked_candidates": checked,
            "missing_z": missing_z,
        }

    return {
        "status": "skipped",
        "reason": "no compatible sweep csv found",
        "requested_target_group": requested_group,
        "requested_z_order": requested_z,
        "allow_partial_z": allow_partial_z,
        "checked_candidates": checked,
    }


def _plot_simple_heatmap(
    mat: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_png: Path,
    title: str,
    cmap: str = "RdBu_r",
    center_zero: bool = False,
    cbar_label: str = "",
) -> None:
    arr = np.asarray(mat, dtype=float)
    fig_h = max(4.8, 0.32 * int(arr.shape[0]) + 2.0)
    fig_w = max(8.4, 0.75 * int(arr.shape[1]) + 3.0)
    plt.figure(figsize=(fig_w, fig_h), dpi=300)
    if center_zero:
        vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
        vmax = max(vmax, 1e-8)
        im = plt.imshow(arr, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    else:
        vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
        vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or np.isclose(vmin, vmax):
            vmin, vmax = 0.0, 1.0
        im = plt.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.yticks(np.arange(len(row_labels)), [str(x) for x in row_labels], fontsize=8)
    plt.xticks(np.arange(len(col_labels)), [str(x) for x in col_labels], fontsize=9)
    plt.xlabel("z score/time")
    plt.ylabel("cell type/variable")
    plt.title(str(title))
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    if str(cbar_label).strip():
        cbar.set_label(str(cbar_label))
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(output_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def _row_cluster_order(
    mat: np.ndarray,
    *,
    row_cluster: bool,
    method: str,
    metric: str,
) -> Tuple[List[int], Dict[str, Any]]:
    n_rows = int(np.asarray(mat).shape[0])
    default = list(range(n_rows))
    meta = {
        "enabled": bool(row_cluster),
        "applied": False,
        "fallback": False,
        "fallback_reason": "",
        "method": str(method),
        "metric": str(metric),
        "n_rows": int(n_rows),
    }
    if (not bool(row_cluster)) or n_rows < 2:
        meta["fallback_reason"] = "disabled_or_too_few_rows"
        return default, meta
    if linkage is None or leaves_list is None:
        meta["fallback"] = True
        meta["fallback_reason"] = "scipy_unavailable"
        return default, meta
    try:
        arr = np.asarray(mat, dtype=float)
        col_mean = np.nanmean(arr, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        rows = np.where(np.isfinite(arr), arr, col_mean[None, :])
        Z = linkage(rows, method=str(method), metric=str(metric), optimal_ordering=True)
        order = [int(x) for x in leaves_list(Z).astype(int).tolist()]
        if len(order) != n_rows:
            raise ValueError("cluster row count mismatch")
        meta["applied"] = True
        return order, meta
    except Exception as exc:
        meta["fallback"] = True
        meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return default, meta


def _build_fate_heatmap_outputs(ctx: TFRunContext, out_fig_dir: Path, cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    fate_cfg = cfg_obj.get("fate_heatmap", {})
    if not isinstance(fate_cfg, dict) or (not bool(fate_cfg.get("enabled", False))):
        return {"enabled": False}
    z_order = _safe_z_order(fate_cfg.get("z_order", None), [0.0, 2.0, 4.0, 6.0, -2.0, -4.0, -6.0])
    sweep_res = _resolve_fate_heatmap_sweep(ctx, cfg_obj, fate_cfg, z_order=z_order)
    if str(sweep_res.get("status", "")) != "success":
        return {
            "enabled": True,
            "status": "skipped",
            "reason": str(sweep_res.get("reason", "no compatible sweep csv found")),
            "requested_target_group": str(sweep_res.get("requested_target_group", "")),
            "requested_z_order": [float(x) for x in sweep_res.get("requested_z_order", [])],
            "allow_partial_z": bool(sweep_res.get("allow_partial_z", False)),
            "checked_candidates": sweep_res.get("checked_candidates", []),
        }
    sweep_csv = Path(str(sweep_res["sweep_csv"])).resolve()
    focus_types = fate_cfg.get("focus_cell_types", [])
    if isinstance(focus_types, str):
        focus_types = [x.strip() for x in focus_types.split(",") if x.strip()]
    focus_types = [str(x) for x in focus_types] if isinstance(focus_types, list) else []
    work = sweep_res["work_df"].copy()
    if focus_types:
        lut = {str(x).lower() for x in focus_types}
        work = work[work["cell_type"].astype(str).str.lower().isin(lut)].copy()
    if work.empty:
        return {"enabled": True, "status": "skipped", "reason": "no rows after focus_cell_types filter"}
    cell_order = (
        work.groupby("cell_type", as_index=False)["delta_rate"]
        .apply(lambda x: float(np.mean(np.abs(np.asarray(x, dtype=float)))))
        .rename(columns={"delta_rate": "score"})
        .sort_values("score", ascending=False)["cell_type"].astype(str).tolist()
    )
    delta_mat = (
        work.pivot_table(index="cell_type", columns="z_score", values="delta_rate", aggfunc="mean")
        .reindex(index=cell_order, columns=z_order)
    )
    prob_mat = (
        work.pivot_table(index="cell_type", columns="z_score", values="perturbed_prob", aggfunc="mean")
        .reindex(index=cell_order, columns=z_order)
    )
    heat_dir = out_fig_dir / "fate_heatmap"
    heat_dir.mkdir(parents=True, exist_ok=True)
    delta_csv = heat_dir / "fate_delta_rate_heatmap.csv"
    prob_csv = heat_dir / "fate_perturbed_prob_heatmap.csv"
    delta_png = heat_dir / "fate_delta_rate_heatmap.pdf"
    prob_png = heat_dir / "fate_perturbed_prob_heatmap.pdf"
    delta_mat.to_csv(delta_csv)
    prob_mat.to_csv(prob_csv)
    _plot_simple_heatmap(
        delta_mat.to_numpy(dtype=float),
        row_labels=delta_mat.index.astype(str).tolist(),
        col_labels=[f"{float(x):g}" for x in z_order],
        output_png=delta_png,
        title="Perturbation Fate Delta Rate Heatmap",
        cmap="RdBu_r",
        center_zero=True,
        cbar_label="delta rate",
    )
    _plot_simple_heatmap(
        prob_mat.to_numpy(dtype=float),
        row_labels=prob_mat.index.astype(str).tolist(),
        col_labels=[f"{float(x):g}" for x in z_order],
        output_png=prob_png,
        title="Perturbation Fate Probability Heatmap",
        cmap="YlGnBu",
        center_zero=False,
        cbar_label="perturbed probability",
    )
    return {
        "enabled": True,
        "status": "success",
        "sweep_csv": str(sweep_csv),
        "requested_target_group": str(sweep_res.get("requested_target_group", "")),
        "z_order_used": [float(x) for x in z_order],
        "missing_z": [float(x) for x in sweep_res.get("missing_z", [])],
        "checked_candidates": sweep_res.get("checked_candidates", []),
        "outputs": {
            "delta_csv": str(delta_csv),
            "prob_csv": str(prob_csv),
            "delta_png": str(delta_png),
            "prob_png": str(prob_png),
        },
    }


def _collect_prior_symbols(prior_txt: Path) -> List[str]:
    if not prior_txt.exists():
        return []
    import re

    text = prior_txt.read_text(encoding="utf-8", errors="ignore")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]*", text)
    out: List[str] = []
    seen = set()
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(str(t))
    return out


def _build_variable_time_heatmap_outputs(ctx: TFRunContext, out_fig_dir: Path, cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    vh_cfg = cfg_obj.get("variable_heatmap", {})
    if not isinstance(vh_cfg, dict) or (not bool(vh_cfg.get("enabled", False))):
        return {"enabled": False}

    primary_proc, secondary_proc = ctx.load_primary_secondary_processed(backed=False)
    primary_bundle, secondary_bundle = ctx.load_primary_secondary_bundles()
    traj_primary, traj_secondary, weights = ctx.load_traj_arrays()
    times = np.asarray(ctx.time_grid, dtype=float).reshape(-1)

    p_vars_raw = vh_cfg.get("primary_vars", [])
    s_vars_raw = vh_cfg.get("secondary_vars", [])
    if isinstance(p_vars_raw, str):
        p_vars_raw = [x.strip() for x in p_vars_raw.split(",") if x.strip()]
    if isinstance(s_vars_raw, str):
        s_vars_raw = [x.strip() for x in s_vars_raw.split(",") if x.strip()]
    p_vars_raw = [str(x) for x in p_vars_raw] if isinstance(p_vars_raw, list) else []
    s_vars_raw = [str(x) for x in s_vars_raw] if isinstance(s_vars_raw, list) else []

    auto_top_n = int(vh_cfg.get("auto_top_n", 25))
    max_vars = int(vh_cfg.get("max_vars", 40))
    source_mode = str(vh_cfg.get("source_mode", "prior_plus_auto")).strip().lower() or "prior_plus_auto"
    prior_path = Path(str(vh_cfg.get("prior_txt", ""))).resolve() if str(vh_cfg.get("prior_txt", "")).strip() else None

    p_lut = {str(v): i for i, v in enumerate(primary_bundle.var_names)}
    s_lut = {str(v): i for i, v in enumerate(secondary_bundle.var_names)}

    auto_p: List[str] = []
    auto_s: List[str] = []
    pair_csv = ctx.output_table_dir / "tf_pairs_final.csv"
    if pair_csv.exists():
        pair_df = pd.read_csv(pair_csv)
        if (not pair_df.empty) and ("primary_var" in pair_df.columns) and ("secondary_var" in pair_df.columns):
            if "score" in pair_df.columns:
                pair_df = pair_df.copy()
                pair_df["abs_score"] = np.abs(np.asarray(pair_df["score"], dtype=float))
                pair_df = pair_df.sort_values("abs_score", ascending=False)
            auto_p = [str(x) for x in pair_df["primary_var"].astype(str).tolist()]
            auto_s = [str(x) for x in pair_df["secondary_var"].astype(str).tolist()]

    prior_vars = _collect_prior_symbols(prior_path) if prior_path is not None else []
    pri_p = [x for x in prior_vars if x in p_lut]
    pri_s = [x for x in prior_vars if x in s_lut]

    def _merge_vars(explicit: List[str], prior: List[str], auto: List[str], lut: Dict[str, int]) -> List[str]:
        seq: List[str] = []
        if source_mode in {"explicit_only"}:
            seq = explicit
        elif source_mode in {"prior_only"}:
            seq = prior
        elif source_mode in {"auto_only"}:
            seq = auto
        else:
            seq = explicit + prior + auto
        out: List[str] = []
        seen = set()
        for x in seq:
            s = str(x)
            if s in seen or s not in lut:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= max_vars:
                break
        if len(out) == 0:
            for x in auto:
                s = str(x)
                if s in seen or s not in lut:
                    continue
                seen.add(s)
                out.append(s)
                if len(out) >= min(auto_top_n, max_vars):
                    break
        return out

    p_vars = _merge_vars(p_vars_raw, pri_p, auto_p, p_lut)
    s_vars = _merge_vars(s_vars_raw, pri_s, auto_s, s_lut)
    if (not p_vars) and (not s_vars):
        return {"enabled": True, "status": "skipped", "reason": "no valid variables for heatmap"}

    def _build_mat(latent_series: np.ndarray, bundle, vars_use: List[str]) -> np.ndarray:
        if not vars_use:
            return np.zeros((0, len(times)), dtype=float)
        idx = [int({str(v): i for i, v in enumerate(bundle.var_names)}[v]) for v in vars_use]
        recon_sel = np.asarray(bundle.reconstruction[:, idx], dtype=float)
        mean_sel = np.asarray(bundle.mean[idx], dtype=float).reshape(1, -1)
        t_num = int(latent_series.shape[0])
        out = np.zeros((len(vars_use), t_num), dtype=float)
        for ti in range(t_num):
            z = np.asarray(latent_series[ti], dtype=float)
            feat = np.asarray(z @ recon_sel + mean_sel, dtype=float)
            wt = None if weights is None else np.asarray(weights[ti], dtype=float)
            if wt is None:
                out[:, ti] = np.asarray(np.mean(feat, axis=0), dtype=float)
            else:
                w = np.where(np.isfinite(wt), wt, 0.0)
                w = np.clip(w, 0.0, None)
                if float(np.sum(w)) <= 1e-12:
                    out[:, ti] = np.asarray(np.mean(feat, axis=0), dtype=float)
                else:
                    out[:, ti] = np.asarray(np.average(feat, axis=0, weights=w), dtype=float)
        return out

    p_mat_raw = _build_mat(np.asarray(traj_primary, dtype=float), primary_bundle, p_vars)
    s_mat_raw = _build_mat(np.asarray(traj_secondary, dtype=float), secondary_bundle, s_vars)

    row_cluster = bool(vh_cfg.get("row_cluster", True))
    method = str(vh_cfg.get("cluster_method", "average"))
    metric = str(vh_cfg.get("cluster_metric", "euclidean"))
    zscore_rows = bool(vh_cfg.get("row_zscore", True))

    def _row_z(mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=float)
        out = np.zeros_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            r = arr[i]
            m = float(np.nanmean(r))
            s = float(np.nanstd(r))
            if (not np.isfinite(s)) or s < 1e-12:
                out[i] = np.zeros_like(r, dtype=float)
            else:
                out[i] = (r - m) / s
        return out

    p_mat = _row_z(p_mat_raw) if zscore_rows and p_mat_raw.size else p_mat_raw
    s_mat = _row_z(s_mat_raw) if zscore_rows and s_mat_raw.size else s_mat_raw

    p_order, p_meta = _row_cluster_order(p_mat, row_cluster=row_cluster, method=method, metric=metric)
    s_order, s_meta = _row_cluster_order(s_mat, row_cluster=row_cluster, method=method, metric=metric)
    if p_mat.size:
        p_mat = p_mat[p_order, :]
        p_vars = [p_vars[i] for i in p_order]
    if s_mat.size:
        s_mat = s_mat[s_order, :]
        s_vars = [s_vars[i] for i in s_order]

    out_dir = out_fig_dir / "variable_time_heatmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    t_labels = [f"{float(x):g}" for x in times.tolist()]

    if p_mat.size:
        p_csv = out_dir / "primary_variable_time_heatmap.csv"
        p_png = out_dir / "primary_variable_time_heatmap.pdf"
        pd.DataFrame(p_mat, index=p_vars, columns=t_labels).to_csv(p_csv)
        _plot_simple_heatmap(
            p_mat,
            row_labels=p_vars,
            col_labels=t_labels,
            output_png=p_png,
            title="Primary Variable Time Heatmap",
            cmap="RdBu_r" if zscore_rows else "viridis",
            center_zero=bool(zscore_rows),
            cbar_label="row z-score" if zscore_rows else "value",
        )
        outputs["primary_csv"] = str(p_csv)
        outputs["primary_png"] = str(p_png)

    if s_mat.size:
        s_csv = out_dir / "secondary_variable_time_heatmap.csv"
        s_png = out_dir / "secondary_variable_time_heatmap.pdf"
        pd.DataFrame(s_mat, index=s_vars, columns=t_labels).to_csv(s_csv)
        _plot_simple_heatmap(
            s_mat,
            row_labels=s_vars,
            col_labels=t_labels,
            output_png=s_png,
            title="Secondary Variable Time Heatmap",
            cmap="RdBu_r" if zscore_rows else "viridis",
            center_zero=bool(zscore_rows),
            cbar_label="row z-score" if zscore_rows else "value",
        )
        outputs["secondary_csv"] = str(s_csv)
        outputs["secondary_png"] = str(s_png)

    return {
        "enabled": True,
        "status": "success",
        "source_mode": str(source_mode),
        "zscore_rows": bool(zscore_rows),
        "primary_cluster": p_meta,
        "secondary_cluster": s_meta,
        "n_primary_vars": int(len(p_vars)),
        "n_secondary_vars": int(len(s_vars)),
        "outputs": outputs,
    }


def _resolve_names_with_aliases(
    requested: Sequence[str],
    *,
    var_names: Sequence[str],
    alias_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[str], List[str]]:
    name_idx: Dict[str, str] = {}
    for name in var_names:
        key = _normalize_symbol(name)
        if key and key not in name_idx:
            name_idx[key] = str(name)

    used: List[str] = []
    missing: List[str] = []
    seen = set()
    for raw in requested:
        token = str(raw).strip()
        if not token:
            continue
        cand: List[str] = [token]
        norm = _normalize_symbol(token)
        if alias_map and norm in alias_map:
            cand.extend([str(x) for x in alias_map[norm]])
        found = None
        for c in cand:
            key = _normalize_symbol(c)
            if key in name_idx:
                found = name_idx[key]
                break
        if found is None:
            missing.append(token)
            continue
        if found in seen:
            continue
        seen.add(found)
        used.append(found)
    return used, missing


def _build_sim_matrix(
    *,
    latent_series: np.ndarray,
    time_grid: np.ndarray,
    reconstruction: np.ndarray,
    mean: np.ndarray,
    var_idx: Sequence[int],
    weights_by_time: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    idx = [int(x) for x in var_idx]
    if not idx:
        return pd.DataFrame(columns=["time"])
    recon_sel = np.asarray(reconstruction[:, idx], dtype=float)
    mean_sel = np.asarray(mean[idx], dtype=float).reshape(1, -1)
    t_num = int(latent_series.shape[0])
    out = np.zeros((t_num, len(idx)), dtype=float)
    for i in range(t_num):
        lat = np.asarray(latent_series[i], dtype=float)
        feat = np.asarray(lat @ recon_sel + mean_sel, dtype=float)
        wt = None if weights_by_time is None else np.asarray(weights_by_time[i], dtype=float)
        if wt is None:
            out[i, :] = np.asarray(np.mean(feat, axis=0), dtype=float)
        else:
            w = np.where(np.isfinite(wt), wt, 0.0)
            w = np.clip(w, 0.0, None)
            if float(np.sum(w)) <= 1e-12:
                out[i, :] = np.asarray(np.mean(feat, axis=0), dtype=float)
            else:
                out[i, :] = np.asarray(np.average(feat, axis=0, weights=w), dtype=float)
    cols = [f"v{i}" for i in range(len(idx))]
    df = pd.DataFrame(out, columns=cols)
    df.insert(0, "time", np.asarray(time_grid, dtype=float))
    return df


def _build_real_matrix(
    *,
    adata,
    time_key: str,
    reconstruction: np.ndarray,
    mean: np.ndarray,
    var_idx: Sequence[int],
) -> pd.DataFrame:
    if str(time_key) not in adata.obs.columns:
        raise ValueError(f"time_key not found in processed adata.obs: {time_key}")
    idx = [int(x) for x in var_idx]
    if not idx:
        return pd.DataFrame(columns=["time"])
    latent = np.asarray(adata.obsm["X_latent"], dtype=float)
    recon_sel = np.asarray(reconstruction[:, idx], dtype=float)
    mean_sel = np.asarray(mean[idx], dtype=float).reshape(1, -1)
    feat = np.asarray(latent @ recon_sel + mean_sel, dtype=float)
    cols = [f"v{i}" for i in range(len(idx))]
    df = pd.DataFrame(feat, columns=cols)
    df["time"] = np.asarray(adata.obs[str(time_key)], dtype=float)
    out = df.groupby("time", as_index=False)[cols].mean().sort_values("time")
    return out


def _aggregate_curve_from_matrix(df: pd.DataFrame, *, zscore_before_mean: bool = True) -> pd.DataFrame:
    if df.empty or "time" not in df.columns:
        return pd.DataFrame(columns=["time", "value"])
    cols = [c for c in df.columns if c != "time"]
    if not cols:
        return pd.DataFrame(columns=["time", "value"])
    mat = df[cols].to_numpy(dtype=float)
    if bool(zscore_before_mean):
        zmat = np.zeros_like(mat, dtype=float)
        for j in range(mat.shape[1]):
            zmat[:, j] = _zscore(mat[:, j])
        agg = np.asarray(np.nanmean(zmat, axis=1), dtype=float)
    else:
        agg = np.asarray(np.nanmean(mat, axis=1), dtype=float)
    return pd.DataFrame({"time": df["time"].to_numpy(dtype=float), "value": agg})


def _mean_feature_subset(feature_matrix: np.ndarray, var_idx: Sequence[int]) -> float:
    idx = [int(x) for x in var_idx]
    if not idx:
        return 0.0
    mat = np.asarray(feature_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape={mat.shape}")
    return float(np.nanmean(mat[:, idx]))


def _reconstruct_feature_subset(
    latent: np.ndarray,
    reconstruction: np.ndarray,
    mean: np.ndarray,
    var_idx: Sequence[int],
) -> np.ndarray:
    idx = [int(x) for x in var_idx]
    if not idx:
        return np.zeros((int(np.asarray(latent).shape[0]), 0), dtype=float)
    z = np.asarray(latent, dtype=float)
    recon_sel = np.asarray(reconstruction[:, idx], dtype=float)
    mean_sel = np.asarray(mean[idx], dtype=float).reshape(1, -1)
    return np.asarray(z @ recon_sel + mean_sel, dtype=float)


def _compute_batch_task_anchors(
    *,
    domain: str,
    x0_feat: np.ndarray,
    primary_idx: Sequence[int],
    secondary_idx: Sequence[int],
    primary_bundle,
    secondary_bundle,
    mapper,
) -> Tuple[np.ndarray, str, float, float]:
    if str(domain) == "primary":
        z0_primary = _project_feature_to_latent(x0_feat, primary_bundle)
        z0_secondary = _map_with_mapper(mapper, z0_primary, reverse=False)
        primary_anchor = _mean_feature_subset(x0_feat, primary_idx)
        secondary_anchor = _mean_feature_subset(
            _reconstruct_feature_subset(
                z0_secondary,
                np.asarray(secondary_bundle.reconstruction, dtype=np.float32),
                np.asarray(secondary_bundle.mean, dtype=np.float32),
                secondary_idx,
            ),
            range(len(secondary_idx)),
        )
        mapping_chain = "primary_feature->primary_latent->ode->secondary_latent"
    else:
        z0_secondary = _project_feature_to_latent(x0_feat, secondary_bundle)
        z0_primary = _map_with_mapper(mapper, z0_secondary, reverse=True)
        secondary_anchor = _mean_feature_subset(x0_feat, secondary_idx)
        primary_anchor = _mean_feature_subset(
            _reconstruct_feature_subset(
                z0_primary,
                np.asarray(primary_bundle.reconstruction, dtype=np.float32),
                np.asarray(primary_bundle.mean, dtype=np.float32),
                primary_idx,
            ),
            range(len(primary_idx)),
        )
        mapping_chain = "secondary_feature->secondary_latent->primary_latent(T_rev)->ode->secondary_latent(T)"
    return np.asarray(z0_primary, dtype=np.float32), str(mapping_chain), float(primary_anchor), float(secondary_anchor)


def _inject_pair_curve_anchor(
    pair_df: pd.DataFrame,
    *,
    primary_anchor: float,
    secondary_anchor: float,
) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df
    out = pair_df.copy()
    first_idx = out.index[0]
    out.loc[first_idx, "primary_value"] = float(primary_anchor)
    out.loc[first_idx, "secondary_value"] = float(secondary_anchor)
    return out


def _t0_gap(sim_df: pd.DataFrame, real_df: pd.DataFrame, column: str) -> float:
    if sim_df.empty or real_df.empty or column not in sim_df.columns or column not in real_df.columns:
        return 0.0
    return float(sim_df.iloc[0][column] - real_df.iloc[0][column])


def _parse_target_genes(raw: Any) -> List[str]:
    return [x.strip() for x in str(raw).split("|") if x.strip()]


def _auto_select_pair(
    ctx: TFRunContext,
    target_genes: Sequence[str],
    primary_var: str = "",
    secondary_var: str = "",
) -> Tuple[str, str, str]:
    p = str(primary_var).strip()
    s = str(secondary_var).strip()
    source = "manual"
    pair_csv = ctx.output_table_dir / "tf_pairs_final.csv"
    if pair_csv.exists():
        df = pd.read_csv(pair_csv)
    else:
        df = pd.DataFrame(columns=["primary_var", "secondary_var", "score"])

    if p and s:
        return p, s, source

    if not df.empty and {"primary_var", "secondary_var"}.issubset(df.columns):
        work = df.copy()
        if "score" in work.columns:
            work["score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0.0)
        else:
            work["score"] = 0.0
        if p:
            work = work[work["primary_var"].astype(str) == p].copy()
        if s:
            work = work[work["secondary_var"].astype(str) == s].copy()
        if (not p) and target_genes:
            gene_set = {str(x) for x in target_genes}
            sub = work[work["primary_var"].astype(str).isin(gene_set)].copy()
            if not sub.empty:
                work = sub
        if not work.empty:
            top = work.sort_values("score", ascending=False).iloc[0]
            p = p or str(top["primary_var"])
            s = s or str(top["secondary_var"])
            return p, s, "tf_pairs_final"

    primary_bundle, secondary_bundle = ctx.load_primary_secondary_bundles()
    if not p:
        p = str(target_genes[0]) if target_genes else str(primary_bundle.var_names[0])
    if not s:
        s = str(secondary_bundle.var_names[0])
    return p, s, "fallback"


def _pick_main_gene(main_gene: str, target_genes: Sequence[str], primary_var: str) -> str:
    g = str(main_gene).strip()
    if g:
        return g
    if target_genes:
        return str(target_genes[0])
    return str(primary_var)


def _get_var_index(var_names: Sequence[str], var_name: str, *, label: str) -> int:
    lut = {str(v): i for i, v in enumerate(var_names)}
    idx = lut.get(str(var_name), None)
    if idx is None:
        raise ValueError(f"{label} variable not found: {var_name}")
    return int(idx)


def _feature_column_from_latent(latent: np.ndarray, reconstruction: np.ndarray, mean: np.ndarray, idx: int) -> np.ndarray:
    return np.asarray(latent @ reconstruction[:, int(idx)] + mean[int(idx)], dtype=float)


def _build_sim_curve(
    latent_series: np.ndarray,
    time_grid: np.ndarray,
    reconstruction: np.ndarray,
    mean: np.ndarray,
    var_idx: int,
    weights_by_time: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    t_num = int(latent_series.shape[0])
    for i in range(t_num):
        vals = _feature_column_from_latent(np.asarray(latent_series[i], dtype=float), reconstruction, mean, int(var_idx))
        wt = None if weights_by_time is None else np.asarray(weights_by_time[i], dtype=float)
        rows.append({"time": float(time_grid[i]), "value": float(_weighted_mean(vals, wt))})
    return pd.DataFrame(rows).sort_values("time")


def _build_real_curve(
    adata,
    time_key: str,
    reconstruction: np.ndarray,
    mean: np.ndarray,
    var_idx: int,
) -> pd.DataFrame:
    if str(time_key) not in adata.obs.columns:
        raise ValueError(f"time_key not found in processed adata.obs: {time_key}")
    latent = np.asarray(adata.obsm["X_latent"], dtype=float)
    vals = _feature_column_from_latent(latent, reconstruction, mean, int(var_idx))
    times = np.asarray(adata.obs[str(time_key)], dtype=float)
    out = pd.DataFrame({"time": times, "value": vals})
    out = out.groupby("time", as_index=False)["value"].mean().sort_values("time")
    return out


def _load_or_make_indices(ctx: TFRunContext, cfg: Dict[str, Any]) -> Tuple[Path, pd.DataFrame]:
    indices_csv = str(cfg.get("indices_csv", "")).strip()
    label_rule = str(cfg.get("indices_label_rule", cfg.get("label_rule", cfg.get("selection_mode", "random")))).strip() or "random"
    cell_type_merge_cfg = cfg.get("cell_type_merge", {})
    if not isinstance(cell_type_merge_cfg, dict):
        cell_type_merge_cfg = {}
    request_sig = _build_indices_request_signature(
        ctx,
        cfg,
        label_rule=str(label_rule),
        cell_type_merge_cfg=cell_type_merge_cfg,
    )
    explicit_path = Path(indices_csv).resolve() if indices_csv else None
    allow_explicit = _indices_csv_is_authoritative(cfg)
    path = _select_cached_indices_path(
        ctx,
        explicit_path=explicit_path,
        allow_explicit=bool(allow_explicit),
        request_sig=request_sig,
    )
    need_regen = bool(path is None or (not path.exists()))

    if need_regen:
        made = extract_perturbation_indices(
            ctx,
            target_group=str(cfg.get("target_group", "auto")),
            z_score=str(cfg.get("z_score", "auto")),
            time_point=str(cfg.get("time_point", "auto")),
            cell_type=str(cfg.get("cell_type", "auto")),
            label_rule=str(label_rule),
            device=str(cfg.get("device", "cpu")),
            output_csv="",
            source_cell_type_key=str(cfg.get("source_cell_type_key", "")),
            merge_cell_types=cell_type_merge_cfg,
        )
        path = Path(str(made["output_csv"])).resolve()

    df = pd.read_csv(path)
    if df.empty and bool(cfg.get("retry_auto_on_empty", True)):
        made = extract_perturbation_indices(
            ctx,
            target_group="auto",
            z_score="auto",
            time_point="auto",
            cell_type="auto",
            label_rule=str(label_rule),
            device=str(cfg.get("device", "cpu")),
            output_csv="",
            source_cell_type_key=str(cfg.get("source_cell_type_key", "")),
            merge_cell_types=cell_type_merge_cfg,
        )
        path = Path(str(made["output_csv"])).resolve()
        df = pd.read_csv(path)
    return path, df


def _run_perturbation_pipeline_batch(
    ctx: TFRunContext,
    cfg_obj: Dict[str, Any],
    *,
    out_table_dir: Path,
    out_fig_dir: Path,
    perturb_domain: str = "primary",
) -> Dict[str, Any]:
    domain = _normalize_perturb_domain(perturb_domain)
    if domain == "both":
        raise ValueError("batch runner requires perturb_domain=primary|secondary")

    device = str(cfg_obj.get("device", "cpu")).strip() or "cpu"
    sample_n_cfg = cfg_obj.get("sample_n", "auto")
    seed = int(cfg_obj.get("seed", 42))
    line_dt = float(cfg_obj.get("line_dt", 0.05))
    dense_points_between = int(cfg_obj.get("dense_points_between", 0))
    dense_target_points = int(cfg_obj.get("dense_target_points", 0))
    dense_keep_real_threshold = int(cfg_obj.get("dense_keep_real_threshold", 10))
    aggregation_method_requested = str(cfg_obj.get("aggregation_method", "raw_mean")).strip().lower() or "raw_mean"
    aggregation_method = "raw_mean"
    zscore_before_mean = False

    idx_path, idx_df = _load_or_make_indices(ctx, cfg_obj)
    primary_proc, secondary_proc = ctx.load_primary_secondary_processed(backed=False)
    primary_bundle, secondary_bundle = ctx.load_primary_secondary_bundles()
    model = ctx.load_model()
    mapper = ctx.build_mapper(device=device)
    dyn_adata = ctx.load_dynamic_adata()
    ode_dt = float(cfg_obj.get("ode_dt", ctx.manifest.get("dt", 0.05)))

    source_proc = primary_proc if domain == "primary" else secondary_proc
    source_bundle = primary_bundle if domain == "primary" else secondary_bundle
    source_pool_mode = str(cfg_obj.get("source_pool_mode", "time0_random")).strip().lower() or "time0_random"
    if idx_df.empty and _is_indices_source_pool_mode(source_pool_mode):
        raise ValueError(f"indices csv is empty: {idx_path}")
    source_idx_all, source_pool_filter, source_pool_time_value = _resolve_source_pool(
        source_proc=source_proc,
        idx_df=idx_df,
        time_key=ctx.time_key,
        source_pool_mode=source_pool_mode,
        domain=domain,
    )
    source_idx, n_take, auto_sample_n, sample_policy, time0_real_n, time0_value = _sample_source_indices(
        source_proc=source_proc,
        time_key=ctx.time_key,
        source_idx_all=source_idx_all,
        sample_n_cfg=sample_n_cfg,
        seed=seed,
    )
    target_group = _resolve_target_group(cfg_obj, idx_df, source_pool_mode)

    tasks = _build_batch_tasks(cfg_obj)
    if not tasks:
        raise ValueError("batch_mode enabled but no perturb tasks found (set task_template=biolist or provide perturb_tasks).")
    scores_use = _parse_perturb_scores(cfg_obj)
    if not scores_use:
        raise ValueError("empty perturb scores in batch mode")

    x0_latent = np.asarray(source_proc.obsm["X_latent"][source_idx], dtype=np.float32)
    x0_feat_base = _reconstruct_latent_to_feature(x0_latent, source_bundle).astype(np.float32, copy=True)
    source_feat_std = np.asarray(np.std(x0_feat_base, axis=0), dtype=np.float32)

    primary_lut = {str(v): i for i, v in enumerate(primary_bundle.var_names)}
    secondary_lut = {str(v): i for i, v in enumerate(secondary_bundle.var_names)}

    dense_time, real_time_points, resolved_points_between = _resolve_dense_time_plan(
        primary_proc=primary_proc,
        time_key=ctx.time_key,
        ctx_time_grid=ctx.time_grid,
        line_dt=float(line_dt),
        dense_points_between=int(dense_points_between),
        dense_target_points=int(dense_target_points),
        dense_keep_real_threshold=int(dense_keep_real_threshold),
    )

    batch_dir = out_table_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    task_rows: List[Dict[str, Any]] = []
    all_sim_rows: List[Dict[str, Any]] = []
    all_real_rows: List[Dict[str, Any]] = []

    representative_pair_sim_csv = ""
    representative_pair_real_csv = ""
    representative_primary_label = ""
    representative_secondary_label = ""
    representative_main_gene = ""
    representative_task_id = ""
    
    for i, task in enumerate(tasks):
        task_id = str(task.get("task_id", f"task_{i+1}")).strip() or f"task_{i+1}"
        tier = str(task.get("tier", "")).strip()
        label = str(task.get("label", task_id)).strip() or task_id
        req_primary = _extract_list(task.get("primary_vars", task.get("genes", [])))
        req_secondary = _extract_list(task.get("secondary_vars", task.get("proteins", [])))

        primary_used, primary_missing = _resolve_names_with_aliases(req_primary, var_names=primary_bundle.var_names, alias_map=None)
        secondary_used, secondary_missing = _resolve_names_with_aliases(
            req_secondary,
            var_names=secondary_bundle.var_names,
            alias_map=PROTEIN_NAME_ALIASES,
        )
        
        primary_idx = [int(primary_lut[g]) for g in primary_used if g in primary_lut]
        secondary_idx = [int(secondary_lut[p]) for p in secondary_used if p in secondary_lut]

        task_payload: Dict[str, Any] = {
            "task_id": task_id,
            "tier": tier,
            "label": label,
            "perturb_domain": domain,
            "primary_vars_requested": req_primary,
            "secondary_vars_requested": req_secondary,
            "primary_vars_used": primary_used,
            "secondary_vars_used": secondary_used,
            "primary_vars_missing": primary_missing,
            "secondary_vars_missing": secondary_missing,
            "genes_requested": req_primary,
            "proteins_requested": req_secondary,
            "genes_used": primary_used,
            "proteins_used": secondary_used,
            "genes_missing": primary_missing,
            "proteins_missing": secondary_missing,
            "status": "pending",
            "reason": "",
            "outputs": {},
            "curve_semantics": "raw_mean",
        }

        if len(primary_idx) == 0:
            task_payload["status"] = "skipped"
            task_payload["reason"] = "no valid primary vars in primary space"
            task_rows.append(task_payload)
            continue
        if len(secondary_idx) == 0:
            task_payload["status"] = "skipped"
            task_payload["reason"] = "no valid secondary vars in secondary space"
            task_rows.append(task_payload)
            continue

        if domain == "primary":
            perturb_requested = list(primary_used)
            perturb_idx = [int(i0) for i0 in primary_idx if float(source_feat_std[int(i0)]) >= 1e-8]

            perturb_missing_std = [str(primary_used[j]) for j, i0 in enumerate(primary_idx) if float(source_feat_std[int(i0)]) < 1e-8]
            perturb_used = [str(primary_used[j]) for j, i0 in enumerate(primary_idx) if float(source_feat_std[int(i0)]) >= 1e-8]
        else:
            perturb_requested = list(secondary_used)
            perturb_idx = [int(i0) for i0 in secondary_idx if float(source_feat_std[int(i0)]) >= 1e-8]
            perturb_missing_std = [str(secondary_used[j]) for j, i0 in enumerate(secondary_idx) if float(source_feat_std[int(i0)]) < 1e-8]
            perturb_used = [str(secondary_used[j]) for j, i0 in enumerate(secondary_idx) if float(source_feat_std[int(i0)]) >= 1e-8]
        task_payload["perturb_vars_std_too_small"] = perturb_missing_std
        task_payload["perturb_vars_requested"] = list(perturb_requested)
        task_payload["perturb_vars_used"] = list(perturb_used)

        if len(perturb_idx) == 0:
            task_payload["status"] = "skipped"
            task_payload["reason"] = f"all selected {domain} vars have near-zero std in source cells"
            task_rows.append(task_payload)
            continue

        real_primary_mat = _build_real_matrix(
            adata=primary_proc,
            time_key=ctx.time_key,
            reconstruction=np.asarray(primary_bundle.reconstruction, dtype=np.float32),
            mean=np.asarray(primary_bundle.mean, dtype=np.float32),
            var_idx=primary_idx,
        )
        real_secondary_mat = _build_real_matrix(
            adata=secondary_proc,
            time_key=ctx.time_key,
            reconstruction=np.asarray(secondary_bundle.reconstruction, dtype=np.float32),
            mean=np.asarray(secondary_bundle.mean, dtype=np.float32),
            var_idx=secondary_idx,
        )
        real_primary = _aggregate_curve_from_matrix(real_primary_mat, zscore_before_mean=zscore_before_mean)
        real_secondary = _aggregate_curve_from_matrix(real_secondary_mat, zscore_before_mean=zscore_before_mean)
        if real_primary.empty or real_secondary.empty:
            task_payload["status"] = "skipped"
            task_payload["reason"] = "empty real aggregated curves"
            task_rows.append(task_payload)
            continue

        real_df = pd.DataFrame(
            {
                "task_id": task_id,
                "tier": tier,
                "label": label,
                "perturb_domain": domain,
                "time": real_primary["time"].to_numpy(dtype=float),
                "primary_value": real_primary["value"].to_numpy(dtype=float),
                "secondary_value": np.interp(
                    real_primary["time"].to_numpy(dtype=float),
                    real_secondary["time"].to_numpy(dtype=float),
                    real_secondary["value"].to_numpy(dtype=float),
                ),
                "primary_vars": "|".join(primary_used),
                "secondary_vars": "|".join(secondary_used),
                "n_primary_vars": int(len(primary_used)),
                "n_secondary_vars": int(len(secondary_used)),
            }
        )
        task_dir = batch_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        real_csv = task_dir / "real_curve.csv"
        real_df.to_csv(real_csv, index=False)

        sim_csv_by_z: Dict[str, str] = {}
        comparison_by_z: Dict[str, Dict[str, float]] = {}
        task_success = False
        for z in scores_use:
            x0_feat = x0_feat_base.copy()
            for vi in perturb_idx:
                x0_feat[:, int(vi)] = x0_feat[:, int(vi)] + float(z) * float(source_feat_std[int(vi)])

            z0_primary, mapping_chain, anchor_primary, anchor_secondary = _compute_batch_task_anchors(
                domain=domain,
                x0_feat=x0_feat,
                primary_idx=primary_idx,
                secondary_idx=secondary_idx,
                primary_bundle=primary_bundle,
                secondary_bundle=secondary_bundle,
                mapper=mapper,
            )
            traj_primary = _build_ode_trajectory(
                model=model,
                x0=z0_primary,
                times=np.asarray(dense_time, dtype=float),
                adata=dyn_adata,
                dt=float(ode_dt),
                device=str(device),
            )
            traj_secondary = _map_to_secondary(mapper, traj_primary["main_latent"])

            sim_primary_mat = _build_sim_matrix(
                latent_series=np.asarray(traj_primary["main_latent"], dtype=np.float32),
                time_grid=np.asarray(dense_time, dtype=float),
                reconstruction=np.asarray(primary_bundle.reconstruction, dtype=np.float32),
                mean=np.asarray(primary_bundle.mean, dtype=np.float32),
                var_idx=primary_idx,
                weights_by_time=np.asarray(traj_primary.get("weights_abs"), dtype=float),
            )
            sim_secondary_mat = _build_sim_matrix(
                latent_series=np.asarray(traj_secondary["sub_latent"], dtype=np.float32),
                time_grid=np.asarray(dense_time, dtype=float),
                reconstruction=np.asarray(secondary_bundle.reconstruction, dtype=np.float32),
                mean=np.asarray(secondary_bundle.mean, dtype=np.float32),
                var_idx=secondary_idx,
                weights_by_time=np.asarray(traj_primary.get("weights_abs"), dtype=float),
            )
            sim_primary = _aggregate_curve_from_matrix(sim_primary_mat, zscore_before_mean=zscore_before_mean)
            sim_secondary = _aggregate_curve_from_matrix(sim_secondary_mat, zscore_before_mean=zscore_before_mean)

            sim_df = pd.DataFrame(
                {
                    "task_id": task_id,
                    "tier": tier,
                    "label": label,
                    "perturb_domain": domain,
                    "z_score": float(z),
                    "time": sim_primary["time"].to_numpy(dtype=float),
                    "primary_value": sim_primary["value"].to_numpy(dtype=float),
                    "secondary_value": sim_secondary["value"].to_numpy(dtype=float),
                    "primary_vars": "|".join(primary_used),
                    "secondary_vars": "|".join(secondary_used),
                    "n_primary_vars": int(len(primary_used)),
                    "n_secondary_vars": int(len(secondary_used)),
                }
            )
            sim_df = _inject_pair_curve_anchor(
                sim_df,
                primary_anchor=float(anchor_primary),
                secondary_anchor=float(anchor_secondary),
            )
            z_token = _signed_float_token(float(z))
            sim_csv = task_dir / f"sim_curve__z-{z_token}.csv"
            sim_df.to_csv(sim_csv, index=False)
            sim_csv_by_z[f"{float(z):g}"] = str(sim_csv)
            comparison_by_z[f"{float(z):g}"] = {
                "anchor_primary_raw": float(anchor_primary),
                "anchor_secondary_raw": float(anchor_secondary),
                "real_t0_primary": float(real_df.iloc[0]["primary_value"]),
                "real_t0_secondary": float(real_df.iloc[0]["secondary_value"]),
                "t0_gap_primary": _t0_gap(sim_df, real_df, "primary_value"),
                "t0_gap_secondary": _t0_gap(sim_df, real_df, "secondary_value"),
            }
            all_sim_rows.extend(sim_df.to_dict(orient="records"))
            task_success = True

            if (not representative_pair_sim_csv) and np.isclose(float(z), 0.0):
                representative_pair_sim_csv = str(sim_csv)
                representative_pair_real_csv = str(real_csv)
                representative_primary_label = "|".join(primary_used)
                representative_secondary_label = "|".join(secondary_used)
                representative_main_gene = str(primary_used[0]) if primary_used else ""
                representative_task_id = str(task_id)

        task_payload["mapping_chain"] = mapping_chain
        if task_success:
            task_payload["status"] = "success"
            task_payload["outputs"] = {
                "real_csv": str(real_csv),
                "sim_csv_by_z": sim_csv_by_z,
            }
            task_payload["comparisons_by_z"] = comparison_by_z
            all_real_rows.extend(real_df.to_dict(orient="records"))
        else:
            task_payload["status"] = "failed"
            task_payload["reason"] = "failed to generate any simulation curve"
        task_rows.append(task_payload)

    if not representative_pair_sim_csv:
        for row in task_rows:
            if str(row.get("status", "")) != "success":
                continue
            outs = row.get("outputs", {})
            if not isinstance(outs, dict):
                continue
            sim_map = outs.get("sim_csv_by_z", {})
            if not isinstance(sim_map, dict) or not sim_map:
                continue
            key = next(iter(sim_map.keys()))
            representative_pair_sim_csv = str(sim_map[key])
            representative_pair_real_csv = str(outs.get("real_csv", ""))
            representative_primary_label = "|".join([str(x) for x in row.get("primary_vars_used", row.get("genes_used", []))])
            representative_secondary_label = "|".join([str(x) for x in row.get("secondary_vars_used", row.get("proteins_used", []))])
            genes_used_row = row.get("primary_vars_used", row.get("genes_used", []))
            representative_main_gene = str(genes_used_row[0]) if isinstance(genes_used_row, list) and genes_used_row else ""
            representative_task_id = str(row.get("task_id", ""))
            break

    if not representative_pair_sim_csv or not representative_pair_real_csv:
        raise ValueError(f"no runnable batch perturbation tasks for perturb_domain={domain}")

    batch_sim_csv = batch_dir / "perturb_batch_curves_sim.csv"
    batch_real_csv = batch_dir / "perturb_batch_curves_real.csv"
    batch_tasks_json = batch_dir / "perturb_batch_tasks.json"
    pd.DataFrame(all_sim_rows).to_csv(batch_sim_csv, index=False)
    pd.DataFrame(all_real_rows).to_csv(batch_real_csv, index=False)
    with batch_tasks_json.open("w", encoding="utf-8") as f:
        json.dump({"tasks": task_rows}, f, ensure_ascii=False, indent=2)

    rep_sim_df = pd.read_csv(representative_pair_sim_csv)
    rep_real_df = pd.read_csv(representative_pair_real_csv)
    main_sim_csv = out_table_dir / "perturb_main_gene_sim__batch_compat.csv"
    main_real_csv = out_table_dir / "perturb_main_gene_real__batch_compat.csv"
    pd.DataFrame(
        {
            "time": rep_sim_df["time"].to_numpy(dtype=float),
            "main_gene": representative_main_gene or representative_primary_label or "primary_agg",
            "value": rep_sim_df["primary_value"].to_numpy(dtype=float),
        }
    ).to_csv(main_sim_csv, index=False)
    pd.DataFrame(
        {
            "time": rep_real_df["time"].to_numpy(dtype=float),
            "main_gene": representative_main_gene or representative_primary_label or "primary_agg",
            "value": rep_real_df["primary_value"].to_numpy(dtype=float),
        }
    ).to_csv(main_real_csv, index=False)

    source_files_read, source_files_exists = _build_source_files_meta(ctx, idx_path)

    payload: Dict[str, Any] = {
        "run_dir": str(ctx.run_dir),
        "source_result_subdir": str(ctx.run_dir),
        "source_files_read": source_files_read,
        "source_files_exists": source_files_exists,
        "batch_mode": True,
        "perturb_domain": domain,
        "source_sampling_domain": domain,
        "mapping_chain": (
            "primary_feature->primary_latent->ode->secondary_latent"
            if domain == "primary"
            else "secondary_feature->secondary_latent->primary_latent(T_rev)->ode->secondary_latent(T)"
        ),
        "task_template": str(cfg_obj.get("task_template", "")).strip().lower() or ("biolist" if str(cfg_obj.get("task_template", "")).strip() else ""),
        "aggregation_method": aggregation_method,
        "aggregation_method_requested": aggregation_method_requested,
        "curve_semantics": "raw_mean",
        "curve_reference_semantics": {
            "real_curve": "per-time full-population observed mean in each domain",
            "sim_curve": "trajectory mean simulated from time0 source cells under perturbation",
        },
        "z_scores": [float(x) for x in scores_use],
        "sample_n": int(n_take),
        "sample_n_policy": str(sample_policy),
        "time0_real_n": int(time0_real_n),
        "time0_value": float(time0_value),
        "time0_auto_sample_n": int(auto_sample_n),
        "source_pool_mode": str(source_pool_mode),
        "source_pool_filter": str(source_pool_filter),
        "source_pool_time_value": float(source_pool_time_value),
        "source_pool_n": int(source_idx_all.size),
        "sampled_source_indices": [int(x) for x in source_idx.tolist()],
        "line_dt": float(line_dt),
        "dense_points_between": int(dense_points_between),
        "resolved_points_between": int(resolved_points_between),
        "dense_target_points": int(dense_target_points),
        "dense_keep_real_threshold": int(dense_keep_real_threshold),
        "real_time_points": [float(x) for x in real_time_points],
        "dense_time_grid": [float(x) for x in np.asarray(dense_time, dtype=float).tolist()],
        "batch_tasks": task_rows,
        "outputs": {
            "batch_sim_csv": str(batch_sim_csv),
            "batch_real_csv": str(batch_real_csv),
            "batch_tasks_json": str(batch_tasks_json),
            "pair_sim_csv": str(representative_pair_sim_csv),
            "pair_real_csv": str(representative_pair_real_csv),
            "main_sim_csv": str(main_sim_csv),
            "main_real_csv": str(main_real_csv),
        },
        "pair_source": "batch_biolist" if str(cfg_obj.get("task_template", "")).strip().lower() in {"biolist", "bio", "biological", "default", "ho_regulome"} else "batch_custom",
        "primary_var": representative_primary_label or "primary_agg",
        "secondary_var": representative_secondary_label or "secondary_agg",
        "main_gene": representative_main_gene or "primary_agg",
        "representative_task_id": representative_task_id,
        "target_group": target_group,
        "target_genes_requested": [],
        "target_genes_used": [],
        "indices_csv": str(idx_path),
        "requested_indices_csv": str(cfg_obj.get("indices_csv", "")).strip(),
        "indices_usage_mode": "source_pool_filter" if _indices_csv_is_authoritative(cfg_obj) else "advisory_cache",
        "z_score": 0.0,
    }

    manifest_json = _write_perturb_manifest(out_fig_dir, payload, batch_mode=True)
    payload["manifest_json"] = str(manifest_json)
    return payload


def _run_perturbation_pipeline_single_domain(
    ctx: TFRunContext,
    cfg_obj: Dict[str, Any],
    *,
    out_table_dir: Path,
    out_fig_dir: Path,
    perturb_domain: str = "primary",
) -> Dict[str, Any]:
    domain = _normalize_perturb_domain(perturb_domain)
    if domain == "both":
        raise ValueError("single-domain runner requires perturb_domain=primary|secondary")

    device = str(cfg_obj.get("device", "cpu")).strip() or "cpu"
    sample_n_cfg = cfg_obj.get("sample_n", "auto")
    seed = int(cfg_obj.get("seed", 42))
    line_dt = float(cfg_obj.get("line_dt", 0.05))
    dense_points_between = int(cfg_obj.get("dense_points_between", 0))
    dense_target_points = int(cfg_obj.get("dense_target_points", 0))
    dense_keep_real_threshold = int(cfg_obj.get("dense_keep_real_threshold", 10))
    ode_dt = float(cfg_obj.get("ode_dt", ctx.manifest.get("dt", 0.05)))

    idx_path, idx_df = _load_or_make_indices(ctx, cfg_obj)
    if idx_df.empty:
        raise ValueError(f"indices csv is empty: {idx_path}")

    target_genes = _parse_target_genes(idx_df["target_genes"].iloc[0])
    z_used = float(cfg_obj.get("z_score_override", idx_df["z_score"].iloc[0]))

    primary_proc, secondary_proc = ctx.load_primary_secondary_processed(backed=False)
    primary_bundle, secondary_bundle = ctx.load_primary_secondary_bundles()
    source_proc = primary_proc if domain == "primary" else secondary_proc
    source_bundle = primary_bundle if domain == "primary" else secondary_bundle
    source_pool_mode = str(cfg_obj.get("source_pool_mode", "time0_random")).strip().lower() or "time0_random"
    source_idx_all, source_pool_filter, source_pool_time_value = _resolve_source_pool(
        source_proc=source_proc,
        idx_df=idx_df,
        time_key=ctx.time_key,
        source_pool_mode=source_pool_mode,
        domain=domain,
    )
    target_group = _resolve_target_group(cfg_obj, idx_df, source_pool_mode)
    source_idx, n_take, auto_sample_n, sample_policy, time0_real_n, time0_value = _sample_source_indices(
        source_proc=source_proc,
        time_key=ctx.time_key,
        source_idx_all=source_idx_all,
        sample_n_cfg=sample_n_cfg,
        seed=seed,
    )
    
    primary_var, secondary_var, pair_source = _auto_select_pair(
        ctx,
        target_genes=target_genes,
        primary_var=str(cfg_obj.get("primary_var", "")),
        secondary_var=str(cfg_obj.get("secondary_var", "")),
    )
    main_gene = _pick_main_gene(str(cfg_obj.get("main_gene", "")), target_genes, primary_var)

    p_idx = _get_var_index(primary_bundle.var_names, primary_var, label="primary")
    s_idx = _get_var_index(secondary_bundle.var_names, secondary_var, label="secondary")
    try:
        mg_idx = _get_var_index(primary_bundle.var_names, main_gene, label="main_gene")
    except Exception:
        main_gene = str(primary_var)
        mg_idx = _get_var_index(primary_bundle.var_names, main_gene, label="main_gene")

    if domain == "primary":
        req_perturb_vars = _extract_list(cfg_obj.get("primary_vars", cfg_obj.get("genes", target_genes)))
        if not req_perturb_vars:
            req_perturb_vars = list(target_genes) if target_genes else [str(primary_var)]
        perturb_used, perturb_missing = _resolve_names_with_aliases(req_perturb_vars, var_names=primary_bundle.var_names, alias_map=None)
    else:
        req_perturb_vars = _extract_list(cfg_obj.get("secondary_vars", cfg_obj.get("proteins", [secondary_var])))
        if not req_perturb_vars:
            req_perturb_vars = [str(secondary_var)]
        perturb_used, perturb_missing = _resolve_names_with_aliases(
            req_perturb_vars,
            var_names=secondary_bundle.var_names,
            alias_map=PROTEIN_NAME_ALIASES,
        )

    source_lut = {str(v): i for i, v in enumerate(source_bundle.var_names)}
    perturb_idx_all = [int(source_lut[v]) for v in perturb_used if v in source_lut]

    x0_latent = np.asarray(source_proc.obsm["X_latent"][source_idx], dtype=np.float32)
    x0_feat = _reconstruct_latent_to_feature(x0_latent, source_bundle).astype(np.float32, copy=True)
    source_feat_std = np.asarray(np.std(x0_feat, axis=0), dtype=np.float32)

    perturb_idx: List[int] = []
    perturb_std_too_small: List[str] = []
    valid_perturb_vars: List[str] = []
    for var_name, idx in zip(perturb_used, perturb_idx_all):
        sd = float(source_feat_std[int(idx)])
        if sd < 1e-8:
            perturb_std_too_small.append(str(var_name))
            continue
        x0_feat[:, int(idx)] = x0_feat[:, int(idx)] + float(z_used) * sd
        perturb_idx.append(int(idx))
        valid_perturb_vars.append(str(var_name))
    if not perturb_idx:
        raise ValueError(
            f"No valid perturb vars in {domain} space. requested={req_perturb_vars}, "
            f"resolved={perturb_used}, missing={perturb_missing}, std_too_small={perturb_std_too_small}"
        )

    model = ctx.load_model()
    mapper = ctx.build_mapper(device=device)
    dense_time, real_time_points, resolved_points_between = _resolve_dense_time_plan(
        primary_proc=primary_proc,
        time_key=ctx.time_key,
        ctx_time_grid=ctx.time_grid,
        line_dt=float(line_dt),
        dense_points_between=int(dense_points_between),
        dense_target_points=int(dense_target_points),
        dense_keep_real_threshold=int(dense_keep_real_threshold),
    )
    traj_primary, traj_secondary, mapping_chain = _simulate_perturbed_trajectories(
        domain=domain,
        x0_feat=x0_feat,
        primary_bundle=primary_bundle,
        secondary_bundle=secondary_bundle,
        model=model,
        mapper=mapper,
        dyn_adata=ctx.load_dynamic_adata(),
        dense_time=np.asarray(dense_time, dtype=float),
        ode_dt=float(ode_dt),
        device=str(device),
    )

    sim_primary = _build_sim_curve(
        np.asarray(traj_primary["main_latent"], dtype=np.float32),
        np.asarray(dense_time, dtype=float),
        np.asarray(primary_bundle.reconstruction, dtype=np.float32),
        np.asarray(primary_bundle.mean, dtype=np.float32),
        p_idx,
        np.asarray(traj_primary.get("weights_abs"), dtype=float),
    )
    sim_secondary = _build_sim_curve(
        np.asarray(traj_secondary["sub_latent"], dtype=np.float32),
        np.asarray(dense_time, dtype=float),
        np.asarray(secondary_bundle.reconstruction, dtype=np.float32),
        np.asarray(secondary_bundle.mean, dtype=np.float32),
        s_idx,
        np.asarray(traj_primary.get("weights_abs"), dtype=float),
    )
    sim_main_gene = _build_sim_curve(
        np.asarray(traj_primary["main_latent"], dtype=np.float32),
        np.asarray(dense_time, dtype=float),
        np.asarray(primary_bundle.reconstruction, dtype=np.float32),
        np.asarray(primary_bundle.mean, dtype=np.float32),
        mg_idx,
        np.asarray(traj_primary.get("weights_abs"), dtype=float),
    )

    real_primary = _build_real_curve(
        primary_proc,
        ctx.time_key,
        np.asarray(primary_bundle.reconstruction, dtype=np.float32),
        np.asarray(primary_bundle.mean, dtype=np.float32),
        p_idx,
    )
    real_secondary = _build_real_curve(
        secondary_proc,
        ctx.time_key,
        np.asarray(secondary_bundle.reconstruction, dtype=np.float32),
        np.asarray(secondary_bundle.mean, dtype=np.float32),
        s_idx,
    )
    real_main_gene = _build_real_curve(
        primary_proc,
        ctx.time_key,
        np.asarray(primary_bundle.reconstruction, dtype=np.float32),
        np.asarray(primary_bundle.mean, dtype=np.float32),
        mg_idx,
    )

    pair_slug = (
        f"p-{_slug(primary_var)}__s-{_slug(secondary_var)}"
        f"__g-{_slug(target_group)}__z-{_signed_float_token(float(z_used))}"
    )
    main_slug = f"g-{_slug(main_gene)}__grp-{_slug(target_group)}__z-{_signed_float_token(float(z_used))}"

    pair_sim_csv = out_table_dir / f"perturb_pair_curve_sim__{pair_slug}.csv"
    pair_real_csv = out_table_dir / f"perturb_pair_curve_real__{pair_slug}.csv"
    main_sim_csv = out_table_dir / f"perturb_main_gene_sim__{main_slug}.csv"
    main_real_csv = out_table_dir / f"perturb_main_gene_real__{main_slug}.csv"

    pair_sim_df = pd.DataFrame(
        {
            "time": sim_primary["time"].to_numpy(dtype=float),
            "primary_var": str(primary_var),
            "secondary_var": str(secondary_var),
            "primary_value": sim_primary["value"].to_numpy(dtype=float),
            "secondary_value": sim_secondary["value"].to_numpy(dtype=float),
            "perturb_domain": domain,
        }
    )
    pair_real_df = pd.DataFrame(
        {
            "time": real_primary["time"].to_numpy(dtype=float),
            "primary_var": str(primary_var),
            "secondary_var": str(secondary_var),
            "primary_value": real_primary["value"].to_numpy(dtype=float),
            "secondary_value": np.interp(
                real_primary["time"].to_numpy(dtype=float),
                real_secondary["time"].to_numpy(dtype=float),
                real_secondary["value"].to_numpy(dtype=float),
            ),
            "perturb_domain": domain,
        }
    )
    main_sim_df = pd.DataFrame({"time": sim_main_gene["time"], "main_gene": str(main_gene), "value": sim_main_gene["value"]})
    main_real_df = pd.DataFrame({"time": real_main_gene["time"], "main_gene": str(main_gene), "value": real_main_gene["value"]})

    pair_sim_df.to_csv(pair_sim_csv, index=False)
    pair_real_df.to_csv(pair_real_csv, index=False)
    main_sim_df.to_csv(main_sim_csv, index=False)
    main_real_df.to_csv(main_real_csv, index=False)

    source_files_read, source_files_exists = _build_source_files_meta(ctx, idx_path)

    payload = {
        "run_dir": str(ctx.run_dir),
        "source_result_subdir": str(ctx.run_dir),
        "source_files_read": source_files_read,
        "source_files_exists": source_files_exists,
        "indices_csv": str(idx_path),
        "target_group": str(target_group),
        "target_genes_requested": target_genes,
        "target_genes_used": valid_perturb_vars if domain == "primary" else [],
        "z_score": float(z_used),
        "sample_n": int(n_take),
        "sample_n_policy": str(sample_policy),
        "time0_real_n": int(time0_real_n),
        "time0_value": float(time0_value),
        "time0_auto_sample_n": int(auto_sample_n),
        "source_pool_mode": str(source_pool_mode),
        "source_pool_filter": str(source_pool_filter),
        "source_pool_time_value": float(source_pool_time_value),
        "source_pool_n": int(source_idx_all.size),
        "sampled_source_indices": [int(x) for x in source_idx.tolist()],
        "pair_source": str(pair_source),
        "primary_var": str(primary_var),
        "secondary_var": str(secondary_var),
        "main_gene": str(main_gene),
        "perturb_domain": domain,
        "source_sampling_domain": domain,
        "mapping_chain": mapping_chain,
        "perturb_vars_requested": req_perturb_vars,
        "perturb_vars_used": valid_perturb_vars,
        "perturb_vars_missing": perturb_missing,
        "perturb_vars_std_too_small": perturb_std_too_small,
        "curve_reference_semantics": {
            "real_curve": "per-time full-population observed mean in each domain",
            "sim_curve": "trajectory mean simulated from time0 source cells under perturbation",
        },
        "line_dt": float(line_dt),
        "dense_points_between": int(dense_points_between),
        "resolved_points_between": int(resolved_points_between),
        "dense_target_points": int(dense_target_points),
        "dense_keep_real_threshold": int(dense_keep_real_threshold),
        "real_time_points": [float(x) for x in real_time_points],
        "dense_time_grid": [float(x) for x in dense_time.tolist()],
        "outputs": {
            "pair_sim_csv": str(pair_sim_csv),
            "pair_real_csv": str(pair_real_csv),
            "main_sim_csv": str(main_sim_csv),
            "main_real_csv": str(main_real_csv),
        },
        "requested_indices_csv": str(cfg_obj.get("indices_csv", "")).strip(),
        "indices_usage_mode": "source_pool_filter" if _indices_csv_is_authoritative(cfg_obj) else "advisory_cache",
        "batch_mode": False,
    }
    manifest_json = _write_perturb_manifest(out_fig_dir, payload, batch_mode=False)
    payload["manifest_json"] = str(manifest_json)
    return payload


def run_perturbation_pipeline_for_run(run_dir: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg_obj = dict(cfg or {})
    ctx = TFRunContext.from_run_dir(run_dir)

    out_table_dir = (
        Path(str(cfg_obj.get("output_table_dir", ""))).resolve()
        if str(cfg_obj.get("output_table_dir", "")).strip()
        else (ctx.output_table_dir / "perturb_pipeline")
    )
    out_fig_dir = (
        Path(str(cfg_obj.get("output_figure_dir", ""))).resolve()
        if str(cfg_obj.get("output_figure_dir", "")).strip()
        else (ctx.output_figure_dir / "perturb_pipeline")
    )
    out_table_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    requested_domain = _normalize_perturb_domain(cfg_obj.get("perturb_domain", "primary"))
    domains = ["primary", "secondary"] if requested_domain == "both" else [requested_domain]
    is_batch = _is_batch_enabled(cfg_obj)

    domain_payloads: Dict[str, Dict[str, Any]] = {}
    domain_errors: Dict[str, str] = {}
    for domain in domains:
        domain_cfg = dict(cfg_obj)
        domain_cfg["perturb_domain"] = domain
        domain_out_table = out_table_dir / domain
        domain_out_fig = out_fig_dir / domain
        domain_out_table.mkdir(parents=True, exist_ok=True)
        domain_out_fig.mkdir(parents=True, exist_ok=True)
        try:
            if is_batch:
                payload = _run_perturbation_pipeline_batch(
                    ctx,
                    domain_cfg,
                    out_table_dir=domain_out_table,
                    out_fig_dir=domain_out_fig,
                    perturb_domain=domain,
                )
            else:
                payload = _run_perturbation_pipeline_single_domain(
                    ctx,
                    domain_cfg,
                    out_table_dir=domain_out_table,
                    out_fig_dir=domain_out_fig,
                    perturb_domain=domain,
                )
            domain_payloads[domain] = payload
        except Exception as exc:
            domain_errors[domain] = f"{type(exc).__name__}: {exc}"

    if not domain_payloads:
        ctx.close_backed()
        raise ValueError(
            "All requested perturbation domains failed: "
            + "; ".join([f"{k}={v}" for k, v in sorted(domain_errors.items())])
        )

    preferred_domain = "primary" if "primary" in domain_payloads else next(iter(domain_payloads.keys()))
    payload = dict(domain_payloads[preferred_domain])
    payload["selected_domain"] = preferred_domain
    payload["perturb_domain"] = "both" if requested_domain == "both" else preferred_domain
    payload["batch_mode"] = bool(is_batch)
    payload["domain_manifests"] = {k: str(v.get("manifest_json", "")) for k, v in domain_payloads.items()}
    payload["domain_outputs"] = {k: dict(v.get("outputs", {})) for k, v in domain_payloads.items()}
    payload["domain_errors"] = domain_errors
    payload["source_result_subdir"] = str(ctx.run_dir)

    root_manifest = _write_perturb_manifest(out_fig_dir, payload, batch_mode=bool(is_batch))
    payload["manifest_json"] = str(root_manifest)

    ctx.close_backed()
    return payload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# Nature 风格全局配置
# ==========================================
plt.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#111111",
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.labelcolor": "#111111",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 300
})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# Nature 风格全局配置
# ==========================================
plt.rcParams.update({
    "font.family": "Arial",
    "axes.linewidth": 1.0,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.grid": False, # 默认关闭，手动开启
})

# 专业配色
C_PRIMARY = "#2E5B88"    # 深海蓝
C_SECONDARY = "#C0392B"  # 砖红色
C_REAL_P = "#1B4F72"     
C_REAL_S = "#7B241C"     

def _save_dual_format(fig: plt.Figure, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")

def _plot_pair_overlay(
    pair_sim: pd.DataFrame,
    pair_real: pd.DataFrame,
    *,
    output_png: Path,
    primary_var: str,
    secondary_var: str,
    y_mode: str = "zscore",
    y_label_override: str = "",
    left_is_primary: bool = True,
    left_axis_tag: str = "1",
    right_axis_tag: str = "2",
) -> None:
    # 1. 数据处理
    x_sim = pair_sim["time"].to_numpy(dtype=float)
    p_sim, s_sim = pair_sim["primary_value"].to_numpy(), pair_sim["secondary_value"].to_numpy()
    x_real = pair_real["time"].to_numpy(dtype=float)
    p_real, s_real = pair_real["primary_value"].to_numpy(), pair_real["secondary_value"].to_numpy()

    if str(y_mode).strip().lower() == "zscore":
        # 假设 _zscore 函数在作用域内
        p_sim, s_sim, p_real, s_real = _zscore(p_sim), _zscore(s_sim), _zscore(p_real), _zscore(s_real)
        y_lab_base = "Z-score"
    else:
        y_lab_base = "Value"

    # 2. 左右分配逻辑
    if left_is_primary:
        l_sim, r_sim, l_real, r_real = p_sim, s_sim, p_real, s_real
        l_var, r_var, l_tag, r_tag = primary_var, secondary_var, left_axis_tag, right_axis_tag
    else:
        l_sim, r_sim, l_real, r_real = s_sim, p_sim, s_real, p_real
        l_var, r_var, l_tag, r_tag = secondary_var, primary_var, right_axis_tag, left_axis_tag

    fig, ax_left = plt.subplots(figsize=(6, 4))
    ax_right = ax_left.twinx()
    
    # 3. 核心绘图 (线宽加大，Nature 风格)
    h1 = ax_left.plot(x_sim, l_sim, color=C_PRIMARY, linewidth=2.5, label=f"Sim D{l_tag}: {l_var}", zorder=3)
    h2 = ax_right.plot(x_sim, r_sim, color=C_SECONDARY, linewidth=2.5, linestyle=(0, (5, 2)), label=f"Sim D{r_tag}: {r_var}", zorder=2)
    h3 = ax_left.scatter(x_real, l_real, color=C_REAL_P, marker="o", s=60, edgecolors='white', linewidths=0.8, label=f"Obs D{l_tag}", zorder=5)
    h4 = ax_right.scatter(x_real, r_real, color=C_REAL_S, marker="D", s=45, edgecolors='white', linewidths=0.8, label=f"Obs D{r_tag}", zorder=4)

    # 4. 关键：数轴颜色与可见性设置
    # 左轴 (主空间)
    ax_left.set_ylabel(str(y_label_override).strip() or f"D{l_tag} {l_var} ({y_lab_base})", color=C_PRIMARY, fontweight='bold')
    ax_left.tick_params(axis="y", colors=C_PRIMARY, which='both')
    ax_left.spines['left'].set_visible(True)
    ax_left.spines['left'].set_color(C_PRIMARY)
    ax_left.spines['left'].set_linewidth(1.5)
    
    # 右轴 (次空间)
    ax_right.set_ylabel(f"D{r_tag} {r_var} ({y_lab_base})", color=C_SECONDARY, fontweight='bold')
    ax_right.tick_params(axis="y", colors=C_SECONDARY, which='both')
    ax_right.spines['right'].set_visible(True)
    ax_right.spines['right'].set_color(C_SECONDARY)
    ax_right.spines['right'].set_linewidth(1.5)
    
    # 顶部和底部
    ax_left.spines['top'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_left.set_xlabel("Time", fontweight='bold')

    # 背景网格 (仅保留左轴对应的水平虚线)
    ax_left.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 图例合并
    lns = h1 + h2 + [h3, h4]
    ax_left.legend(lns, [l.get_label() for l in lns], loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)

    _save_dual_format(fig, output_png)
    plt.close(fig)


def _plot_pair_overlay_multi_scores(
    sim_by_score: Dict[float, pd.DataFrame],
    real_df: pd.DataFrame,
    *,
    output_png: Path,
    primary_var: str,
    secondary_var: str,
    title_suffix: str = "",
    y_label: str = "value (z-score)",
    left_is_primary: bool = True,
    left_axis_tag: str = "1",
    right_axis_tag: str = "2",
) -> None:
    # 1. 渐变色准备
    scores = sorted(sim_by_score.keys())
    import matplotlib.cm as cm
    # 主色调渐变
    p_colors = [cm.Blues(i) for i in np.linspace(0.5, 0.9, len(scores))]
    s_colors = [cm.Reds(i) for i in np.linspace(0.5, 0.9, len(scores))]
    
    c_left_main = p_colors[-1] if left_is_primary else s_colors[-1]
    c_right_main = s_colors[-1] if left_is_primary else p_colors[-1]

    fig, ax_left = plt.subplots(figsize=(7, 5))
    ax_right = ax_left.twinx()
    
    # 2. 曲线绘制
    handles_sim = []
    secondary_zero_handle = None
    for i, z in enumerate(scores):
        df = sim_by_score[z]
        l_val, r_val = (df["primary_value"], df["secondary_value"]) if left_is_primary else (df["secondary_value"], df["primary_value"])
        
        left_name = primary_var if left_is_primary else secondary_var
        line_l = ax_left.plot(df["time"], l_val, color=p_colors[i] if left_is_primary else s_colors[i], 
                              linewidth=2.0, alpha=0.8, label=f"Sim {left_name} (z={z:g})")
        line_r = ax_right.plot(
            df["time"], r_val,
            color=s_colors[i] if left_is_primary else p_colors[i],
            linewidth=2.0, linestyle='--', alpha=0.8,
            label=f"Sim {secondary_var if left_is_primary else primary_var} (z=0)"
            if np.isclose(float(z), 0.0) else "_nolegend_",
        )
        if np.isclose(float(z), 0.0):
            secondary_zero_handle = line_r[0]
        handles_sim += line_l

    # 3. 真实点绘制
    p_real, s_real = real_df["primary_value"], real_df["secondary_value"]
    l_real, r_real = (p_real, s_real) if left_is_primary else (s_real, p_real)
    
    left_name = primary_var if left_is_primary else secondary_var
    right_name = secondary_var if left_is_primary else primary_var
    h_real_l = ax_left.scatter(real_df["time"], l_real, color="#212121", marker="o", s=70, edgecolors='white', zorder=10, label=f"Real {left_name}")
    h_real_r = ax_right.scatter(real_df["time"], r_real, color="#757575", marker="X", s=60, edgecolors='white', zorder=9, label=f"Real {right_name}")

    # 4. 数轴颜色同步显示
    ax_left.set_ylabel(f"D{left_axis_tag} {y_label}", color=c_left_main, fontweight='bold')
    ax_left.tick_params(axis="y", colors=c_left_main)
    ax_left.spines['left'].set_visible(True)
    ax_left.spines['left'].set_color(c_left_main)
    ax_left.spines['left'].set_linewidth(1.5)

    ax_right.set_ylabel(f"D{right_axis_tag} {y_label}", color=c_right_main, fontweight='bold')
    ax_right.tick_params(axis="y", colors=c_right_main)
    ax_right.spines['right'].set_visible(True)
    ax_right.spines['right'].set_color(c_right_main)
    ax_right.spines['right'].set_linewidth(1.5)

    ax_left.spines['top'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_left.set_xlabel("Time", fontweight='bold')
    ax_left.grid(True, axis='y', alpha=0.2)

    legend_handles = handles_sim.copy()
    if secondary_zero_handle is not None:
        legend_handles.append(secondary_zero_handle)
    legend_handles.extend([h_real_l, h_real_r])
    ax_left.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.15, 1.0))

    _save_dual_format(fig, output_png)
    plt.close(fig)


def plot_saved_perturbation_batch_curves(
    table_dir: str | Path,
    output_path: str | Path,
    *,
    scores: Optional[Sequence[float]] = None,
    primary_var: str = "TFRC",
    secondary_var: str = "CD71",
    title_suffix: str = "TFRC->CD71",
    internal_time: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    display_time: Sequence[float] = (2.0, 4.0, 7.0, 10.0),
) -> Dict[str, Any]:
    """Plot archived perturbation tables with the original batch-curve renderer.

    The optional time mapping converts the model's ordinal time coordinate to
    experimentally observed days using piecewise-linear interpolation.  It does
    not reverse or otherwise alter the response values.
    """

    source = Path(table_dir).resolve()
    real_path = source / "real_curve.csv"
    if not real_path.exists():
        raise FileNotFoundError(real_path)
    requested = [float(x) for x in (scores if scores is not None else [-40, -20, 0, 20, 40])]

    x_internal = np.asarray(internal_time, dtype=float)
    x_display = np.asarray(display_time, dtype=float)
    if x_internal.ndim != 1 or x_display.ndim != 1 or x_internal.size != x_display.size:
        raise ValueError("internal_time and display_time must be one-dimensional and equally sized")

    def remap(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["time"] = np.interp(result["time"].to_numpy(dtype=float), x_internal, x_display)
        return result

    real_df = remap(pd.read_csv(real_path))
    sim_by_score: Dict[float, pd.DataFrame] = {}
    source_files: Dict[str, str] = {"real": str(real_path)}
    for z in requested:
        token = f"m{abs(float(z)):g}" if float(z) < 0 else f"{float(z):g}"
        path = source / f"sim_curve__z-{token}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        sim_by_score[float(z)] = remap(pd.read_csv(path))
        source_files[f"z={float(z):g}"] = str(path)

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _plot_pair_overlay_multi_scores(
        sim_by_score,
        real_df,
        output_png=out,
        primary_var=str(primary_var),
        secondary_var=str(secondary_var),
        title_suffix=str(title_suffix),
        y_label="value",
        left_is_primary=True,
        left_axis_tag="RNA",
        right_axis_tag="protein",
    )
    return {
        "output": str(out),
        "scores": requested,
        "time_mapping": dict(zip(x_internal.tolist(), x_display.tolist())),
        "source_files": source_files,
    }
# def _plot_pair_overlay(
#     pair_sim: pd.DataFrame,
#     pair_real: pd.DataFrame,
#     *,
#     output_png: Path,
#     primary_var: str,
#     secondary_var: str,
#     y_mode: str = "zscore",
#     y_label_override: str = "",
#     left_is_primary: bool = True,
#     left_axis_tag: str = "1",
#     right_axis_tag: str = "2",
# ) -> None:
#     x_sim = pair_sim["time"].to_numpy(dtype=float)
#     p_sim = pair_sim["primary_value"].to_numpy(dtype=float)
#     s_sim = pair_sim["secondary_value"].to_numpy(dtype=float)

#     x_real = pair_real["time"].to_numpy(dtype=float)
#     p_real = pair_real["primary_value"].to_numpy(dtype=float)
#     s_real = pair_real["secondary_value"].to_numpy(dtype=float)

#     if str(y_mode).strip().lower() == "zscore":
#         p_sim, s_sim, p_real, s_real = _zscore(p_sim), _zscore(s_sim), _zscore(p_real), _zscore(s_real)
#         y_label = "value (z-score)"
#     else:
#         y_label = "value"
#     if str(y_label_override).strip():
#         y_label = str(y_label_override).strip()

#     plt.style.use("seaborn-v0_8-darkgrid")
#     fig, ax_left = plt.subplots(figsize=(8.2, 5.2))
#     ax_right = ax_left.twinx()
#     if bool(left_is_primary):
#         left_sim, right_sim = p_sim, s_sim
#         left_real, right_real = p_real, s_real
#         left_var, right_var = primary_var, secondary_var
#     else:
#         left_sim, right_sim = s_sim, p_sim
#         left_real, right_real = s_real, p_real
#         left_var, right_var = secondary_var, primary_var

#     h1 = ax_left.plot(x_sim, left_sim, color="#1f77b4", linewidth=2.2, label=f"sim D{left_axis_tag} {left_var}")
#     h2 = ax_right.plot(x_sim, right_sim, color="#ff7f0e", linewidth=2.2, linestyle="--", label=f"sim D{right_axis_tag} {right_var}")
#     h3 = ax_left.scatter(x_real, left_real, color="#1f77b4", marker="X", s=48, label=f"real D{left_axis_tag} {left_var}")
#     h4 = ax_right.scatter(x_real, right_real, color="#ff7f0e", marker="P", s=40, label=f"real D{right_axis_tag} {right_var}")
#     ax_left.set_xlabel("time")
#     if str(y_mode).strip().lower() == "zscore":
#         ax_left.set_ylabel(f"D{left_axis_tag} z-score", color="#1f77b4")
#         ax_right.set_ylabel(f"D{right_axis_tag} z-score", color="#ff7f0e")
#     else:
#         ax_left.set_ylabel(f"D{left_axis_tag} value", color="#1f77b4")
#         ax_right.set_ylabel(f"D{right_axis_tag} value", color="#ff7f0e")
#     if str(y_label_override).strip():
#         ax_left.set_ylabel(str(y_label_override).strip(), color="#1f77b4")
#     ax_left.tick_params(axis="y", colors="#1f77b4")
#     ax_right.tick_params(axis="y", colors="#ff7f0e")
#     ax_left.set_title(f"Perturbation Variable Pair: {primary_var} vs {secondary_var}")
#     handles = h1 + h2 + [h3, h4]
#     labels = [h.get_label() for h in handles]
#     ax_left.legend(handles, labels, frameon=False, ncol=2, fontsize=8)
#     fig.tight_layout()
#     output_png.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)


# def _plot_pair_overlay_multi_scores(
#     sim_by_score: Dict[float, pd.DataFrame],
#     real_df: pd.DataFrame,
#     *,
#     output_png: Path,
#     primary_var: str,
#     secondary_var: str,
#     title_suffix: str = "",
#     y_label: str = "value (z-score mean)",
#     left_is_primary: bool = True,
#     left_axis_tag: str = "1",
#     right_axis_tag: str = "2",
# ) -> None:
#     if not sim_by_score:
#         raise ValueError("sim_by_score is empty")
#     x_real = real_df["time"].to_numpy(dtype=float)
#     p_real = real_df["primary_value"].to_numpy(dtype=float)
#     s_real = real_df["secondary_value"].to_numpy(dtype=float)
#     scores = sorted(sim_by_score.keys(), key=lambda x: float(x))

#     left_var = primary_var if bool(left_is_primary) else secondary_var
#     right_var = secondary_var if bool(left_is_primary) else primary_var

#     plt.style.use("seaborn-v0_8-darkgrid")
#     fig, ax_left = plt.subplots(figsize=(9.2, 5.6))
#     ax_right = ax_left.twinx()
#     cmap = plt.get_cmap("tab10")
#     handles: List[Any] = []
#     for i, z in enumerate(scores):
#         df = sim_by_score[float(z)]
#         x = df["time"].to_numpy(dtype=float)
#         p = df["primary_value"].to_numpy(dtype=float)
#         s = df["secondary_value"].to_numpy(dtype=float)
#         if bool(left_is_primary):
#             left_curve, right_curve = p, s
#         else:
#             left_curve, right_curve = s, p
#         color = cmap(i % 10)
#         handles += ax_left.plot(x, left_curve, color=color, linewidth=2.0, linestyle="-", label=f"sim D{left_axis_tag} z={float(z):g}")
#         handles += ax_right.plot(x, right_curve, color=color, linewidth=2.0, linestyle="--", label=f"sim D{right_axis_tag} z={float(z):g}")
#     if bool(left_is_primary):
#         left_real, right_real = p_real, s_real
#     else:
#         left_real, right_real = s_real, p_real
#     h_real_rna = ax_left.scatter(x_real, left_real, color="#1f77b4", marker="X", s=48, label=f"real D{left_axis_tag} {left_var}")
#     h_real_pro = ax_right.scatter(x_real, right_real, color="#ff7f0e", marker="P", s=40, label=f"real D{right_axis_tag} {right_var}")
#     ax_left.set_xlabel("time")
#     ax_left.set_ylabel(f"D{left_axis_tag} value", color="#1f77b4")
#     ax_right.set_ylabel(f"D{right_axis_tag} value", color="#ff7f0e")
#     ax_left.tick_params(axis="y", colors="#1f77b4")
#     ax_right.tick_params(axis="y", colors="#ff7f0e")
#     suffix = f" | {title_suffix}" if str(title_suffix).strip() else ""
#     ax_left.set_title(f"Perturbation Variable Pair: {primary_var} vs {secondary_var}{suffix}")
#     handles += [h_real_rna, h_real_pro]
#     labels = [h.get_label() for h in handles]
#     ax_left.legend(handles, labels, frameon=False, ncol=2, fontsize=7)
#     fig.tight_layout()
#     output_png.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)


def _plot_main_gene_overlay(
    main_sim: pd.DataFrame,
    main_real: pd.DataFrame,
    *,
    output_png: Path,
    main_gene: str,
    y_mode: str = "zscore",
) -> None:
    x_sim = main_sim["time"].to_numpy(dtype=float)
    y_sim = main_sim["value"].to_numpy(dtype=float)
    x_real = main_real["time"].to_numpy(dtype=float)
    y_real = main_real["value"].to_numpy(dtype=float)

    if str(y_mode).strip().lower() == "zscore":
        y_sim = _zscore(y_sim)
        y_real = _zscore(y_real)
        y_label = "value (z-score)"
    else:
        y_label = "value"

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x_sim, y_sim, color="#1f77b4", linewidth=2.2, label=f"sim {main_gene}")
    ax.scatter(x_real, y_real, color="#d62728", marker="X", s=48, label=f"real {main_gene}")
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)
    ax.set_title(f"Perturbation Main Gene: {main_gene}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_dual_format(fig, output_png)
    plt.close(fig)


def draw_perturbation_figures_for_run(run_dir: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg_obj = dict(cfg or {})
    ctx = TFRunContext.from_run_dir(run_dir)
    out_fig_dir = (
        Path(str(cfg_obj.get("output_figure_dir", ""))).resolve()
        if str(cfg_obj.get("output_figure_dir", "")).strip()
        else (ctx.output_figure_dir / "perturb_pipeline")
    )
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    batch_mode_hint = _is_batch_enabled(cfg_obj)
    manifest_json = (
        Path(str(cfg_obj.get("manifest_json", ""))).resolve()
        if str(cfg_obj.get("manifest_json", "")).strip()
        else _resolve_existing_manifest_path(out_fig_dir, batch_mode=bool(batch_mode_hint), kind="perturb")
    )
    if not manifest_json.exists():
        made = run_perturbation_pipeline_for_run(str(ctx.run_dir), cfg=cfg_obj)
        manifest_json = Path(str(made["manifest_json"])).resolve()

    with manifest_json.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    domain_manifests = manifest.get("domain_manifests", {})
    if (
        isinstance(domain_manifests, dict)
        and domain_manifests
        and (not bool(cfg_obj.get("_skip_domain_dispatch", False)))
    ):
        draw_domain = _normalize_perturb_domain(cfg_obj.get("perturb_domain", cfg_obj.get("draw_domain", "both")))
        requested_domains: List[str]
        if draw_domain == "both":
            requested_domains = [d for d in ["primary", "secondary"] if str(domain_manifests.get(d, "")).strip()]
        else:
            requested_domains = [draw_domain] if str(domain_manifests.get(draw_domain, "")).strip() else []
        if not requested_domains:
            selected = str(manifest.get("selected_domain", "")).strip()
            if selected and str(domain_manifests.get(selected, "")).strip():
                requested_domains = [selected]
            else:
                requested_domains = [d for d, p in domain_manifests.items() if str(p).strip()]

        domain_results: Dict[str, Dict[str, Any]] = {}
        domain_errors: Dict[str, str] = {}
        for domain in requested_domains:
            sub_manifest = Path(str(domain_manifests.get(domain, ""))).resolve()
            if not sub_manifest.exists():
                domain_errors[domain] = f"manifest missing: {sub_manifest}"
                continue
            sub_cfg = dict(cfg_obj)
            sub_cfg["manifest_json"] = str(sub_manifest)
            sub_cfg["output_figure_dir"] = str(out_fig_dir / domain)
            sub_cfg["_skip_domain_dispatch"] = True
            sub_cfg["perturb_domain"] = domain
            try:
                domain_results[domain] = draw_perturbation_figures_for_run(str(ctx.run_dir), cfg=sub_cfg)
            except Exception as exc:
                domain_errors[domain] = f"{type(exc).__name__}: {exc}"

        if not domain_results:
            ctx.close_backed()
            raise ValueError(
                "Failed to draw perturbation figures for requested domains: "
                + "; ".join([f"{k}={v}" for k, v in sorted(domain_errors.items())])
            )

        preferred = str(manifest.get("selected_domain", "")).strip()
        if preferred not in domain_results:
            preferred = "primary" if "primary" in domain_results else next(iter(domain_results.keys()))
        preferred_payload = domain_results[preferred]

        payload = {
            "run_dir": str(ctx.run_dir),
            "source_result_subdir": str(ctx.run_dir),
            "source_manifest_json": str(manifest_json),
            "perturb_domain": str(manifest.get("perturb_domain", "both")),
            "selected_domain": str(preferred),
            "domain_results": {
                d: {
                    "figure_manifest_json": str(v.get("figure_manifest_json", "")),
                    "generated": dict(v.get("generated", {})),
                    "generated_batch_count": int(len(v.get("generated_batch", []))),
                    "generated_extra": dict(v.get("generated_extra", {})) if isinstance(v.get("generated_extra", {}), dict) else {},
                }
                for d, v in domain_results.items()
            },
            "domain_errors": domain_errors,
            "y_mode": "raw",
            "batch_mode": bool(preferred_payload.get("batch_mode", False)),
            "pair_left_domain": str(preferred_payload.get("pair_left_domain", "")),
            "pair_right_domain": str(preferred_payload.get("pair_right_domain", "")),
            "left_is_primary": bool(preferred_payload.get("left_is_primary", True)),
            "generated_batch": preferred_payload.get("generated_batch", []),
            "generated": preferred_payload.get("generated", {}),
            "generated_extra": preferred_payload.get("generated_extra", {}),
            "curve_reference_semantics": preferred_payload.get("curve_reference_semantics", {}),
        }
        fig_manifest = out_fig_dir / _figure_manifest_filename(batch_mode=bool(payload["batch_mode"]))
        with fig_manifest.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        payload["figure_manifest_json"] = str(fig_manifest)
        ctx.close_backed()
        return payload

    outputs = dict(manifest.get("outputs", {})) if isinstance(manifest.get("outputs"), dict) else {}
    primary_var = str(manifest.get("primary_var", "primary"))
    secondary_var = str(manifest.get("secondary_var", "secondary"))
    main_gene = str(manifest.get("main_gene", "gene"))
    y_mode = "raw"
    left_domain_raw = str(cfg_obj.get("pair_left_domain", "1")).strip().lower()
    primary_domain = str(ctx.manifest.get("primary_domain", "")).strip()
    secondary_domain = str(ctx.manifest.get("secondary_domain", "")).strip()
    left_is_primary = True
    if left_domain_raw in {"primary", "pri", "p"}:
        left_is_primary = True
    elif left_domain_raw in {"secondary", "sub", "s"}:
        left_is_primary = False
    elif left_domain_raw in {"1", "2"}:
        if primary_domain and left_domain_raw == primary_domain:
            left_is_primary = True
        elif secondary_domain and left_domain_raw == secondary_domain:
            left_is_primary = False
        elif left_domain_raw == "2":
            left_is_primary = False
    left_axis_tag = str(primary_domain if left_is_primary and primary_domain else (secondary_domain if (not left_is_primary) and secondary_domain else ("1" if left_is_primary else "2")))
    right_axis_tag = str(secondary_domain if left_is_primary and secondary_domain else (primary_domain if (not left_is_primary) and primary_domain else ("2" if left_is_primary else "1")))
    batch_mode = bool(manifest.get("batch_mode", False))

    pair_png = out_fig_dir / (
        f"perturb_pair_curve__p-{_slug(primary_var)}__s-{_slug(secondary_var)}"
        f"__g-{_slug(manifest.get('target_group', 'group'))}.pdf"
    )
    main_png = out_fig_dir / (
        f"perturb_main_gene_curve__g-{_slug(main_gene)}"
        f"__grp-{_slug(manifest.get('target_group', 'group'))}.pdf"
    )

    batch_generated: List[Dict[str, Any]] = []
    if batch_mode:
        plot_mode = str(cfg_obj.get("batch_plot_mode", "both")).strip().lower() or "both"
        if plot_mode not in {"both", "combined", "split"}:
            plot_mode = "both"
        tasks = manifest.get("batch_tasks", [])
        if not isinstance(tasks, list):
            tasks = []
        batch_fig_dir = out_fig_dir / "batch"
        batch_fig_dir.mkdir(parents=True, exist_ok=True)
        for row in tasks:
            if not isinstance(row, dict):
                continue
            if str(row.get("status", "")).strip().lower() != "success":
                continue
            outs = row.get("outputs", {})
            if not isinstance(outs, dict):
                continue
            real_csv = str(outs.get("real_csv", "")).strip()
            sim_map = outs.get("sim_csv_by_z", {})
            if (not real_csv) or (not isinstance(sim_map, dict)) or (not sim_map):
                continue
            real_path = Path(real_csv).resolve()
            if not real_path.exists():
                continue
            real_df = pd.read_csv(real_path)
            sim_by_score: Dict[float, pd.DataFrame] = {}
            for key, path in sim_map.items():
                try:
                    z = float(key)
                except Exception:
                    continue
                p = Path(str(path)).resolve()
                if not p.exists():
                    continue
                sim_by_score[float(z)] = pd.read_csv(p)
            if not sim_by_score:
                continue
            y_label = "value"

            task_id = str(row.get("task_id", "task")).strip() or "task"
            task_label = str(row.get("label", task_id)).strip() or task_id
            primary_label = "|".join([str(x) for x in row.get("genes_used", [])]) or primary_var
            secondary_label = "|".join([str(x) for x in row.get("proteins_used", [])]) or secondary_var
            entry: Dict[str, Any] = {
                "task_id": task_id,
                "label": task_label,
                "tier": str(row.get("tier", "")).strip(),
                "primary_label": primary_label,
                "secondary_label": secondary_label,
                "combined_png": "",
                "split_pngs": {},
                "real_csv": str(real_path),
                "sim_csv_by_z": {f"{float(k):g}": str(Path(str(v)).resolve()) for k, v in sim_map.items()},
                "joint_normalized": False,
                "joint_normalization_stats": {},
            }

            if plot_mode in {"both", "combined"}:
                combined_png = batch_fig_dir / f"perturb_batch_curve__task-{_slug(task_id)}__allz.pdf"
                _plot_pair_overlay_multi_scores(
                    sim_by_score,
                    real_df,
                    output_png=combined_png,
                    primary_var=primary_label,
                    secondary_var=secondary_label,
                    title_suffix=f"{task_label}",
                    y_label=y_label,
                    left_is_primary=left_is_primary,
                    left_axis_tag=left_axis_tag,
                    right_axis_tag=right_axis_tag,
                )
                entry["combined_png"] = str(combined_png)

            if plot_mode in {"both", "split"}:
                split_pngs: Dict[str, str] = {}
                for z in sorted(sim_by_score.keys(), key=lambda x: float(x)):
                    sim_df = sim_by_score[float(z)]
                    split_png = batch_fig_dir / f"perturb_batch_curve__task-{_slug(task_id)}__z-{_signed_float_token(float(z))}.pdf"
                    _plot_pair_overlay(
                        sim_df,
                        real_df,
                        output_png=split_png,
                        primary_var=primary_label,
                        secondary_var=secondary_label,
                        y_mode="raw",
                        y_label_override=y_label,
                        left_is_primary=left_is_primary,
                        left_axis_tag=left_axis_tag,
                        right_axis_tag=right_axis_tag,
                    )
                    split_pngs[f"{float(z):g}"] = str(split_png)
                entry["split_pngs"] = split_pngs

            batch_generated.append(entry)

    pair_sim = pd.read_csv(str(outputs["pair_sim_csv"]))
    pair_real = pd.read_csv(str(outputs["pair_real_csv"]))
    main_sim = pd.read_csv(str(outputs["main_sim_csv"]))
    main_real = pd.read_csv(str(outputs["main_real_csv"]))

    _plot_pair_overlay(
        pair_sim,
        pair_real,
        output_png=pair_png,
        primary_var=primary_var,
        secondary_var=secondary_var,
        y_mode=y_mode,
        left_is_primary=left_is_primary,
        left_axis_tag=left_axis_tag,
        right_axis_tag=right_axis_tag,
    )
    _plot_main_gene_overlay(
        main_sim,
        main_real,
        output_png=main_png,
        main_gene=main_gene,
        y_mode=y_mode,
    )
    fate_heatmap_outputs = _build_fate_heatmap_outputs(ctx, out_fig_dir, cfg_obj)
    variable_heatmap_outputs = _build_variable_time_heatmap_outputs(ctx, out_fig_dir, cfg_obj)
    regulation_umap_outputs = build_regulation_umap_outputs(
        ctx=ctx,
        out_fig_dir=out_fig_dir,
        cfg_obj=cfg_obj,
    )

    payload = {
        "run_dir": str(ctx.run_dir),
        "source_result_subdir": str(ctx.run_dir),
        "source_manifest_json": str(manifest_json),
        "perturb_domain": str(manifest.get("perturb_domain", "primary")),
        "selected_domain": str(manifest.get("selected_domain", manifest.get("perturb_domain", "primary"))),
        "y_mode": str(y_mode),
        "batch_mode": bool(batch_mode),
        "pair_left_domain": str(left_axis_tag),
        "pair_right_domain": str(right_axis_tag),
        "left_is_primary": bool(left_is_primary),
        "generated_batch": batch_generated,
        "generated": {
            "pair_png": str(pair_png),
            "main_gene_png": str(main_png),
        },
        "generated_extra": {
            "fate_heatmap": fate_heatmap_outputs,
            "variable_time_heatmap": variable_heatmap_outputs,
            "regulation_umap": regulation_umap_outputs,
        },
        "curve_reference_semantics": dict(manifest.get("curve_reference_semantics", {})),
    }
    fig_manifest = out_fig_dir / _figure_manifest_filename(batch_mode=bool(batch_mode))
    with fig_manifest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    payload["figure_manifest_json"] = str(fig_manifest)

    ctx.close_backed()
    return payload
