#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.lines import Line2D

# Compatibility for the fitted paper UMAPs, which were serialized with an
# older numba/pynndescent stack.  These aliases preserve the archived model;
# no replacement UMAP is fitted here.
try:  # pragma: no cover - exercised by the archived release assets
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
                    float(np.min(graph_data))
                    if graph_data is not None and len(graph_data)
                    else 0.0
                )
            return _nn_setstate(self, state)

        _legacy_nn_setstate._mofi_legacy_compat = True
        _NNDescent.__setstate__ = _legacy_nn_setstate
except Exception:
    pass

mpl.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("umap-learn is required. Please install umap-learn.") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.Map.tl.transport_factory import build_transport_map
from CytoBridge.tl.analysis_dense_time import (
    _build_classifier_space_latent,
    _build_ode_trajectory,
    _map_to_secondary,
    _resolve_runtime_device,
    build_classifier_bundle,
)
from CytoBridge.utils import load_model_from_adata

EXPECTED_REAL_TIMES = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
DISPLAY_TIMES = np.asarray([0.0, 0.4, 0.7, 1.0, 1.4, 1.7, 2.0, 2.4, 2.7, 3.0], dtype=float)
INTERP_BY_ANCHOR: Dict[float, List[float]] = {
    0.0: [0.4, 0.7],
    1.0: [1.4, 1.7],
    2.0: [2.4, 2.7],
}

  # Example output paths are repository-relative; pass --output-dir explicitly when needed.
RNA_BG_COLOR = "#F1F4F9"
SUB_BG_COLOR = "#F8F5EE"
OTHER_COLOR = "#8D96A0"
CELL_TYPE_ORDER = ["HSC", "EryP", "NeuP", "MasP", "MkP", "MoP"]
CELL_TYPE_COLORS: Dict[str, str] = {
    "HSC": "#88D2C6",
    "EryP": "#F5B39A",
    "NeuP": "#BCC7E5",
    "MasP": "#E8B7DA",
    "MkP": "#B6D86C",
    "MoP": "#E8D162",
    "Other": OTHER_COLOR,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _safe_float_list(values: Sequence[float]) -> List[float]:
    out = sorted({float(v) for v in values})
    if not out:
        raise ValueError("Expected a non-empty time list")
    return out


def _slug_time(value: float) -> str:
    txt = f"{float(value):g}"
    return txt.replace("-", "m").replace(".", "p")


def _is_real_time(value: float) -> bool:
    return bool(np.any(np.isclose(EXPECTED_REAL_TIMES, float(value), atol=1e-8)))


def _mask_for_time(obs_times: np.ndarray, target: float) -> np.ndarray:
    return np.asarray(np.isclose(obs_times, float(target), atol=1e-8), dtype=bool)


def _validate_time_contract(adata: ad.AnnData, time_key: str) -> np.ndarray:
    if time_key not in adata.obs:
        raise KeyError(f"{time_key} not found in adata.obs")
    unique = np.asarray(sorted({float(x) for x in np.asarray(adata.obs[time_key], dtype=float).tolist()}), dtype=float)
    if unique.shape != EXPECTED_REAL_TIMES.shape or not np.allclose(unique, EXPECTED_REAL_TIMES, atol=1e-8):
        raise ValueError(
            f"Expected real time points {EXPECTED_REAL_TIMES.tolist()}, got {unique.tolist()}"
        )
    return unique


def _extract_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")
    required = ["dynamic_adata", "main_processed", "sub_processed", "map_model_path"]
    out: Dict[str, Path] = {}
    for key in required:
        raw = str(paths.get(key, "")).strip()
        if not raw:
            raise ValueError(f"Missing config.paths.{key}")
        out[key] = Path(raw).expanduser().resolve()
    return out


def _extract_output_root(cfg: Dict[str, Any]) -> Path:
    output_cfg = cfg.get("output", {})
    if not isinstance(output_cfg, dict):
        raise ValueError("config.output must be a mapping")
    root_raw = str(output_cfg.get("root_dir", "")).strip()
    if not root_raw:
        raise ValueError("Missing config.output.root_dir")
    return Path(root_raw).expanduser().resolve()


def _resolve_output_dir(cfg: Dict[str, Any], explicit_output_dir: str) -> Path:
    if str(explicit_output_dir).strip():
        out_dir = Path(explicit_output_dir).expanduser().resolve()
    else:
        output_root = _extract_output_root(cfg)
        out_dir = output_root / "time_slice_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_cache_signature(
    *,
    dynamic_adata_path: Path,
    main_processed_path: Path,
    sub_processed_path: Path,
    map_model_path: Path,
    dt: float,
    batch_size: int,
    display_times: Sequence[float],
    device: str,
) -> Dict[str, Any]:
    return {
        "dynamic_adata_path": str(dynamic_adata_path),
        "main_processed_path": str(main_processed_path),
        "sub_processed_path": str(sub_processed_path),
        "map_model_path": str(map_model_path),
        "dt": float(dt),
        "batch_size": int(batch_size),
        "display_times": [float(x) for x in display_times],
        "device": str(device),
    }


def _load_cached_anchor(
    *,
    manifest_path: Path,
    main_npz_path: Path,
    sub_npz_path: Path,
    expected_signature: Dict[str, Any],
    query_times: Sequence[float],
) -> Optional[Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[str, Any]]]:
    if not manifest_path.exists() or not main_npz_path.exists() or not sub_npz_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict):
            return None
        current_signature = manifest.get("cache_signature", {})
        if json.dumps(current_signature, sort_keys=True) != json.dumps(expected_signature, sort_keys=True):
            return None
        main_npz = np.load(main_npz_path, allow_pickle=False)
        sub_npz = np.load(sub_npz_path, allow_pickle=False)
        out_main: Dict[float, np.ndarray] = {}
        out_sub: Dict[float, np.ndarray] = {}
        for t in query_times:
            key = f"t_{_slug_time(float(t))}"
            if key not in main_npz or key not in sub_npz:
                return None
            out_main[float(t)] = np.asarray(main_npz[key], dtype=np.float32)
            out_sub[float(t)] = np.asarray(sub_npz[key], dtype=np.float32)
        return out_main, out_sub, manifest
    except Exception:
        return None


def _save_cached_anchor(
    *,
    manifest_path: Path,
    main_npz_path: Path,
    sub_npz_path: Path,
    cache_signature: Dict[str, Any],
    anchor_time: float,
    source_count: int,
    query_times: Sequence[float],
    main_slices: Dict[float, np.ndarray],
    sub_slices: Dict[float, np.ndarray],
) -> Dict[str, Any]:
    main_payload = {f"t_{_slug_time(float(t))}": np.asarray(main_slices[float(t)], dtype=np.float32) for t in query_times}
    sub_payload = {f"t_{_slug_time(float(t))}": np.asarray(sub_slices[float(t)], dtype=np.float32) for t in query_times}
    np.savez_compressed(main_npz_path, **main_payload)
    np.savez_compressed(sub_npz_path, **sub_payload)
    manifest = {
        "anchor_time": float(anchor_time),
        "source_count": int(source_count),
        "query_times": [float(x) for x in query_times],
        "cache_signature": cache_signature,
        "main_npz_path": str(main_npz_path),
        "sub_npz_path": str(sub_npz_path),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _to_numpy_2d(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return arr


def _map_forward_batched(mapper: Any, x_np: np.ndarray, batch_size: int) -> np.ndarray:
    x = _to_numpy_2d(x_np, name="x_np")
    out: List[np.ndarray] = []
    for start in range(0, x.shape[0], int(batch_size)):
        xb = x[start : start + int(batch_size)]
        yb = mapper.T(xb)
        if isinstance(yb, torch.Tensor):
            arr = yb.detach().cpu().numpy()
        else:
            arr = np.asarray(yb)
        out.append(np.asarray(arr, dtype=np.float32))
    return np.concatenate(out, axis=0) if out else np.zeros((0, x.shape[1]), dtype=np.float32)


def _map_reverse_batched(mapper: Any, y_np: np.ndarray, batch_size: int) -> np.ndarray:
    if not hasattr(mapper, "T_rev"):
        raise ValueError("mapper does not provide T_rev, cannot classify real sub-space slices in joint/main space")
    y = _to_numpy_2d(y_np, name="y_np")
    out: List[np.ndarray] = []
    for start in range(0, y.shape[0], int(batch_size)):
        yb = y[start : start + int(batch_size)]
        xb = mapper.T_rev(yb)
        if isinstance(xb, torch.Tensor):
            arr = xb.detach().cpu().numpy()
        else:
            arr = np.asarray(xb)
        out.append(np.asarray(arr, dtype=np.float32))
    return np.concatenate(out, axis=0) if out else np.zeros((0, y.shape[1]), dtype=np.float32)


def _build_mapper(
    *,
    dynamic_adata: ad.AnnData,
    map_model_path: Path,
    main_dim: int,
    sub_dim: int,
    device: str,
    primary_domain: int = 1,
    secondary_domain: int = 2,
) -> Any:
    multi_cfg = dynamic_adata.uns.get("all_model", {}).get("model_config", {}).get("multi", {})
    hidden_dim = int(multi_cfg.get("hidden_dim", 64))
    mapper_type = str(multi_cfg.get("mapper_type", "ae"))
    mapper_kwargs = dict(multi_cfg.get("mapper_kwargs", {}))
    return build_transport_map(
        map_model_path=str(map_model_path),
        input_dim1=int(main_dim),
        input_dim2=int(sub_dim),
        mode=(int(primary_domain), int(secondary_domain)),
        hidden_dim=int(hidden_dim),
        device=str(device),
        mapper_type=mapper_type,
        mapper_kwargs=mapper_kwargs,
    )


def _discover_existing_classifier(
    *,
    search_root: Path,
    desired_train_space: str,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    candidates = sorted(search_root.rglob("classifier_manifest.json"))
    for manifest_path in candidates:
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                continue
            train_space = str(meta.get("train_space", "joint")).strip().lower()
            if train_space != str(desired_train_space).strip().lower():
                continue
            classifier_path = Path(str(meta.get("classifier_path", ""))).expanduser()
            if not classifier_path.is_absolute():
                classifier_path = (manifest_path.parent / classifier_path).resolve()
            if not classifier_path.exists():
                continue
            with classifier_path.open("rb") as f:
                payload = pickle.load(f)
            classifier = payload.get("classifier", None) if isinstance(payload, dict) else None
            if classifier is None or not hasattr(classifier, "predict"):
                continue
            return classifier, meta
        except Exception:
            continue
    return None


def _train_or_load_classifier(
    *,
    main_proc: ad.AnnData,
    sub_proc: ad.AnnData,
    mapper: Any,
    cell_type_key: str,
    classifier_cfg: Dict[str, Any],
    output_root: Path,
    reports_dir: Path,
) -> Tuple[Any, Dict[str, Any]]:
    desired_train_space = str(classifier_cfg.get("train_space", "joint")).strip().lower() or "joint"
    existing = _discover_existing_classifier(search_root=output_root, desired_train_space=desired_train_space)
    if existing is not None:
        classifier, meta = existing
        meta = dict(meta)
        meta["source"] = "reused_existing"
        return classifier, meta

    bundle, meta = build_classifier_bundle(
        main_proc=main_proc,
        sub_proc=sub_proc,
        mapper=mapper,
        cell_type_key=cell_type_key,
        classifier_cfg=classifier_cfg,
        reports_dir=reports_dir,
    )
    eval_meta = dict(meta)
    eval_meta["source"] = "refit_from_real_data"
    eval_meta["train_space"] = bundle.train_space
    eval_meta["joint_map_batch"] = bundle.map_batch
    return bundle.classifier, eval_meta


def _sample_transform_check(reducer: Any, latent: np.ndarray) -> bool:
    if not hasattr(reducer, "transform"):
        return False
    sample = np.asarray(latent[: min(len(latent), 32)], dtype=np.float32)
    if sample.size == 0:
        return False
    out = reducer.transform(sample)
    arr = np.asarray(out, dtype=float)
    return arr.ndim == 2 and arr.shape[0] == sample.shape[0] and arr.shape[1] == 2


def _score_umap_candidate(path: Path, tokens: Sequence[str]) -> int:
    lower = str(path).lower()
    return sum(3 for token in tokens if token in lower) + (5 if "umap" in lower else 0)


def _discover_existing_umap_reducer(
    *,
    search_roots: Sequence[Path],
    latent_sample: np.ndarray,
    tokens: Sequence[str],
) -> Optional[Tuple[Any, Path]]:
    candidate_paths: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.pkl"):
            candidate_paths.append(path.resolve())
        for path in root.rglob("*.pickle"):
            candidate_paths.append(path.resolve())
    ranked = sorted(set(candidate_paths), key=lambda p: (-_score_umap_candidate(p, tokens), len(str(p))))
    for path in ranked:
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
            reducer = obj.get("reducer", None) if isinstance(obj, dict) else obj
            if reducer is None:
                continue
            if not _sample_transform_check(reducer, latent_sample):
                continue
            return reducer, path
        except Exception:
            continue
    return None


def _resolve_umap(
    *,
    label: str,
    latent_real: np.ndarray,
    output_dir: Path,
    search_roots: Sequence[Path],
    tokens: Sequence[str],
    n_neighbors: int,
    min_dist: float,
    seed: int,
    force_refit: bool,
    explicit_model: Optional[Path] = None,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    latent = np.asarray(latent_real, dtype=np.float32)
    if latent.ndim != 2:
        raise ValueError(f"{label} latent must be 2D, got shape={latent.shape}")

    if not force_refit and explicit_model is not None:
        with explicit_model.open("rb") as f:
            obj = pickle.load(f)
        reducer = obj.get("reducer", None) if isinstance(obj, dict) else obj
        if reducer is None:
            raise ValueError(f"No reducer found in {explicit_model}")
        fitted = getattr(reducer, "embedding_", None)
        if fitted is not None and np.asarray(fitted).shape == (len(latent), 2):
            embedding = np.asarray(fitted, dtype=float)
            coordinate_source = "fitted_embedding"
        else:
            embedding = np.asarray(reducer.transform(latent), dtype=float)
            coordinate_source = "transform"
        return reducer, embedding, {
            "source": "paper_selected",
            "reducer_path": str(explicit_model),
            "coordinate_source": coordinate_source,
            "n_neighbors": getattr(reducer, "n_neighbors", None),
            "min_dist": getattr(reducer, "min_dist", None),
            "random_state": getattr(reducer, "random_state", None),
        }

    if not force_refit:
        existing = _discover_existing_umap_reducer(
            search_roots=search_roots,
            latent_sample=latent,
            tokens=tokens,
        )
        if existing is not None:
            reducer, reducer_path = existing
            embedding = np.asarray(reducer.transform(latent), dtype=float)
            return reducer, embedding, {
                "source": "reused_existing",
                "reducer_path": str(reducer_path),
                "n_neighbors": None,
                "min_dist": None,
                "random_state": None,
            }

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, int(n_neighbors)),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(seed),
    )
    embedding = np.asarray(reducer.fit_transform(latent), dtype=float)
    reducer_path = output_dir / f"{label}_umap_reducer.pkl"
    with reducer_path.open("wb") as f:
        pickle.dump({"reducer": reducer}, f)
    return reducer, embedding, {
        "source": "refit_same_recipe",
        "reducer_path": str(reducer_path),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "random_state": int(seed),
    }


def _transform_2d(reducer: Any, latent: np.ndarray) -> np.ndarray:
    arr = np.asarray(latent, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D latent matrix, got shape={arr.shape}")
    return np.asarray(reducer.transform(arr), dtype=float)


def _simulate_anchor_slices(
    *,
    main_proc: ad.AnnData,
    time_key: str,
    anchor_time: float,
    query_times: Sequence[float],
    model: Any,
    dynamic_adata: ad.AnnData,
    mapper: Any,
    dt: float,
    device: str,
    batch_size: int,
    cache_dir: Path,
    cache_signature: Dict[str, Any],
    force_rerun: bool,
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[str, Any]]:
    manifest_path = cache_dir / f"cache_anchor_t{_slug_time(anchor_time)}_manifest.json"
    main_npz_path = cache_dir / f"cache_anchor_t{_slug_time(anchor_time)}_main.npz"
    sub_npz_path = cache_dir / f"cache_anchor_t{_slug_time(anchor_time)}_sub.npz"
    if not force_rerun:
        cached = _load_cached_anchor(
            manifest_path=manifest_path,
            main_npz_path=main_npz_path,
            sub_npz_path=sub_npz_path,
            expected_signature=cache_signature,
            query_times=query_times,
        )
        if cached is not None:
            return cached

    obs_times = np.asarray(main_proc.obs[time_key], dtype=float)
    mask = _mask_for_time(obs_times, float(anchor_time))
    x0_all = np.asarray(main_proc.obsm["X_latent"][mask], dtype=np.float32)
    if x0_all.ndim != 2 or x0_all.shape[0] == 0:
        raise ValueError(f"No source cells found at anchor time {anchor_time:g}")

    sim_times = [float(anchor_time)] + [float(x) for x in query_times]
    main_chunks: Dict[float, List[np.ndarray]] = {float(t): [] for t in query_times}
    sub_chunks: Dict[float, List[np.ndarray]] = {float(t): [] for t in query_times}
    for start in range(0, x0_all.shape[0], int(batch_size)):
        x0 = x0_all[start : start + int(batch_size)]
        traj_main = _build_ode_trajectory(
            model=model,
            x0=x0,
            times=sim_times,
            adata=dynamic_adata,
            dt=float(dt),
            device=str(device),
        )
        traj_sub = _map_to_secondary(mapper, traj_main["main_latent"])
        for idx, t in enumerate(query_times, start=1):
            main_chunks[float(t)].append(np.asarray(traj_main["main_latent"][idx], dtype=np.float32))
            sub_chunks[float(t)].append(np.asarray(traj_sub["sub_latent"][idx], dtype=np.float32))

    main_slices = {
        float(t): np.concatenate(chunks, axis=0) if chunks else np.zeros((0, x0_all.shape[1]), dtype=np.float32)
        for t, chunks in main_chunks.items()
    }
    sub_slices = {
        float(t): np.concatenate(chunks, axis=0) if chunks else np.zeros((0, x0_all.shape[1]), dtype=np.float32)
        for t, chunks in sub_chunks.items()
    }
    manifest = _save_cached_anchor(
        manifest_path=manifest_path,
        main_npz_path=main_npz_path,
        sub_npz_path=sub_npz_path,
        cache_signature=cache_signature,
        anchor_time=float(anchor_time),
        source_count=int(x0_all.shape[0]),
        query_times=query_times,
        main_slices=main_slices,
        sub_slices=sub_slices,
    )
    return main_slices, sub_slices, manifest


def _subsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if int(max_points) <= 0 or arr.shape[0] <= int(max_points):
        return arr
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(arr.shape[0], size=int(max_points), replace=False))
    return arr[keep]


def _classify_main_slice(
    *,
    classifier: Any,
    train_space: str,
    map_batch: int,
    mapper: Any,
    main_latent: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    main_arr = np.asarray(main_latent, dtype=np.float32)
    sub_arr = _map_forward_batched(mapper, main_arr, batch_size=map_batch)
    X = _build_classifier_space_latent(
        train_space=str(train_space),
        main_latent=main_arr,
        sub_latent=sub_arr,
        mapper=mapper,
        map_batch=int(map_batch),
    )
    pred = np.asarray(classifier.predict(X)).astype(str)
    return pred, sub_arr


def _classify_sub_slice(
    *,
    classifier: Any,
    train_space: str,
    map_batch: int,
    mapper: Any,
    sub_latent: np.ndarray,
    main_latent: Optional[np.ndarray],
) -> np.ndarray:
    sub_arr = np.asarray(sub_latent, dtype=np.float32)
    if main_latent is None:
        main_arr = _map_reverse_batched(mapper, sub_arr, batch_size=map_batch)
    else:
        main_arr = np.asarray(main_latent, dtype=np.float32)
    X = _build_classifier_space_latent(
        train_space=str(train_space),
        main_latent=main_arr,
        sub_latent=sub_arr,
        mapper=mapper,
        map_batch=int(map_batch),
    )
    return np.asarray(classifier.predict(X)).astype(str)


def _labels_to_colors(labels: Sequence[str]) -> np.ndarray:
    return np.asarray([CELL_TYPE_COLORS.get(str(x), OTHER_COLOR) for x in labels], dtype=object)


def _set_panel_frame(ax: plt.Axes, *, is_real: bool) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.6)
        spine.set_color("black")
        spine.set_linestyle("-" if is_real else (0, (3, 2)))


def _axis_limits(
    slices: Iterable[np.ndarray],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    all_pts: List[np.ndarray] = []
    for arr in slices:
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.size > 0:
            all_pts.append(arr_np)
    if not all_pts:
        raise ValueError("Cannot determine axis limits from empty slices.")
    stacked = np.vstack(all_pts)
    x_min, y_min = np.min(stacked, axis=0)
    x_max, y_max = np.max(stacked, axis=0)
    dx = max(float(x_max - x_min), 1e-6)
    dy = max(float(y_max - y_min), 1e-6)
    pad_x = 0.05 * dx
    pad_y = 0.05 * dy
    return (float(x_min - pad_x), float(x_max + pad_x)), (float(y_min - pad_y), float(y_max + pad_y))


def _plot_slice_grid(
    *,
    output_dir: Path,
    display_times: Sequence[float],
    main_panel_coords: Dict[float, np.ndarray],
    sub_panel_coords: Dict[float, np.ndarray],
    main_panel_labels: Dict[float, np.ndarray],
    sub_panel_labels: Dict[float, np.ndarray],
) -> Dict[str, str]:
    display = [float(x) for x in display_times]
    main_xlim, main_ylim = _axis_limits([main_panel_coords[t] for t in display])
    sub_xlim, sub_ylim = _axis_limits([sub_panel_coords[t] for t in display])

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "axes.grid": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        len(display),
        figsize=(2.45 * len(display), 5.8),
        dpi=900,
        gridspec_kw={"hspace": 0.28, "wspace": 0.10},
    )

    for col, t in enumerate(display):
        t_val = float(t)
        is_real = _is_real_time(t_val)

        ax_main = axes[0, col]
        ax_main.scatter(
            main_panel_coords[t_val][:, 0],
            main_panel_coords[t_val][:, 1],
            s=7.5,
            c=_labels_to_colors(main_panel_labels[t_val]),
            alpha=0.78,
            linewidths=0.0,
            rasterized=True,
            zorder=2,
        )
        ax_main.set_facecolor(RNA_BG_COLOR)
        ax_main.set_title(f"t={t_val:.1f}", fontsize=10.5, pad=8.0)
        ax_main.set_xlim(main_xlim)
        ax_main.set_ylim(main_ylim)
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        if col == 0:
            ax_main.set_ylabel("RNA", fontsize=11.0)
        _set_panel_frame(ax_main, is_real=is_real)

        ax_sub = axes[1, col]
        ax_sub.scatter(
            sub_panel_coords[t_val][:, 0],
            sub_panel_coords[t_val][:, 1],
            s=7.5,
            c=_labels_to_colors(sub_panel_labels[t_val]),
            alpha=0.78,
            linewidths=0.0,
            rasterized=True,
            zorder=2,
        )
        ax_sub.set_facecolor(SUB_BG_COLOR)
        ax_sub.set_xlim(sub_xlim)
        ax_sub.set_ylim(sub_ylim)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])
        if col == 0:
            ax_sub.set_ylabel("Mapped Sub-space", fontsize=11.0)
        _set_panel_frame(ax_sub, is_real=is_real)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=CELL_TYPE_COLORS[label],
            markeredgecolor=CELL_TYPE_COLORS[label],
            markersize=6.0,
            label=label,
        )
        for label in CELL_TYPE_ORDER
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.6,
            linestyle="-",
            label="real",
        )
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.6,
            linestyle=(0, (3, 2)),
            label="interpolated",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.992, 0.50),
        frameon=False,
        fontsize=9.2,
    )
    fig.subplots_adjust(right=0.93)

    png_path = output_dir / "time_slice_panels.png"
    pdf_path = output_dir / "time_slice_panels.pdf"
    svg_path = output_dir / "time_slice_panels.svg"
    fig.savefig(png_path, bbox_inches="tight", dpi=900)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _build_coordinate_table(
    *,
    display_times: Sequence[float],
    main_panel_coords: Dict[float, np.ndarray],
    sub_panel_coords: Dict[float, np.ndarray],
    main_panel_labels: Dict[float, np.ndarray],
    sub_panel_labels: Dict[float, np.ndarray],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in display_times:
        t_val = float(t)
        for modality, coords_map, label_map in [
            ("rna", main_panel_coords, main_panel_labels),
            ("sub", sub_panel_coords, sub_panel_labels),
        ]:
            coords = np.asarray(coords_map[t_val], dtype=float)
            labels = np.asarray(label_map[t_val]).astype(str)
            for idx in range(coords.shape[0]):
                rows.append(
                    {
                        "time": float(t_val),
                        "modality": modality,
                        "is_real": bool(_is_real_time(t_val)),
                        "umap1": float(coords[idx, 0]),
                        "umap2": float(coords[idx, 1]),
                        "pred_cell_type": str(labels[idx]),
                    }
                )
    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 31800 RNA/sub-space temporal slice panels with real+interpolated time points."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "analysis" / "hspc_31800.yaml"),
        help="Path to downstream 31800 YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory. Defaults to <output.root_dir>/time_slice_panels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Cell batch size for chunked ODE inference and mapper calls.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=40,
        help="UMAP n_neighbors when refitting reducers.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.25,
        help="UMAP min_dist when refitting reducers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP and plotting subsampling.",
    )
    parser.add_argument(
        "--background-max-points",
        type=int,
        default=40000,
        help="Maximum background points plotted per modality.",
    )
    parser.add_argument(
        "--force-refit-umap",
        action="store_true",
        help="Ignore reusable pickled UMAP reducers and refit from real data.",
    )
    parser.add_argument(
        "--rna-umap-model", type=Path, default=None,
        help="Paper-selected fitted RNA UMAP reducer.",
    )
    parser.add_argument(
        "--sub-umap-model", type=Path, default=None,
        help="Paper-selected fitted protein/ATAC UMAP reducer.",
    )
    parser.add_argument(
        "--force-rerun-sim",
        action="store_true",
        help="Ignore cached interpolated slices and rerun ODE simulation.",
    )
    return parser.parse_args(argv)


def render(**overrides: Any) -> Path:
    """Render HSPC time slices from Python without a subprocess or CLI."""

    args = parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"Unknown HSPC time-slice option: {key}")
        setattr(args, key, value)
    main(args)
    return Path(args.output_dir).resolve()


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = parse_args() if args is None else args
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)
    paths = _extract_paths(cfg)
    output_root = _extract_output_root(cfg)
    output_dir = _resolve_output_dir(cfg, args.output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    analysis_cfg = cfg.get("analysis", {})
    if not isinstance(analysis_cfg, dict):
        raise ValueError("config.analysis must be a mapping")
    inference_cfg = cfg.get("inference", {})
    if not isinstance(inference_cfg, dict):
        raise ValueError("config.inference must be a mapping")
    classifier_cfg = analysis_cfg.get("classifier", {})
    if not isinstance(classifier_cfg, dict):
        raise ValueError("config.analysis.classifier must be a mapping")

    time_key = str(analysis_cfg.get("time_key", "time_point_processed"))
    cell_type_key = str(analysis_cfg.get("cell_type_key", "cell_type"))
    dt = float(inference_cfg.get("dt", 0.05))
    requested_device = str(inference_cfg.get("device", "cpu"))
    device = _resolve_runtime_device(requested_device)
    if requested_device.lower().startswith("cuda") and device != requested_device and device == "cpu":
        print(f"[slice-panels] requested device={requested_device} but CUDA is unavailable; fallback to cpu")

    main_proc = ad.read_h5ad(str(paths["main_processed"]))
    sub_proc = ad.read_h5ad(str(paths["sub_processed"]))
    dynamic_adata = ad.read_h5ad(str(paths["dynamic_adata"]))

    _validate_time_contract(main_proc, time_key)
    _validate_time_contract(sub_proc, time_key)
    main_latent_all = np.asarray(main_proc.obsm["X_latent"], dtype=np.float32)
    sub_latent_all = np.asarray(sub_proc.obsm["X_latent"], dtype=np.float32)
    main_times = np.asarray(main_proc.obs[time_key], dtype=float)
    sub_times = np.asarray(sub_proc.obs[time_key], dtype=float)

    model = load_model_from_adata(dynamic_adata)
    mapper = _build_mapper(
        dynamic_adata=dynamic_adata,
        map_model_path=paths["map_model_path"],
        main_dim=int(main_latent_all.shape[1]),
        sub_dim=int(sub_latent_all.shape[1]),
        device=device,
    )

    classifier, classifier_meta = _train_or_load_classifier(
        main_proc=main_proc,
        sub_proc=sub_proc,
        mapper=mapper,
        cell_type_key=cell_type_key,
        classifier_cfg=classifier_cfg,
        output_root=output_root,
        reports_dir=reports_dir,
    )
    train_space = str(classifier_meta.get("train_space", classifier_cfg.get("train_space", "joint"))).strip().lower()
    map_batch = int(classifier_meta.get("joint_map_batch", classifier_cfg.get("joint_map_batch", 2048)))

    search_roots = [output_dir, output_root]
    main_reducer, main_bg_embed, main_umap_meta = _resolve_umap(
        label="rna",
        latent_real=main_latent_all,
        output_dir=output_dir,
        search_roots=search_roots,
        tokens=["rna", "main", "primary"],
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
        force_refit=args.force_refit_umap,
        explicit_model=args.rna_umap_model,
    )
    sub_reducer, sub_bg_embed, sub_umap_meta = _resolve_umap(
        label="sub",
        latent_real=sub_latent_all,
        output_dir=output_dir,
        search_roots=search_roots,
        tokens=["sub", "secondary", "protein", "prot", "adt", "atac"],
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
        force_refit=args.force_refit_umap,
        explicit_model=args.sub_umap_model,
    )

    cache_signature = _build_cache_signature(
        dynamic_adata_path=paths["dynamic_adata"],
        main_processed_path=paths["main_processed"],
        sub_processed_path=paths["sub_processed"],
        map_model_path=paths["map_model_path"],
        dt=dt,
        batch_size=args.batch_size,
        display_times=DISPLAY_TIMES,
        device=device,
    )

    main_slices: Dict[float, np.ndarray] = {}
    sub_slices: Dict[float, np.ndarray] = {}
    cache_manifests: List[Dict[str, Any]] = []

    for real_t in EXPECTED_REAL_TIMES.tolist():
        main_slices[float(real_t)] = np.asarray(main_latent_all[_mask_for_time(main_times, real_t)], dtype=np.float32)
        sub_slices[float(real_t)] = np.asarray(sub_latent_all[_mask_for_time(sub_times, real_t)], dtype=np.float32)

    for anchor_time, query_times in INTERP_BY_ANCHOR.items():
        anchor_main, anchor_sub, cache_meta = _simulate_anchor_slices(
            main_proc=main_proc,
            time_key=time_key,
            anchor_time=float(anchor_time),
            query_times=query_times,
            model=model,
            dynamic_adata=dynamic_adata,
            mapper=mapper,
            dt=dt,
            device=device,
            batch_size=args.batch_size,
            cache_dir=cache_dir,
            cache_signature=cache_signature,
            force_rerun=args.force_rerun_sim,
        )
        cache_manifests.append(cache_meta)
        for query_t in query_times:
            main_slices[float(query_t)] = anchor_main[float(query_t)]
            sub_slices[float(query_t)] = anchor_sub[float(query_t)]

    main_panel_labels: Dict[float, np.ndarray] = {}
    sub_panel_labels: Dict[float, np.ndarray] = {}
    main_panel_coords: Dict[float, np.ndarray] = {}
    sub_panel_coords: Dict[float, np.ndarray] = {}
    cell_counts: Dict[str, Dict[str, int]] = {"rna": {}, "sub": {}}

    for t in DISPLAY_TIMES.tolist():
        t_val = float(t)
        if t_val not in main_slices or t_val not in sub_slices:
            raise ValueError(f"Missing slice at time {t_val:g}")

        main_slice = np.asarray(main_slices[t_val], dtype=np.float32)
        sub_slice = np.asarray(sub_slices[t_val], dtype=np.float32)
        if main_slice.shape[0] == 0 or sub_slice.shape[0] == 0:
            raise ValueError(f"Slice at time {t_val:g} is empty")

        main_pred, mapped_sub = _classify_main_slice(
            classifier=classifier,
            train_space=train_space,
            map_batch=map_batch,
            mapper=mapper,
            main_latent=main_slice,
        )
        if _is_real_time(t_val):
            sub_pred = _classify_sub_slice(
                classifier=classifier,
                train_space=train_space,
                map_batch=map_batch,
                mapper=mapper,
                sub_latent=sub_slice,
                main_latent=None,
            )
        else:
            sub_pred = _classify_sub_slice(
                classifier=classifier,
                train_space=train_space,
                map_batch=map_batch,
                mapper=mapper,
                sub_latent=sub_slice,
                main_latent=main_slice,
            )

        main_panel_labels[t_val] = np.asarray(main_pred).astype(str)
        sub_panel_labels[t_val] = np.asarray(sub_pred).astype(str)
        main_panel_coords[t_val] = _transform_2d(main_reducer, main_slice)
        sub_panel_coords[t_val] = _transform_2d(sub_reducer, sub_slice)
        cell_counts["rna"][f"{t_val:g}"] = int(main_slice.shape[0])
        cell_counts["sub"][f"{t_val:g}"] = int(sub_slice.shape[0])

        if not _is_real_time(t_val):
            # Keep the mapped counterpart alive for classifier consistency checks.
            _ = mapped_sub

    figure_paths = _plot_slice_grid(
        output_dir=output_dir,
        display_times=DISPLAY_TIMES,
        main_panel_coords=main_panel_coords,
        sub_panel_coords=sub_panel_coords,
        main_panel_labels=main_panel_labels,
        sub_panel_labels=sub_panel_labels,
    )

    coords_df = _build_coordinate_table(
        display_times=DISPLAY_TIMES,
        main_panel_coords=main_panel_coords,
        sub_panel_coords=sub_panel_coords,
        main_panel_labels=main_panel_labels,
        sub_panel_labels=sub_panel_labels,
    )
    coords_csv = output_dir / "time_slice_coordinates.csv"
    coords_df.to_csv(coords_csv, index=False)

    manifest = {
        "config_path": str(cfg_path),
        "output_dir": str(output_dir),
        "output_root": str(output_root),
        "device_requested": requested_device,
        "device_used": device,
        "dt": float(dt),
        "batch_size": int(args.batch_size),
        "time_key": str(time_key),
        "cell_type_key": str(cell_type_key),
        "display_times": [float(x) for x in DISPLAY_TIMES.tolist()],
        "real_times": [float(x) for x in EXPECTED_REAL_TIMES.tolist()],
        "cache_signature": cache_signature,
        "cache_manifests": cache_manifests,
        "classifier": classifier_meta,
        "umap": {
            "rna": main_umap_meta,
            "sub": sub_umap_meta,
        },
        "cell_counts": cell_counts,
        "outputs": {
            "png": figure_paths["png"],
            "pdf": figure_paths["pdf"],
            "coordinates_csv": str(coords_csv),
        },
    }
    manifest_path = output_dir / "time_slice_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"[slice-panels] wrote figure: {figure_paths['png']}")
    print(f"[slice-panels] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
