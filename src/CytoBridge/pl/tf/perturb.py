from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from CytoBridge.pl.plot_dense_time import (
    DEFAULT_FIGURE_CFG,
    plot_perturbation_sweep_heatmap,
    plot_perturbation_sweep_heatmap_prob,
    plot_perturbation_timecurve_focus,
    plot_perturbation_zscore_sweep,
)
from CytoBridge.pl.plot_joint import apply_umap_transform, get_or_train_umap
from CytoBridge.pl.plot_ode_v3 import plot_ode_v3 as plot_ode_v3_core
from CytoBridge.tl.analysis_dense_time import (
    _build_classifier_space_latent,
    _build_ode_trajectory,
    _map_main_to_sub_like,
    _map_to_secondary,
    _predict_proba_for_classes,
    _project_feature_to_latent,
    _reconstruct_latent_to_feature,
    _resolve_target_cell_type,
)

from .context import TFRunContext


def _slugify(text: Any) -> str:
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
    return _slugify(txt)


def _load_sweep_df(ctx: TFRunContext) -> pd.DataFrame:
    path = ctx.latest_perturbation_sweep_csv()
    return pd.read_csv(path)


def _auto_pick_group_z_cell(df: pd.DataFrame) -> Tuple[str, float, str]:
    work = df.copy()
    if "is_selected_cell_type" in work.columns and bool(np.any(work["is_selected_cell_type"])):
        work = work[work["is_selected_cell_type"] == True].copy()  # noqa: E712
        if work.empty:
            work = df.copy()

    by_group = (
        work.groupby("target_group", as_index=False)["delta_rate"]
        .apply(lambda x: float(np.mean(np.abs(np.asarray(x, dtype=float)))))
        .rename(columns={"delta_rate": "score"})
        .sort_values("score", ascending=False)
    )
    if by_group.empty:
        raise ValueError("cannot auto-pick target_group from sweep")
    target_group = str(by_group.iloc[0]["target_group"])

    z_vals = sorted({float(x) for x in work["z_score"].astype(float).tolist()})
    z_score = sorted(z_vals, key=lambda x: (abs(float(x)), float(x)), reverse=True)[0]

    sub = work[(work["target_group"].astype(str) == target_group) & (np.isclose(work["z_score"].astype(float), float(z_score)))].copy()
    if sub.empty:
        sub = work[work["target_group"].astype(str) == target_group].copy()
    if sub.empty:
        raise ValueError("cannot auto-pick cell_type from sweep")

    cell_rank = (
        sub.groupby("cell_type", as_index=False)["delta_rate"]
        .apply(lambda x: float(np.mean(np.abs(np.asarray(x, dtype=float)))))
        .rename(columns={"delta_rate": "score"})
        .sort_values("score", ascending=False)
    )
    cell_type = str(cell_rank.iloc[0]["cell_type"])
    return target_group, float(z_score), cell_type


def _resolve_target_genes_from_group(df: pd.DataFrame, target_group: str) -> List[str]:
    sub = df[df["target_group"].astype(str) == str(target_group)].copy()
    if sub.empty:
        raise ValueError(f"target_group not found in sweep: {target_group}")
    genes_raw = str(sub["target_genes"].iloc[0]).strip()
    genes = [x.strip() for x in genes_raw.split("|") if x.strip()]
    if not genes:
        raise ValueError(f"empty target_genes for group={target_group}")
    return genes


def _normalize_index_label_rule(raw: Any) -> str:
    token = str(raw).strip().lower()
    if token in {"", "random", "rand"}:
        return "random"
    if token in {"argmax", "endpoint", "endpoint_argmax", "simulate_endpoint"}:
        return "endpoint"
    if token in {"time0", "time0_cell_type", "source_time0", "source"}:
        return "time0"
    if token in {"auto", "fallback", "hybrid"}:
        return "auto"
    return "random"


def _normalize_merge_cell_types_config(raw: Any) -> Dict[str, Any]:
    cfg = raw if isinstance(raw, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    override_key = str(cfg.get("override_key", "neuron_type")).strip() or "neuron_type"
    na_tokens_raw = cfg.get("na_tokens", [])
    if isinstance(na_tokens_raw, str):
        na_tokens = [x.strip().lower() for x in na_tokens_raw.split(",") if x.strip()]
    elif isinstance(na_tokens_raw, (list, tuple, np.ndarray)):
        na_tokens = [str(x).strip().lower() for x in na_tokens_raw if str(x).strip()]
    else:
        na_tokens = []
    na_tokens = sorted(set(na_tokens))
    return {
        "enabled": True,
        "override_key": str(override_key),
        "na_tokens": [str(x) for x in na_tokens],
    }


def _build_indices_cache_settings(
    *,
    default_cell_type_key: str,
    source_cell_type_key: str = "",
    merge_cell_types: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cell_key = str(source_cell_type_key).strip() or str(default_cell_type_key).strip() or "cell_type"
    return {
        "source_cell_type_key": str(cell_key),
        "cell_type_merge": _normalize_merge_cell_types_config(merge_cell_types),
    }


def _exact_cell_type_mask(values: np.ndarray, requested: str) -> tuple[str, np.ndarray]:
    vals = np.asarray(values, dtype=object).astype(str)
    req = str(requested).strip()
    if vals.size == 0 or (not req):
        return "", np.zeros(int(vals.size), dtype=bool)
    mask_exact = vals == req
    if bool(np.any(mask_exact)):
        return str(req), np.asarray(mask_exact, dtype=bool)
    req_l = req.lower()
    mask_ci = np.asarray([str(x).lower() == req_l for x in vals], dtype=bool)
    if bool(np.any(mask_ci)):
        matched = str(vals[np.where(mask_ci)[0][0]])
        return matched, mask_ci
    return "", np.zeros(int(vals.size), dtype=bool)


def _normalized_na_tokens(raw: Any) -> set[str]:
    tokens = {"", "na", "nan", "none", "null"}
    if raw is None:
        return tokens
    if isinstance(raw, str):
        items = [x.strip().lower() for x in raw.split(",")]
    elif isinstance(raw, (list, tuple, np.ndarray)):
        items = [str(x).strip().lower() for x in raw]
    else:
        items = [str(raw).strip().lower()]
    for item in items:
        if item:
            tokens.add(item)
    return tokens


def _merge_cell_labels(
    base_values: np.ndarray,
    override_values: np.ndarray,
    *,
    na_tokens: Any = None,
) -> np.ndarray:
    base = np.asarray(base_values, dtype=object).astype(str)
    over = np.asarray(override_values, dtype=object).astype(str)
    if base.shape[0] != over.shape[0]:
        raise ValueError(f"base/override length mismatch: {base.shape[0]} vs {over.shape[0]}")
    na_set = _normalized_na_tokens(na_tokens)
    out = base.copy()
    mask = np.asarray([str(x).strip().lower() not in na_set for x in over], dtype=bool)
    out[mask] = over[mask]
    return out


def _normalize_label_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [x.strip() for x in raw.split(",")]
    elif isinstance(raw, (list, tuple, np.ndarray, pd.Series)):
        items = [str(x).strip() for x in raw]
    else:
        items = [str(raw).strip()]
    out: List[str] = []
    seen = set()
    for item in items:
        if (not item) or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _match_requested_labels(requested: Sequence[str], available: Sequence[str]) -> Tuple[List[str], List[str]]:
    avail = [str(x) for x in available]
    used: List[str] = []
    missing: List[str] = []
    seen = set()
    for req in _normalize_label_list(requested):
        hit = ""
        for item in avail:
            if str(item).lower() == str(req).lower():
                hit = str(item)
                break
        if not hit:
            missing.append(str(req))
            continue
        if hit in seen:
            continue
        seen.add(hit)
        used.append(hit)
    return used, missing


def _weighted_prob_mean(proba: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"probability matrix must be 2D, got shape={arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros((arr.shape[1],), dtype=float)
    if weights is None:
        return np.nanmean(arr, axis=0)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != arr.shape[0]:
        return np.nanmean(arr, axis=0)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    denom = float(np.sum(w))
    if denom <= 1e-12:
        return np.nanmean(arr, axis=0)
    return np.sum(arr * w[:, None], axis=0) / denom


def _predict_distribution_by_time(
    *,
    classifier: Any,
    classes: np.ndarray,
    train_space: str,
    map_batch: int,
    mapper: Any,
    main_latent_series: np.ndarray,
    sub_latent_series: np.ndarray,
    weights_series: Optional[np.ndarray],
) -> np.ndarray:
    latent_series = _build_classifier_space_latent(
        train_space=str(train_space),
        main_latent=np.asarray(main_latent_series, dtype=np.float32),
        sub_latent=np.asarray(sub_latent_series, dtype=np.float32),
        mapper=mapper,
        map_batch=int(map_batch),
    )
    if latent_series.ndim != 3:
        raise ValueError(f"classifier latent series must be 3D, got shape={latent_series.shape}")

    weights_use: Optional[np.ndarray] = None
    if weights_series is not None:
        arr = np.asarray(weights_series, dtype=float)
        if arr.shape[:2] == latent_series.shape[:2]:
            weights_use = arr

    out: List[np.ndarray] = []
    for t_idx in range(latent_series.shape[0]):
        proba_t, _ = _predict_proba_for_classes(
            classifier=classifier,
            X=np.asarray(latent_series[t_idx], dtype=np.float32),
            target_classes=np.asarray(classes, dtype=object),
            renormalize_subset=False,
        )
        w_t = None if weights_use is None else np.asarray(weights_use[t_idx], dtype=float)
        out.append(_weighted_prob_mean(proba_t, w_t))
    return np.asarray(out, dtype=float)


def build_perturbation_fate_sweep(
    ctx: TFRunContext,
    *,
    indices_csv: str = "",
    target_group: str = "",
    target_genes: Optional[Sequence[str]] = None,
    scores: Optional[Sequence[float]] = None,
    sample_n: Any = "auto",
    seed: int = 42,
    device: str = "",
    output_table_dir: str = "",
    output_figure_dir: str = "",
    sweep_cell_types: Optional[Sequence[str]] = None,
    focus_curve_cell_types: Optional[Sequence[str]] = None,
    focus_curve_scores: Optional[Sequence[float]] = None,
    heatmap_z_order: Optional[Sequence[float]] = None,
    heatmap_orientation: str = "cell_type_by_z",
    heatmap_cell_type_order: Optional[Sequence[str]] = None,
    heatmap_vmax_percent: Optional[float] = None,
    prob_heatmap_value_col: str = "perturbed_prob",
    dpi: int = 260,
) -> Dict[str, Any]:
    idx_csv_raw = str(indices_csv).strip()
    idx_csv = Path(idx_csv_raw).resolve() if idx_csv_raw else None

    out_table_dir = (
        Path(output_table_dir).resolve()
        if str(output_table_dir).strip()
        else (ctx.output_table_dir / "perturb_fate_sweep")
    )
    out_figure_dir = (
        Path(output_figure_dir).resolve()
        if str(output_figure_dir).strip()
        else (ctx.output_figure_dir / "perturb_fate_sweep")
    )
    out_table_dir.mkdir(parents=True, exist_ok=True)
    out_figure_dir.mkdir(parents=True, exist_ok=True)

    indices_df = pd.DataFrame()
    if idx_csv is not None:
        if not idx_csv.exists():
            raise FileNotFoundError(f"indices csv not found: {idx_csv}")
        indices_df = pd.read_csv(idx_csv)
        if indices_df.empty:
            raise ValueError(f"indices csv is empty: {idx_csv}")

    target_group_used = str(target_group).strip()
    target_genes_used = _normalize_label_list(target_genes)
    if (not target_group_used) and (not indices_df.empty):
        target_group_used = str(indices_df["target_group"].iloc[0]).strip()
    if not target_genes_used and (not indices_df.empty):
        target_genes_used = [x.strip() for x in str(indices_df["target_genes"].iloc[0]).split("|") if x.strip()]
    if not target_group_used:
        target_group_used = "custom_group"
    if not target_genes_used:
        raise ValueError("target_genes must be provided when indices_csv is not set")

    primary_proc, _ = ctx.load_primary_secondary_processed(backed=False)
    if not indices_df.empty:
        source_idx_all = np.asarray(indices_df["source_cell_index"], dtype=int)
        source_idx_all = np.unique(source_idx_all)
        if source_idx_all.size == 0:
            raise ValueError("No source_cell_index values found in indices csv")
    else:
        obs_time = np.asarray(primary_proc.obs[ctx.time_key], dtype=float)
        t0 = float(np.min(obs_time))
        source_idx_all = np.where(np.isclose(obs_time, t0))[0].astype(int)
        if source_idx_all.size == 0:
            raise ValueError("No time0 cells found for implicit source pool")

    sample_n_token = str(sample_n).strip().lower()
    if sample_n_token in {"", "auto", "all"}:
        sample_idx = np.asarray(source_idx_all, dtype=int)
        sample_policy = "all_source_pool"
    else:
        rng = np.random.default_rng(int(seed))
        n_take = int(min(max(int(sample_n), 1), source_idx_all.size))
        sample_idx = np.asarray(rng.choice(source_idx_all, size=n_take, replace=False), dtype=int)
        sample_policy = "random_without_replacement"
    sample_idx = np.asarray(np.unique(sample_idx), dtype=int)

    scores_use = sorted({float(x) for x in (scores if scores is not None else [0.0])})
    if not any(np.isclose(float(x), 0.0) for x in scores_use):
        scores_use = sorted(scores_use + [0.0])
    primary_bundle, _ = ctx.load_primary_secondary_bundles()
    cls_bundle = ctx.load_classifier_bundle()
    classifier = cls_bundle["classifier"]
    classifier_classes = np.asarray(cls_bundle["classes"], dtype=object)
    classifier_space = str(cls_bundle["train_space"]).strip().lower()
    map_batch = int(cls_bundle["map_batch"])

    run_device = str(device).strip() or "cpu"
    mapper = ctx.build_mapper(device=run_device)
    model = ctx.load_model()
    dyn_adata = ctx.load_dynamic_adata()

    x0_latent = np.asarray(primary_proc.obsm["X_latent"][sample_idx], dtype=np.float32)
    x0_feat_base = _reconstruct_latent_to_feature(x0_latent, primary_bundle).astype(np.float32, copy=True)
    feat_lut = {str(g): i for i, g in enumerate(primary_bundle.var_names)}
    feat_std = np.asarray(np.std(x0_feat_base, axis=0), dtype=np.float32)

    valid_gene_items: List[Tuple[str, int, float]] = []
    gene_stats: List[Dict[str, Any]] = []
    for gene in target_genes_used:
        idx = feat_lut.get(str(gene), None)
        if idx is None:
            gene_stats.append({"gene": str(gene), "status": "missing_in_var_names", "gene_index": -1, "feature_std": 0.0})
            continue
        sd = float(feat_std[int(idx)])
        if sd < 1e-8:
            gene_stats.append({"gene": str(gene), "status": "near_zero_std", "gene_index": int(idx), "feature_std": sd})
            continue
        gene_stats.append({"gene": str(gene), "status": "ok", "gene_index": int(idx), "feature_std": sd})
        valid_gene_items.append((str(gene), int(idx), sd))
    if not valid_gene_items:
        raise ValueError(f"No valid target genes in primary space. requested={target_genes_used}")

    classes = [str(x) for x in classifier_classes.tolist()]
    sweep_types_req = _normalize_label_list(sweep_cell_types)
    sweep_types_used, sweep_types_missing = _match_requested_labels(sweep_types_req, classes)
    if not sweep_types_used:
        sweep_types_used = list(classes)
    focus_types_req = _normalize_label_list(focus_curve_cell_types)
    focus_types_used, focus_types_missing = _match_requested_labels(focus_types_req, classes)
    if not focus_types_used:
        focus_types_used = [x for x in sweep_types_used] if sweep_types_used else list(classes)
    focus_scores_req = sorted({float(x) for x in (focus_curve_scores if focus_curve_scores is not None else [0.0])})
    focus_scores_used = [float(z) for z in scores_use if any(np.isclose(float(z), float(req)) for req in focus_scores_req)]
    if not focus_scores_used:
        focus_scores_used = [0.0] if any(np.isclose(float(z), 0.0) for z in scores_use) else [float(scores_use[0])]

    time_grid = np.asarray(ctx.time_grid, dtype=float)
    dt = float(ctx.manifest.get("dt", 0.05))

    dist_by_time: Dict[float, np.ndarray] = {}
    for z in scores_use:
        x0_feat = x0_feat_base.copy()
        for _, idx, sd in valid_gene_items:
            x0_feat[:, int(idx)] = x0_feat[:, int(idx)] + float(z) * float(sd)
        z0 = _project_feature_to_latent(x0_feat, primary_bundle)
        traj_primary = _build_ode_trajectory(
            model=model,
            x0=z0,
            times=time_grid,
            adata=dyn_adata,
            dt=dt,
            device=run_device,
        )
        traj_secondary = _map_to_secondary(mapper, traj_primary["main_latent"])
        weights_series = traj_primary.get("weights_abs", None)
        dist_by_time[float(z)] = _predict_distribution_by_time(
            classifier=classifier,
            classes=np.asarray(classifier_classes, dtype=object),
            train_space=classifier_space,
            map_batch=map_batch,
            mapper=mapper,
            main_latent_series=np.asarray(traj_primary["main_latent"], dtype=np.float32),
            sub_latent_series=np.asarray(traj_secondary["sub_latent"], dtype=np.float32),
            weights_series=None if weights_series is None else np.asarray(weights_series, dtype=float),
        )

    baseline_key = 0.0
    if baseline_key not in dist_by_time:
        raise ValueError("baseline z=0 distribution missing")
    baseline_time = np.asarray(dist_by_time[baseline_key], dtype=float)
    baseline_final = np.asarray(baseline_time[-1], dtype=float)

    class_index = {str(name): i for i, name in enumerate(classes)}
    sweep_active_set = {str(x).lower() for x in sweep_types_used}
    focus_active_set = {str(x).lower() for x in focus_types_used}
    mean_feature_std = float(np.mean([float(x[2]) for x in valid_gene_items])) if valid_gene_items else 0.0

    fate_rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []
    focus_curve_rows: List[Dict[str, Any]] = []
    for z in scores_use:
        pert_time = np.asarray(dist_by_time[float(z)], dtype=float)
        pert_final = np.asarray(pert_time[-1], dtype=float)
        tag = f"{target_group_used}@z{float(z):g}"
        for cell_type in classes:
            idx = int(class_index[cell_type])
            base_prob = float(baseline_final[idx])
            pert_prob = float(pert_final[idx])
            delta_prob = float(pert_prob - base_prob)
            delta_rate = float(delta_prob / max(abs(base_prob), 1e-8))
            fate_rows.append(
                {
                    "target": str(target_group_used),
                    "target_group": str(target_group_used),
                    "target_genes": "|".join(target_genes_used),
                    "n_genes": int(len(target_genes_used)),
                    "gene_index": -1,
                    "feature_std": float(mean_feature_std),
                    "delta_feature": float(z) * float(mean_feature_std),
                    "z_score": float(z),
                    "tag": str(tag),
                    "cell_type": str(cell_type),
                    "baseline_prob": base_prob,
                    "perturbed_prob": pert_prob,
                    "delta_prob": delta_prob,
                    "space": classifier_space,
                }
            )
            sweep_rows.append(
                {
                    "target_group": str(target_group_used),
                    "target_genes": "|".join(target_genes_used),
                    "n_genes": int(len(target_genes_used)),
                    "z_score": float(z),
                    "cell_type": str(cell_type),
                    "baseline_prob": base_prob,
                    "perturbed_prob": pert_prob,
                    "delta_prob": delta_prob,
                    "delta_rate": delta_rate,
                    "space": classifier_space,
                    "sweep_space": classifier_space,
                    "is_selected_cell_type": bool(str(cell_type).lower() in sweep_active_set),
                }
            )
        if any(np.isclose(float(z), float(req)) for req in focus_scores_used):
            for t_idx, t in enumerate(time_grid):
                for cell_type in classes:
                    if str(cell_type).lower() not in focus_active_set:
                        continue
                    idx = int(class_index[cell_type])
                    base_prob_t = float(baseline_time[t_idx, idx])
                    pert_prob_t = float(baseline_time[t_idx, idx] if t_idx == 0 else pert_time[t_idx, idx])
                    delta_prob_t = float(pert_prob_t - base_prob_t)
                    focus_curve_rows.append(
                        {
                            "target_group": str(target_group_used),
                            "target_genes": "|".join(target_genes_used),
                            "n_genes": int(len(target_genes_used)),
                            "z_score": float(z),
                            "time": float(t),
                            "cell_type": str(cell_type),
                            "baseline_prob": base_prob_t,
                            "perturbed_prob": pert_prob_t,
                            "delta_prob": delta_prob_t,
                            "delta_rate": float(delta_prob_t / max(abs(base_prob_t), 1e-8)),
                            "space": classifier_space,
                            "sweep_space": classifier_space,
                            "is_selected_cell_type": bool(str(cell_type).lower() in focus_active_set),
                            "status": "success",
                        }
                    )

    fate_df = pd.DataFrame(fate_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    focus_curve_df = pd.DataFrame(focus_curve_rows)

    fate_csv = out_table_dir / "perturbation_fate_shift_main.csv"
    sweep_csv = out_table_dir / "perturbation_zscore_sweep.csv"
    focus_curve_csv = out_table_dir / "perturbation_timecurve_focus.csv"
    fate_df.to_csv(fate_csv, index=False)
    sweep_df.to_csv(sweep_csv, index=False)
    focus_curve_df.to_csv(focus_curve_csv, index=False)

    fig_cfg = dict(DEFAULT_FIGURE_CFG)
    fig_cfg["dpi"] = int(dpi)
    fig_cfg["perturb_max_groups"] = 1
    fig_cfg["perturb_max_cell_types"] = max(len(classes), 1)
    fig_cfg["perturb_curve_cell_types"] = [str(x) for x in focus_types_used]
    fig_cfg["perturb_curve_scores"] = [float(x) for x in focus_scores_used]
    fig_cfg["perturb_heatmap_z_order"] = [float(x) for x in _normalize_label_list(heatmap_z_order or scores_use)] if heatmap_z_order else [float(x) for x in scores_use]
    fig_cfg["perturb_heatmap_orientation"] = str(heatmap_orientation)
    fig_cfg["perturb_heatmap_cell_type_order"] = [
        str(x) for x in (heatmap_cell_type_order or [])
    ]
    if heatmap_vmax_percent is not None:
        fig_cfg["perturb_heatmap_vmax_percent"] = float(heatmap_vmax_percent)

    sweep_png = out_figure_dir / "perturbation_zscore_sweep.pdf"
    heatmap_png = out_figure_dir / "perturbation_sweep_heatmap.pdf"
    prob_heatmap_png = out_figure_dir / "perturbation_sweep_heatmap_prob.pdf"
    plot_perturbation_zscore_sweep(str(sweep_csv), str(sweep_png), cfg=fig_cfg)
    plot_perturbation_sweep_heatmap(str(sweep_csv), str(heatmap_png), cfg=fig_cfg)
    if str(prob_heatmap_value_col) == "perturbed_prob":
        plot_perturbation_sweep_heatmap_prob(str(sweep_csv), str(prob_heatmap_png), cfg=fig_cfg)
    else:
        plot_perturbation_sweep_heatmap(
            str(sweep_csv),
            str(prob_heatmap_png),
            cfg=fig_cfg,
            value_col=str(prob_heatmap_value_col),
        )

    focus_curve_pngs: Dict[str, str] = {}
    for z in focus_scores_used:
        out_png = out_figure_dir / f"perturbation_timecurve_focus__z-{_signed_float_token(float(z))}.png"
        plot_perturbation_timecurve_focus(
            str(focus_curve_csv),
            str(out_png),
            cfg=fig_cfg,
            target_group=str(target_group_used),
            z_score=float(z),
        )
        focus_curve_pngs[f"{float(z):g}"] = str(out_png)

    summary = {
        "run_dir": str(ctx.run_dir),
        "indices_csv": (str(idx_csv) if idx_csv is not None else ""),
        "target_group": str(target_group_used),
        "target_genes": [str(x) for x in target_genes_used],
        "sample_n": int(sample_idx.size),
        "sample_policy": str(sample_policy),
        "sampled_source_indices": [int(x) for x in sample_idx.tolist()],
        "scores": [float(x) for x in scores_use],
        "classifier_space": str(classifier_space),
        "sweep_cell_types_requested": [str(x) for x in sweep_types_req],
        "sweep_cell_types_used": [str(x) for x in sweep_types_used],
        "sweep_cell_types_missing": [str(x) for x in sweep_types_missing],
        "focus_curve_cell_types_requested": [str(x) for x in focus_types_req],
        "focus_curve_cell_types_used": [str(x) for x in focus_types_used],
        "focus_curve_cell_types_missing": [str(x) for x in focus_types_missing],
        "focus_curve_scores_requested": [float(x) for x in focus_scores_req],
        "focus_curve_scores_used": [float(x) for x in focus_scores_used],
        "gene_stats": gene_stats,
        "outputs": {
            "fate_csv": str(fate_csv),
            "sweep_csv": str(sweep_csv),
            "focus_curve_csv": str(focus_curve_csv),
            "sweep_png": str(sweep_png),
            "heatmap_png": str(heatmap_png),
            "prob_heatmap_png": str(prob_heatmap_png),
            "focus_curve_pngs": focus_curve_pngs,
        },
    }
    summary_json = out_table_dir / "perturbation_fate_sweep_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_json"] = str(summary_json)
    return summary


def extract_perturbation_indices(
    ctx: TFRunContext,
    *,
    target_group: str = "auto",
    z_score: str = "auto",
    time_point: str = "auto",
    cell_type: str = "auto",
    label_rule: str = "random",
    device: str = "",
    output_csv: str = "",
    source_cell_type_key: str = "",
    merge_cell_types: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sweep_df = _load_sweep_df(ctx)

    if str(target_group).strip().lower() == "auto" or str(z_score).strip().lower() == "auto" or str(cell_type).strip().lower() == "auto":
        g_auto, z_auto, c_auto = _auto_pick_group_z_cell(sweep_df)
    else:
        g_auto, z_auto, c_auto = str(target_group), float(z_score), str(cell_type)

    group_used = str(g_auto if str(target_group).strip().lower() == "auto" else target_group)
    z_used = float(z_auto if str(z_score).strip().lower() == "auto" else float(z_score))
    cell_used_in = str(c_auto if str(cell_type).strip().lower() == "auto" else cell_type)

    if str(time_point).strip().lower() == "auto":
        t_req = float(max(ctx.time_grid))
    else:
        t_req = float(time_point)

    target_genes = _resolve_target_genes_from_group(sweep_df, group_used)

    selection_mode = _normalize_index_label_rule(label_rule)

    primary_proc, secondary_proc = ctx.load_primary_secondary_processed(backed=False)
    model = ctx.load_model()
    mapper = ctx.build_mapper(device=(str(device).strip() or "cpu"))
    cls_bundle = ctx.load_classifier_bundle()

    classifier = cls_bundle["classifier"]
    classifier_classes = np.asarray(cls_bundle["classes"], dtype=object)
    train_space = str(cls_bundle["train_space"]).strip().lower()
    map_batch = int(cls_bundle["map_batch"])

    init_indices = ctx.load_init_indices_from_pred_main()

    primary_bundle, _ = ctx.load_primary_secondary_bundles()
    x0_latent = np.asarray(primary_proc.obsm["X_latent"][init_indices], dtype=np.float32)
    x0_feat = _reconstruct_latent_to_feature(x0_latent, primary_bundle).astype(np.float32, copy=True)

    feat_lut = {str(g): i for i, g in enumerate(primary_bundle.var_names)}
    feat_std = np.asarray(np.std(x0_feat, axis=0), dtype=np.float32)

    valid_genes: List[str] = []
    x0_pert_feat = x0_feat.copy()
    for gene in target_genes:
        idx = feat_lut.get(str(gene), None)
        if idx is None:
            continue
        sd = float(feat_std[int(idx)])
        if sd < 1e-8:
            continue
        x0_pert_feat[:, int(idx)] = x0_pert_feat[:, int(idx)] + float(z_used) * sd
        valid_genes.append(str(gene))
    if not valid_genes:
        raise ValueError(
            f"No valid perturbation genes in primary space. Requested={target_genes}"
        )

    z0_pert = _project_feature_to_latent(x0_pert_feat, primary_bundle)
    tg = np.asarray(ctx.time_grid, dtype=float)
    t_idx = int(np.argmin(np.abs(tg - float(t_req))))
    t_used = float(tg[t_idx])

    need_endpoint_sim = selection_mode in {"endpoint", "auto"}
    pred_labels = np.asarray([""] * int(len(init_indices)), dtype=object)
    target_prob = np.asarray(np.full(int(len(init_indices)), np.nan), dtype=float)
    max_prob = np.asarray(np.full(int(len(init_indices)), np.nan), dtype=float)
    classes = np.asarray([], dtype=object)
    cell_type_used = str(cell_used_in)
    endpoint_keep_mask: Optional[np.ndarray] = None

    if need_endpoint_sim:
        traj_primary = _build_ode_trajectory(
            model=model,
            x0=z0_pert,
            times=tg,
            adata=ctx.load_dynamic_adata(),
            dt=float(ctx.manifest.get("dt", 0.05)),
            device=(str(device).strip() or "cpu"),
        )
        traj_secondary = _map_to_secondary(mapper, traj_primary["main_latent"])

        classifier_latent = _build_classifier_space_latent(
            train_space=train_space,
            main_latent=np.asarray(traj_primary["main_latent"], dtype=np.float32),
            sub_latent=np.asarray(traj_secondary["sub_latent"], dtype=np.float32),
            mapper=mapper,
            map_batch=map_batch,
        )
        proba_t, classes = _predict_proba_for_classes(
            classifier=classifier,
            X=np.asarray(classifier_latent[t_idx], dtype=np.float32),
            target_classes=classifier_classes,
            renormalize_subset=False,
        )

        pred_idx = np.argmax(proba_t, axis=1)
        pred_labels = classes[pred_idx].astype(str)
        cell_type_used = _resolve_target_cell_type(classes.astype(str), cell_used_in)
        target_col_idx = int(np.where(classes.astype(str) == str(cell_type_used))[0][0])
        target_prob = np.asarray(proba_t[:, target_col_idx], dtype=float)
        max_prob = np.asarray(np.max(proba_t, axis=1), dtype=float)
        endpoint_keep_mask = pred_labels == str(cell_type_used)

    source_label_settings = _build_indices_cache_settings(
        default_cell_type_key=str(ctx.cell_type_key),
        source_cell_type_key=str(source_cell_type_key),
        merge_cell_types=merge_cell_types,
    )
    cell_key = str(source_label_settings["source_cell_type_key"])
    if cell_key not in primary_proc.obs.columns:
        raise KeyError(f"source_cell_type_key not found in primary obs: {cell_key}")
    source_cell_types = np.asarray(primary_proc.obs[cell_key], dtype=object)
    merge_cell_types_cfg = source_label_settings["cell_type_merge"]
    if bool(merge_cell_types_cfg.get("enabled", False)):
        override_key = str(merge_cell_types_cfg.get("override_key", "neuron_type")).strip()
        if override_key in primary_proc.obs.columns:
            source_cell_types = _merge_cell_labels(
                source_cell_types,
                np.asarray(primary_proc.obs[override_key], dtype=object),
                na_tokens=merge_cell_types_cfg.get("na_tokens", None),
            )
    source_cell_types = source_cell_types[init_indices].astype(str)
    time0_type_used, time0_keep_mask = _exact_cell_type_mask(source_cell_types, str(cell_used_in))

    selection_source = selection_mode
    if selection_mode == "endpoint":
        if endpoint_keep_mask is None:
            raise ValueError("endpoint selection requires endpoint simulation")
        keep_mask = np.asarray(endpoint_keep_mask, dtype=bool)
    elif selection_mode == "time0":
        keep_mask = np.asarray(time0_keep_mask, dtype=bool)
        cell_type_used = str(time0_type_used if time0_type_used else cell_used_in)
    elif selection_mode == "auto":
        if endpoint_keep_mask is not None and bool(np.any(endpoint_keep_mask)):
            keep_mask = np.asarray(endpoint_keep_mask, dtype=bool)
            selection_source = "endpoint"
        elif bool(np.any(time0_keep_mask)):
            keep_mask = np.asarray(time0_keep_mask, dtype=bool)
            cell_type_used = str(time0_type_used)
            selection_source = "time0"
        else:
            keep_mask = np.ones(int(len(init_indices)), dtype=bool)
            selection_source = "random"
    else:
        keep_mask = np.ones(int(len(init_indices)), dtype=bool)
        selection_source = "random"

    sel_traj_idx = np.where(np.asarray(keep_mask, dtype=bool))[0]
    sel_source_idx = init_indices[sel_traj_idx]

    rows = pd.DataFrame(
        {
            "trajectory_index": sel_traj_idx.astype(int),
            "source_cell_index": sel_source_idx.astype(int),
            "predicted_label": pred_labels[sel_traj_idx].astype(str),
            "target_prob": target_prob[sel_traj_idx].astype(float),
            "max_prob": max_prob[sel_traj_idx].astype(float),
            "target_group": str(group_used),
            "target_genes": "|".join(target_genes),
            "z_score": float(z_used),
            "time_requested": float(t_req),
            "time_used": float(t_used),
            "cell_type_requested": str(cell_used_in),
            "cell_type_used": str(cell_type_used),
            "label_rule": str(selection_mode),
            "selection_source": str(selection_source),
            "primary_domain": int(ctx.primary_domain),
            "secondary_domain": int(ctx.secondary_domain),
        }
    )

    if str(output_csv).strip():
        out_csv = Path(output_csv).resolve()
    else:
        g_slug = _slugify(group_used)
        x_slug = _slugify(cell_type_used)
        z_slug = _signed_float_token(float(z_used))
        t_slug = _slugify(f"{float(t_used):g}")
        out_csv = ctx.output_table_dir / f"perturbation_selected_indices__g-{g_slug}__z-{z_slug}__t-{t_slug}__x-{x_slug}.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_csv, index=False)

    summary = {
        "run_dir": str(ctx.run_dir),
        "target_group": str(group_used),
        "target_genes": target_genes,
        "z_score": float(z_used),
        "time_requested": float(t_req),
        "time_used": float(t_used),
        "cell_type_requested": str(cell_used_in),
        "cell_type_used": str(cell_type_used),
        "label_rule": str(selection_mode),
        "selection_source": str(selection_source),
        "n_total_trajectories": int(len(init_indices)),
        "n_selected": int(len(sel_traj_idx)),
        "output_csv": str(out_csv),
    }
    summary.update(source_label_settings)
    out_json = out_csv.with_suffix(".json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary["summary_json"] = str(out_json)
    return summary


def _to_2d_series(arr3: np.ndarray, reducer: Any, method: str) -> np.ndarray:
    arr = np.asarray(arr3, dtype=float)
    t_num, n_cells, dim = arr.shape
    flat = arr.reshape(t_num * n_cells, dim)
    method_norm = str(method).strip().lower()
    if method_norm == "umap":
        out = apply_umap_transform(reducer, flat).reshape(t_num, n_cells, 2)
    else:
        out = reducer.transform(flat).reshape(t_num, n_cells, 2)
    return np.asarray(out, dtype=float)


def _group_background_by_time(
    emb: np.ndarray,
    times: np.ndarray,
    labels: Sequence[str],
) -> tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    uniq = np.sort(np.unique(np.asarray(times, dtype=float)))
    x_by_t: List[np.ndarray] = []
    y_by_t: List[np.ndarray] = []
    for t in uniq:
        m = np.isclose(np.asarray(times, dtype=float), float(t))
        x_by_t.append(np.asarray(emb[m], dtype=float))
        y_by_t.append(np.asarray(np.asarray(labels)[m], dtype=object))
    return x_by_t, y_by_t, uniq


def _build_dense_time_grid(native_grid: Sequence[float], n_points: int) -> np.ndarray:
    native = np.asarray(native_grid, dtype=float).reshape(-1)
    if native.size <= 1:
        return native.copy()
    n_use = int(max(int(n_points), int(native.size)))
    if n_use == int(native.size):
        return native.copy()
    return np.linspace(float(native[0]), float(native[-1]), num=n_use, dtype=float)


def _interp_traj_array_linear(traj_array: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj_array, dtype=float)
    src = np.asarray(src_times, dtype=float).reshape(-1)
    dst = np.asarray(dst_times, dtype=float).reshape(-1)
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"traj_array must have shape (T,N,2), got {traj.shape}")
    if src.shape[0] != traj.shape[0]:
        raise ValueError("src_times length must match traj_array first dimension")
    if dst.shape[0] == src.shape[0] and np.allclose(dst, src):
        return traj.copy()
    out = np.empty((dst.shape[0], traj.shape[1], traj.shape[2]), dtype=float)
    for cell_idx in range(traj.shape[1]):
        for dim_idx in range(traj.shape[2]):
            out[:, cell_idx, dim_idx] = np.interp(dst, src, traj[:, cell_idx, dim_idx])
    return out


def _sample_background_indices_by_time(
    times: np.ndarray,
    *,
    max_cells: int,
    seed: int,
) -> np.ndarray:
    time_arr = np.asarray(times, dtype=float).reshape(-1)
    n_obs = int(time_arr.size)
    if max_cells <= 0 or n_obs <= max_cells:
        return np.arange(n_obs, dtype=int)

    rng = np.random.default_rng(int(seed))
    uniq = np.sort(np.unique(time_arr))
    if uniq.size == 0:
        return np.arange(n_obs, dtype=int)

    selected: List[np.ndarray] = []
    remaining = int(max_cells)
    total = int(n_obs)
    for i, t in enumerate(uniq):
        idx = np.flatnonzero(np.isclose(time_arr, float(t)))
        if idx.size == 0:
            continue
        if i == uniq.size - 1:
            take = min(int(idx.size), int(remaining))
        else:
            take = max(1, int(round(max_cells * (idx.size / max(total, 1)))))
            take = min(int(idx.size), int(take), int(remaining))
        if take <= 0:
            continue
        picked = np.asarray(rng.choice(idx, size=take, replace=False), dtype=int)
        selected.append(picked)
        remaining -= int(take)
        total -= int(idx.size)
        if remaining <= 0:
            break

    if not selected:
        return np.arange(min(n_obs, max_cells), dtype=int)
    out = np.sort(np.concatenate(selected).astype(int, copy=False))
    if out.size > max_cells:
        out = np.sort(np.asarray(rng.choice(out, size=int(max_cells), replace=False), dtype=int))
    return out


def plot_perturbation_trajectories(
    ctx: TFRunContext,
    *,
    indices_csv: str,
    scores: Optional[Sequence[float]] = None,
    sample_n: int = 30,
    seed: int = 42,
    device: str = "",
    output_dir: str = "",
    dim_reduction: str = "umap",
    umap_n_neighbors: int = 10,
    umap_min_dist: float = 0.2,
    umap_random_state: int = 42,
    umap_force_retrain: bool = False,
    background_max_cells: int = 30000,
    background_seed: int = 42,
    background_map_batch: int = 2048,
    modalities: Sequence[str] = ("primary", "secondary"),
    dense_time_n_points: int = 49,
    plot_interp_n_points: int = 97,
) -> Dict[str, Any]:
    idx_csv = Path(indices_csv).resolve()
    out_dir = Path(output_dir).resolve() if str(output_dir).strip() else (ctx.output_figure_dir / "perturb_index_trajectory")
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_reduction_used = str(dim_reduction).strip().lower() or "pca"
    if dim_reduction_used not in {"pca", "umap"}:
        raise ValueError(f"Unsupported dim_reduction={dim_reduction!r}; expected 'pca' or 'umap'")
    modalities_use = {str(value).strip().lower() for value in modalities}
    if not modalities_use or not modalities_use.issubset({"primary", "secondary"}):
        raise ValueError("modalities must contain one or both of: primary, secondary")

    indices_df = pd.read_csv(idx_csv)
    if indices_df.empty:
        raise ValueError(f"indices csv is empty: {idx_csv}")

    target_group = str(indices_df["target_group"].iloc[0])
    target_genes = [x.strip() for x in str(indices_df["target_genes"].iloc[0]).split("|") if x.strip()]
    if not target_genes:
        raise ValueError("target_genes parsed from indices csv is empty")

    source_idx_all = np.asarray(indices_df["source_cell_index"], dtype=int)
    source_idx_all = np.unique(source_idx_all)
    if source_idx_all.size == 0:
        raise ValueError("No source_cell_index in indices csv")

    rng = np.random.default_rng(int(seed))
    n_take = int(min(max(int(sample_n), 1), source_idx_all.size))
    sample_idx = np.asarray(rng.choice(source_idx_all, size=n_take, replace=False), dtype=int)

    sweep_df = _load_sweep_df(ctx)
    if scores is None:
        z_vals = sorted({float(x) for x in sweep_df["z_score"].astype(float).tolist()})
        scores_use = z_vals
    else:
        scores_use = sorted({float(x) for x in scores})

    primary_proc, secondary_proc = ctx.load_primary_secondary_processed(backed=False)
    model = ctx.load_model()
    run_device = str(device).strip() or "cpu"
    mapper = ctx.build_mapper(device=run_device)

    primary_bundle, _ = ctx.load_primary_secondary_bundles()
    x0_latent = np.asarray(primary_proc.obsm["X_latent"][sample_idx], dtype=np.float32)
    x0_feat = _reconstruct_latent_to_feature(x0_latent, primary_bundle).astype(np.float32, copy=True)
    feat_lut = {str(g): i for i, g in enumerate(primary_bundle.var_names)}
    feat_std = np.asarray(np.std(x0_feat, axis=0), dtype=np.float32)

    valid_gene_items: List[tuple[str, int, float]] = []
    for g in target_genes:
        idx = feat_lut.get(str(g), None)
        if idx is None:
            continue
        sd = float(feat_std[int(idx)])
        if sd < 1e-8:
            continue
        valid_gene_items.append((str(g), int(idx), sd))
    if not valid_gene_items:
        raise ValueError(f"No valid target genes in primary space. requested={target_genes}")

    obs_times_full = np.asarray(primary_proc.obs[ctx.time_key], dtype=float)
    obs_types_full = np.asarray(primary_proc.obs[ctx.cell_type_key]).astype(str)
    bg_idx = _sample_background_indices_by_time(
        obs_times_full,
        max_cells=int(background_max_cells),
        seed=int(background_seed),
    )
    obs_times = np.asarray(obs_times_full[bg_idx], dtype=float)
    obs_types = np.asarray(obs_types_full[bg_idx]).astype(str)
    x_primary_bg = np.asarray(primary_proc.obsm["X_latent"][bg_idx], dtype=np.float32)
    if dim_reduction_used == "umap":
        primary_umap_path = out_dir / "umap_model_primary.pkl"
        primary_reducer = get_or_train_umap(
            data=x_primary_bg,
            save_path=str(primary_umap_path),
            n_neighbors=int(umap_n_neighbors),
            min_dist=float(umap_min_dist),
            n_components=2,
            random_state=int(umap_random_state),
            force_retrain=bool(umap_force_retrain),
        )
        x_primary_bg_2d = np.asarray(apply_umap_transform(primary_reducer, x_primary_bg), dtype=float)
    else:
        primary_umap_path = None
        primary_reducer = PCA(n_components=2, random_state=42)
        primary_reducer.fit(x_primary_bg)
        x_primary_bg_2d = np.asarray(primary_reducer.transform(x_primary_bg), dtype=float)
    bg_primary_x, bg_primary_labels, bg_ticks = _group_background_by_time(x_primary_bg_2d, obs_times, obs_types)
    # 只保留指定的细胞类型
    # keep_types = {"HSC", "EryP", "MasP", "MkP"}
    # # 过滤 primary 背景
    # filtered_primary_x = []
    # filtered_primary_labels = []
    # for x_t, lab_t in zip(bg_primary_x, bg_primary_labels):
    #     mask = np.isin(lab_t, list(keep_types))
    #     filtered_primary_x.append(x_t[mask])
    #     filtered_primary_labels.append(lab_t[mask])
    # bg_primary_x = filtered_primary_x
    # bg_primary_labels = filtered_primary_labels

    x_map_bg = _map_main_to_sub_like(mapper, x_primary_bg, map_batch=int(background_map_batch))
    if dim_reduction_used == "umap":
        secondary_umap_path = out_dir / "umap_model_secondary.pkl"
        secondary_reducer = get_or_train_umap(
            data=x_map_bg,
            save_path=str(secondary_umap_path),
            n_neighbors=int(umap_n_neighbors),
            min_dist=float(umap_min_dist),
            n_components=2,
            random_state=int(umap_random_state),
            force_retrain=bool(umap_force_retrain),
        )
        x_map_bg_2d = np.asarray(apply_umap_transform(secondary_reducer, x_map_bg), dtype=float)
    else:
        secondary_umap_path = None
        secondary_reducer = PCA(n_components=2, random_state=42)
        secondary_reducer.fit(x_map_bg)
        x_map_bg_2d = np.asarray(secondary_reducer.transform(x_map_bg), dtype=float)
    bg_map_x, bg_map_labels, _ = _group_background_by_time(x_map_bg_2d, obs_times, obs_types)
    # # 同样加上：
    # filtered_map_x = []
    # filtered_map_labels = []
    # for x_t, lab_t in zip(bg_map_x, bg_map_labels):
    #     mask = np.isin(lab_t, list(keep_types))
    #     filtered_map_x.append(x_t[mask])
    #     filtered_map_labels.append(lab_t[mask])
    # bg_map_x = filtered_map_x
    # bg_map_labels = filtered_map_labels


    
    native_time_grid = np.asarray(ctx.time_grid, dtype=float)
    time_grid = _build_dense_time_grid(native_time_grid, int(dense_time_n_points))
    plot_time_grid = _build_dense_time_grid(time_grid, int(plot_interp_n_points))
    dt = float(ctx.manifest.get("dt", 0.05))

    generated: List[str] = []
    for z in scores_use:
        x0_pert = x0_feat.copy()
        for _, idx, sd in valid_gene_items:
            x0_pert[:, int(idx)] = x0_pert[:, int(idx)] + float(z) * float(sd)
        z0 = _project_feature_to_latent(x0_pert, primary_bundle)

        traj_primary = _build_ode_trajectory(
            model=model,
            x0=z0,
            times=time_grid,
            adata=ctx.load_dynamic_adata(),
            dt=dt,
            device=run_device,
        )
        traj_secondary = _map_to_secondary(mapper, traj_primary["main_latent"])

        traj_primary_2d = _to_2d_series(
            np.asarray(traj_primary["main_latent"], dtype=np.float32),
            primary_reducer,
            dim_reduction_used,
        )
        traj_secondary_2d = _to_2d_series(
            np.asarray(traj_secondary["sub_latent"], dtype=np.float32),
            secondary_reducer,
            dim_reduction_used,
        )
        traj_primary_plot_2d = _interp_traj_array_linear(traj_primary_2d, time_grid, plot_time_grid)
        traj_secondary_plot_2d = _interp_traj_array_linear(traj_secondary_2d, time_grid, plot_time_grid)

        g_slug = _slugify(target_group)
        z_slug = _signed_float_token(float(z))
        dr_slug = _slugify(dim_reduction_used)
        primary_png = out_dir / f"perturb_index_traj_primary__g-{g_slug}__z-{z_slug}__dr-{dr_slug}.pdf"
        secondary_png = out_dir / f"perturb_index_traj_secondary__g-{g_slug}__z-{z_slug}__dr-{dr_slug}.pdf"

        if "primary" in modalities_use:
            plot_ode_v3_core(
                X=bg_primary_x,
                traj_array=traj_primary_plot_2d,
                save_path=str(primary_png),
                traj_times=plot_time_grid,
                time_ticks=bg_ticks,
                bg_obs_values=bg_primary_labels,
                bg_obs_name=ctx.cell_type_key,
                bg_palette="Set2",
                traj_cmap="sunset1",
                background_mode="auto",
                show_axes=True,
            )
            generated.append(str(primary_png))
        if "secondary" in modalities_use:
            plot_ode_v3_core(
                X=bg_map_x,
                traj_array=traj_secondary_plot_2d,
                save_path=str(secondary_png),
                traj_times=plot_time_grid,
                time_ticks=bg_ticks,
                bg_obs_values=bg_map_labels,
                bg_obs_name=ctx.cell_type_key,
                bg_palette="Set2",
                traj_cmap="sunset1",
                background_mode="auto",
                show_axes=True,
            )
            generated.append(str(secondary_png))

    summary = {
        "run_dir": str(ctx.run_dir),
        "indices_csv": str(idx_csv),
        "indices_usage_mode": "trajectory_visualization_only",
        "target_group": target_group,
        "target_genes": target_genes,
        "sample_n": int(n_take),
        "sampled_source_indices": [int(x) for x in sample_idx.tolist()],
        "background_n_cells": int(bg_idx.size),
        "background_max_cells": int(background_max_cells),
        "background_seed": int(background_seed),
        "background_map_batch": int(background_map_batch),
        "native_time_grid": [float(x) for x in native_time_grid],
        "dense_time_grid": [float(x) for x in time_grid],
        "plot_time_grid": [float(x) for x in plot_time_grid],
        "native_time_grid_n_points": int(native_time_grid.size),
        "dense_time_grid_n_points": int(time_grid.size),
        "plot_time_grid_n_points": int(plot_time_grid.size),
        "dense_time_n_points": int(dense_time_n_points),
        "plot_interp_n_points": int(plot_interp_n_points),
        "scores": [float(x) for x in scores_use],
        "modalities": sorted(modalities_use),
        "dim_reduction": str(dim_reduction_used),
        "umap_n_neighbors": int(umap_n_neighbors) if dim_reduction_used == "umap" else None,
        "umap_min_dist": float(umap_min_dist) if dim_reduction_used == "umap" else None,
        "umap_random_state": int(umap_random_state) if dim_reduction_used == "umap" else None,
        "umap_model_primary": str(primary_umap_path) if primary_umap_path is not None else "",
        "umap_model_secondary": str(secondary_umap_path) if secondary_umap_path is not None else "",
        "generated": generated,
        "output_dir": str(out_dir),
    }
    summary_json = out_dir / "perturb_index_trajectory_summary_tf.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary["summary_json"] = str(summary_json)
    return summary
