from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

from CytoBridge.tl.analysis_dense_time import (
    _batched_main_to_sub_terms_autograd,
    _batched_mapper_reverse_grad_autograd,
    _batched_velocity,
    _batched_velocity_jacobian,
    _build_classifier_space_latent,
    _build_ode_trajectory,
    _map_to_secondary,
)

from .context import TFRunContext


def _normalize_obs_key(raw: Any) -> str:
    key = str(raw).strip().lower()
    if key in {"time", "time_point", "pred_time", "time_point_processed"}:
        return "time_point_processed"
    if key in {"cell_type", "celltype"}:
        return "cell_type"
    if key in {"cell_type1", "cell_type_1", "celltype1"}:
        return "cell_type1"
    return key


def _to_label_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(x) for x in values], dtype=object)


def _merge_cell_type_labels(
    *,
    base_labels: Sequence[Any],
    override_labels: Sequence[Any],
    na_tokens: Sequence[str],
) -> np.ndarray:
    base = _to_label_array(base_labels)
    override = _to_label_array(override_labels)
    if base.shape[0] != override.shape[0]:
        raise ValueError(f"base/override length mismatch: {base.shape[0]} vs {override.shape[0]}")
    na_set = {str(x).strip().lower() for x in na_tokens}
    out = base.copy()
    keep = np.asarray([str(x).strip().lower() not in na_set for x in override], dtype=bool)
    out[keep] = override[keep]
    return out


def _sample_time0_indices(
    *,
    times: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("empty time array")
    if np.any(np.isclose(t, 0.0, atol=1e-8)):
        t0 = 0.0
    else:
        t0 = float(np.nanmin(t))
    pool = np.where(np.isclose(t, float(t0), atol=1e-8))[0]
    if pool.size == 0:
        raise ValueError("no cells found at t0")
    n_take = int(min(5000, max(1, int(pool.size // 2))))
    rng = np.random.default_rng(int(seed))
    idx = np.asarray(rng.choice(pool, size=n_take, replace=False), dtype=int)
    return np.sort(idx), {
        "t0_value": float(t0),
        "t0_pool_n": int(pool.size),
        "sample_n": int(n_take),
        "sample_rule": "min(5000, floor(time0_n/2))",
    }


def _predict_labels_3d(
    *,
    classifier: Any,
    X_3d: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(X_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"classifier input must be 3D, got shape={arr.shape}")
    t_num = int(arr.shape[0])
    out = []
    for i in range(t_num):
        out.append(_to_label_array(classifier.predict(arr[i])))
    return np.stack(out, axis=0)


def _train_cell_type1_classifier(
    *,
    ctx: TFRunContext,
    train_space: str,
    map_batch: int,
    device: str,
    base_key: str,
    override_key: str,
    na_tokens: Sequence[str],
    max_iter: int,
) -> Tuple[Any, Dict[str, Any]]:
    primary_proc, secondary_proc = ctx.load_primary_secondary_processed(backed=False)
    if base_key not in primary_proc.obs.columns:
        raise KeyError(f"missing obs key for cell_type1 base: {base_key}")
    if override_key not in primary_proc.obs.columns:
        raise KeyError(f"missing obs key for cell_type1 override: {override_key}")

    y = _merge_cell_type_labels(
        base_labels=np.asarray(primary_proc.obs[base_key], dtype=object),
        override_labels=np.asarray(primary_proc.obs[override_key], dtype=object),
        na_tokens=na_tokens,
    ).astype(str)

    mapper = ctx.build_mapper(device=device)
    X_main = np.asarray(primary_proc.obsm["X_latent"], dtype=np.float32)
    X_sub = None
    if "X_latent" in secondary_proc.obsm and int(secondary_proc.n_obs) == int(primary_proc.n_obs):
        X_sub = np.asarray(secondary_proc.obsm["X_latent"], dtype=np.float32)

    X_train = _build_classifier_space_latent(
        train_space=str(train_space),
        main_latent=X_main,
        sub_latent=X_sub,
        mapper=mapper,
        map_batch=int(map_batch),
    )
    if len(np.unique(y)) < 2:
        raise ValueError("cell_type1 training needs >=2 classes")

    clf = LogisticRegression(
        max_iter=int(max_iter),
        solver="lbfgs",
        multi_class="auto",
        class_weight="balanced",
    )
    clf.fit(X_train, y)
    train_acc = float(np.mean(clf.predict(X_train) == y))
    meta = {
        "train_space": str(train_space),
        "feature_dim": int(X_train.shape[1]),
        "num_classes": int(len(np.unique(y))),
        "classes": [str(x) for x in clf.classes_.tolist()],
        "train_accuracy": float(train_acc),
        "label_rule": f"merge({base_key},{override_key})",
        "na_tokens": [str(x) for x in na_tokens],
    }
    return clf, meta


def _group_selected_by_time(
    *,
    selected_pairs: np.ndarray,
    t_num: int,
) -> List[np.ndarray]:
    out: List[np.ndarray] = [np.zeros((0,), dtype=int) for _ in range(int(t_num))]
    if selected_pairs.size == 0:
        return out
    for t_idx in range(int(t_num)):
        mask = selected_pairs[:, 0] == int(t_idx)
        out[t_idx] = np.asarray(selected_pairs[mask, 1], dtype=int)
    return out


def _compute_bidirectional_latent_jacobians(
    *,
    ctx: TFRunContext,
    traj_main: np.ndarray,
    selected_by_time: List[np.ndarray],
    device: str,
    mapper_batch_size: int,
    include_hessian_term: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    model = ctx.load_model()
    mapper = ctx.build_mapper(device=device)
    time_grid = np.asarray(ctx.time_grid, dtype=float).reshape(-1)
    t_num = int(traj_main.shape[0])
    if t_num != int(time_grid.shape[0]):
        raise ValueError(f"time grid mismatch: traj={t_num}, grid={time_grid.shape[0]}")

    sum_main_to_sub: Optional[np.ndarray] = None
    sum_sub_to_main: Optional[np.ndarray] = None
    n_total = 0
    used_time_points: List[float] = []

    for t_idx in range(t_num):
        idx = np.asarray(selected_by_time[t_idx], dtype=int)
        if idx.size == 0:
            continue
        x_main = np.asarray(traj_main[t_idx, idx], dtype=np.float32)
        t_val = float(time_grid[t_idx])
        xdot = _batched_velocity(model, x_main, float(t_val), device=device)
        jac_z = _batched_velocity_jacobian(model, x_main, float(t_val), device=device)
        _, term1, term2, y_sub = _batched_main_to_sub_terms_autograd(
            mapper=mapper,
            x_main=x_main,
            xdot_main=xdot,
            jac_z=jac_z,
            batch_size=int(mapper_batch_size),
            include_hessian_term=bool(include_hessian_term),
            device=device,
        )
        j_main_to_sub_cell = np.asarray((term1 + term2) if include_hessian_term else term2, dtype=np.float32)
        # raw shape (n, sub_target, main_source) -> (n, main_source, sub_target)
        j_main_to_sub_cell = np.asarray(np.transpose(j_main_to_sub_cell, (0, 2, 1)), dtype=np.float32)

        jac_map_reverse = _batched_mapper_reverse_grad_autograd(
            mapper=mapper,
            y_sub=np.asarray(y_sub, dtype=np.float32),
            batch_size=int(mapper_batch_size),
            device=device,
        )  # (n, main_target, sub_source)
        j_sub_to_main_cell = np.einsum("bij,bjk->bik", jac_z, jac_map_reverse)  # (n, main_target, sub_source)
        j_sub_to_main_cell = np.asarray(np.transpose(j_sub_to_main_cell, (0, 2, 1)), dtype=np.float32)  # (n, sub_source, main_target)

        batch_main_to_sub = np.asarray(np.sum(j_main_to_sub_cell, axis=0), dtype=np.float64)
        batch_sub_to_main = np.asarray(np.sum(j_sub_to_main_cell, axis=0), dtype=np.float64)
        if sum_main_to_sub is None:
            sum_main_to_sub = np.zeros_like(batch_main_to_sub, dtype=np.float64)
        elif sum_main_to_sub.shape != batch_main_to_sub.shape:
            raise ValueError(
                f"inconsistent main->sub Jacobian shape across batches: "
                f"expected {sum_main_to_sub.shape}, got {batch_main_to_sub.shape}"
            )
        if sum_sub_to_main is None:
            sum_sub_to_main = np.zeros_like(batch_sub_to_main, dtype=np.float64)
        elif sum_sub_to_main.shape != batch_sub_to_main.shape:
            raise ValueError(
                f"inconsistent sub->main Jacobian shape across batches: "
                f"expected {sum_sub_to_main.shape}, got {batch_sub_to_main.shape}"
            )

        sum_main_to_sub += batch_main_to_sub
        sum_sub_to_main += batch_sub_to_main
        n_total += int(idx.size)
        used_time_points.append(float(t_val))

    if n_total <= 0:
        raise ValueError("no selected simulated cells for Jacobian aggregation")
    if sum_main_to_sub is None or sum_sub_to_main is None:
        raise ValueError("Jacobian aggregation failed to initialize")
    j_ms = np.asarray(sum_main_to_sub / float(n_total), dtype=np.float32)
    j_sm = np.asarray(sum_sub_to_main / float(n_total), dtype=np.float32)
    meta = {
        "n_selected_cells_effective": int(n_total),
        "n_time_points_used": int(len(used_time_points)),
        "time_points_used": [float(x) for x in sorted(set(used_time_points))],
    }
    return j_ms, j_sm, meta


def _fit_umap_and_save(
    *,
    matrix: np.ndarray,
    var_names: Sequence[str],
    output_csv: Path,
    output_png: Path,
    title: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> Dict[str, Any]:
    if umap is None:
        raise RuntimeError("umap-learn is required but unavailable")
    X = np.asarray(matrix, dtype=np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, int(n_neighbors)),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(seed),
    )
    emb = np.asarray(reducer.fit_transform(X), dtype=float)
    df = pd.DataFrame(
        {
            "variable": [str(x) for x in var_names],
            "umap1": emb[:, 0],
            "umap2": emb[:, 1],
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    plt.figure(figsize=(8.0, 6.4), dpi=260)
    plt.scatter(df["umap1"], df["umap2"], s=8, alpha=0.85, linewidths=0.0)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(str(title))
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }


def build_regulation_umap_outputs(
    *,
    ctx: TFRunContext,
    out_fig_dir: Path,
    cfg_obj: Dict[str, Any],
) -> Dict[str, Any]:
    reg_cfg = cfg_obj.get("regulation_umap", {})
    if not isinstance(reg_cfg, dict) or (not bool(reg_cfg.get("enabled", False))):
        return {"enabled": False}
    out_dir = out_fig_dir / "regulation_umap"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        obs_key = _normalize_obs_key(reg_cfg.get("obs_key", "time_point_processed"))
        obs_value = reg_cfg.get("obs_value", None)
        if obs_value is None:
            raise ValueError("regulation_umap.obs_value is required")

        seed = int(reg_cfg.get("seed", 42))
        device = str(reg_cfg.get("device", cfg_obj.get("device", "cpu"))).strip() or "cpu"
        mapper_batch_size = int(reg_cfg.get("mapper_batch_size", 128))
        include_hessian_term = bool(reg_cfg.get("include_hessian_term", False))
        max_selected_cells = int(reg_cfg.get("max_selected_cells", 512))
        umap_neighbors = int(reg_cfg.get("umap_neighbors", 30))
        umap_min_dist = float(reg_cfg.get("umap_min_dist", 0.25))
        umap_seed = int(reg_cfg.get("umap_seed", seed))

        celltype_time_scope = str(reg_cfg.get("celltype_time_scope", "all_times")).strip().lower() or "all_times"
        if celltype_time_scope not in {"all_times"}:
            celltype_time_scope = "all_times"

        primary_proc, secondary_proc = ctx.load_primary_secondary_processed(backed=False)
        primary_bundle, secondary_bundle = ctx.load_primary_secondary_bundles()
        times = np.asarray(primary_proc.obs[ctx.time_key], dtype=float)
        sampled_idx, sampling_meta = _sample_time0_indices(times=times, seed=seed)
        x0 = np.asarray(primary_proc.obsm["X_latent"][sampled_idx], dtype=np.float32)

        traj_main_obj = _build_ode_trajectory(
            model=ctx.load_model(),
            x0=x0,
            times=np.asarray(ctx.time_grid, dtype=float),
            adata=ctx.load_dynamic_adata(),
            dt=float(ctx.manifest.get("dt", 0.05)),
            device=device,
        )
        traj_sub_obj = _map_to_secondary(ctx.build_mapper(device=device), traj_main_obj["main_latent"])
        traj_main = np.asarray(traj_main_obj["main_latent"], dtype=np.float32)
        traj_sub = np.asarray(traj_sub_obj["sub_latent"], dtype=np.float32)

        t_num, n_cells = int(traj_main.shape[0]), int(traj_main.shape[1])
        selected_pairs: np.ndarray
        classifier_eval: Dict[str, Any] = {}

        if obs_key == "time_point_processed":
            target_t = float(obs_value)
            tg = np.asarray(ctx.time_grid, dtype=float)
            t_idx = int(np.argmin(np.abs(tg - target_t)))
            selected_pairs = np.asarray([[t_idx, j] for j in range(n_cells)], dtype=int)
            selection_meta = {
                "mode": "time_point_processed",
                "time_requested": float(target_t),
                "time_used": float(tg[t_idx]),
                "n_selected_pre_cap": int(selected_pairs.shape[0]),
            }
        elif obs_key in {"cell_type", "cell_type1"}:
            if obs_key == "cell_type":
                cls_bundle = ctx.load_classifier_bundle()
                classifier = cls_bundle["classifier"]
                train_space = str(cls_bundle["train_space"])
                map_batch = int(cls_bundle["map_batch"])
                classifier_eval = {
                    "source": "existing_classifier_bundle",
                    "train_space": str(train_space),
                    "classes": [str(x) for x in np.asarray(getattr(classifier, "classes_", []), dtype=object).tolist()],
                }
            else:
                train_space = str(reg_cfg.get("classifier_train_space", "joint")).strip().lower() or "joint"
                map_batch = int(reg_cfg.get("classifier_map_batch", 2048))
                base_key = str(reg_cfg.get("cell_type1_merge_base_key", "cell_type")).strip() or "cell_type"
                override_key = str(reg_cfg.get("cell_type1_merge_override_key", "neuron_type")).strip() or "neuron_type"
                na_tokens = reg_cfg.get("cell_type1_na_tokens", ["", "NA", "na", "nan", "None", "null"])
                if isinstance(na_tokens, str):
                    na_tokens = [x.strip() for x in na_tokens.split(",") if x.strip()]
                classifier, classifier_eval = _train_cell_type1_classifier(
                    ctx=ctx,
                    train_space=train_space,
                    map_batch=map_batch,
                    device=device,
                    base_key=base_key,
                    override_key=override_key,
                    na_tokens=[str(x) for x in na_tokens],
                    max_iter=int(reg_cfg.get("classifier_max_iter", 1200)),
                )

            cls_latent = _build_classifier_space_latent(
                train_space=str(train_space),
                main_latent=np.asarray(traj_main, dtype=np.float32),
                sub_latent=np.asarray(traj_sub, dtype=np.float32),
                mapper=ctx.build_mapper(device=device),
                map_batch=int(map_batch),
            )
            pred_labels = _predict_labels_3d(classifier=classifier, X_3d=cls_latent)
            target_label = str(obs_value)
            matched = np.asarray(pred_labels == target_label, dtype=bool)
            selected_pairs = np.argwhere(matched)
            selection_meta = {
                "mode": str(obs_key),
                "target_label": str(target_label),
                "time_scope": str(celltype_time_scope),
                "n_selected_pre_cap": int(selected_pairs.shape[0]),
            }
        else:
            raise ValueError(f"unsupported regulation_umap.obs_key={obs_key}")

        if selected_pairs.size == 0:
            return {
                "enabled": True,
                "status": "skipped",
                "reason": "no selected simulated cells",
                "obs_key": str(obs_key),
                "obs_value": str(obs_value),
            }

        rng = np.random.default_rng(int(seed))
        if int(max_selected_cells) > 0 and int(selected_pairs.shape[0]) > int(max_selected_cells):
            keep = np.sort(rng.choice(np.arange(selected_pairs.shape[0]), size=int(max_selected_cells), replace=False))
            selected_pairs = np.asarray(selected_pairs[keep], dtype=int)
        selection_meta["n_selected_post_cap"] = int(selected_pairs.shape[0])

        selected_by_time = _group_selected_by_time(selected_pairs=selected_pairs, t_num=t_num)
        j_main_to_sub, j_sub_to_main, jac_meta = _compute_bidirectional_latent_jacobians(
            ctx=ctx,
            traj_main=traj_main,
            selected_by_time=selected_by_time,
            device=device,
            mapper_batch_size=int(mapper_batch_size),
            include_hessian_term=bool(include_hessian_term),
        )

        # User-confirmed formula:
        # main_variable_matrix = main_forward @ J_main_to_sub
        # sub_variable_matrix  = sub_forward  @ J_sub_to_main
        main_forward = np.asarray(primary_bundle.forward, dtype=np.float32)  # (G_main, 50)
        sub_forward = np.asarray(secondary_bundle.forward, dtype=np.float32)  # (G_sub, 50)
        main_var_matrix = np.asarray(main_forward @ j_main_to_sub, dtype=np.float32)  # (G_main, 50)
        sub_var_matrix = np.asarray(sub_forward @ j_sub_to_main, dtype=np.float32)  # (G_sub, 50)

        latent_ms_csv = out_dir / "latent_jacobian_main_to_sub.csv"
        latent_sm_csv = out_dir / "latent_jacobian_sub_to_main.csv"
        main_mat_csv = out_dir / "main_variable_matrix.csv"
        sub_mat_csv = out_dir / "sub_variable_matrix.csv"

        pd.DataFrame(
            j_main_to_sub,
            index=[f"main_latent_{i}" for i in range(j_main_to_sub.shape[0])],
            columns=[f"sub_latent_{i}" for i in range(j_main_to_sub.shape[1])],
        ).to_csv(latent_ms_csv, index=True, index_label="source_main_latent")
        pd.DataFrame(
            j_sub_to_main,
            index=[f"sub_latent_{i}" for i in range(j_sub_to_main.shape[0])],
            columns=[f"main_latent_{i}" for i in range(j_sub_to_main.shape[1])],
        ).to_csv(latent_sm_csv, index=True, index_label="source_sub_latent")

        pd.DataFrame(main_var_matrix, index=[str(x) for x in primary_bundle.var_names]).to_csv(main_mat_csv, index=True, index_label="main_variable")
        pd.DataFrame(sub_var_matrix, index=[str(x) for x in secondary_bundle.var_names]).to_csv(sub_mat_csv, index=True, index_label="sub_variable")

        main_umap_csv = out_dir / "main_variable_umap.csv"
        main_umap_png = out_dir / "main_variable_umap.pdf"
        sub_umap_csv = out_dir / "sub_variable_umap.csv"
        sub_umap_png = out_dir / "sub_variable_umap.pdf"

        main_umap_meta = _fit_umap_and_save(
            matrix=main_var_matrix,
            var_names=[str(x) for x in primary_bundle.var_names],
            output_csv=main_umap_csv,
            output_png=main_umap_png,
            title="Main Variable UMAP (forward @ J_main_to_sub)",
            seed=int(umap_seed),
            n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
        )
        sub_umap_meta = _fit_umap_and_save(
            matrix=sub_var_matrix,
            var_names=[str(x) for x in secondary_bundle.var_names],
            output_csv=sub_umap_csv,
            output_png=sub_umap_png,
            title="Sub Variable UMAP (forward @ J_sub_to_main)",
            seed=int(umap_seed),
            n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
        )

        if classifier_eval:
            cls_eval_path = out_dir / "classifier_cell_type1_eval.json"
            with cls_eval_path.open("w", encoding="utf-8") as f:
                json.dump(classifier_eval, f, ensure_ascii=False, indent=2)
            classifier_eval["output_json"] = str(cls_eval_path)

        summary = {
            "enabled": True,
            "status": "success",
            "obs_key": str(obs_key),
            "obs_value": str(obs_value),
            "sampling": sampling_meta,
            "selection": selection_meta,
            "jacobian": jac_meta,
            "projection_formula": {
                "main": "main_forward @ J_main_to_sub",
                "sub": "sub_forward @ J_sub_to_main",
            },
            "shapes": {
                "latent_j_main_to_sub": [int(x) for x in j_main_to_sub.shape],
                "latent_j_sub_to_main": [int(x) for x in j_sub_to_main.shape],
                "main_variable_matrix": [int(x) for x in main_var_matrix.shape],
                "sub_variable_matrix": [int(x) for x in sub_var_matrix.shape],
            },
            "umap": {
                "main": main_umap_meta,
                "sub": sub_umap_meta,
                "neighbors": int(umap_neighbors),
                "min_dist": float(umap_min_dist),
                "seed": int(umap_seed),
            },
            "classifier_eval": classifier_eval,
            "outputs": {
                "latent_jacobian_main_to_sub_csv": str(latent_ms_csv),
                "latent_jacobian_sub_to_main_csv": str(latent_sm_csv),
                "main_variable_matrix_csv": str(main_mat_csv),
                "sub_variable_matrix_csv": str(sub_mat_csv),
                "main_variable_umap_csv": str(main_umap_csv),
                "main_variable_umap_png": str(main_umap_png),
                "sub_variable_umap_csv": str(sub_umap_csv),
                "sub_variable_umap_png": str(sub_umap_png),
            },
        }
        summary_json = out_dir / "regulation_umap_summary.json"
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        summary["summary_json"] = str(summary_json)
        return summary
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
