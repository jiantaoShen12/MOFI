from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .auto_pairs import build_pair_tables
from .context import TFRunContext
from .grn import build_grn_focus_from_pairs, export_grn_tables
from .heatmap import build_relation_heatmap
from .pairs import build_pair_timeseries
from .perturb import extract_perturbation_indices, plot_perturbation_trajectories
from .types import TFPipelineResult, TFStepResult, resolve_tf_config


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_pairs_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["primary_var", "secondary_var", "score", "source"])
    return pd.read_csv(p)


def _run_step(
    result: TFPipelineResult,
    *,
    step_name: str,
    fn,
    failure_policy: str,
) -> Optional[Dict[str, Any]]:
    try:
        artifacts = fn()
        if not isinstance(artifacts, dict):
            artifacts = {"result": artifacts}
        result.steps.append(TFStepResult(step=step_name, status="success", artifacts=artifacts))
        return artifacts
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        result.steps.append(TFStepResult(step=step_name, status="failed", reason=reason, artifacts={}))
        if str(failure_policy).strip().lower() != "best_effort":
            raise
        return None


def run_tf_pipeline_for_run(run_dir: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg_obj = resolve_tf_config(cfg)
    ctx = TFRunContext.from_run_dir(
        run_dir,
        output_table_dir=cfg_obj.output_table_dir,
        output_figure_dir=cfg_obj.output_figure_dir,
    )

    result = TFPipelineResult(
        run_dir=str(ctx.run_dir),
        output_table_dir=str(ctx.output_table_dir),
        output_figure_dir=str(ctx.output_figure_dir),
    )

    time_points = ctx.resolve_time_points(cfg_obj.time_points)

    state: Dict[str, Any] = {
        "pairs_df": pd.DataFrame(columns=["primary_var", "secondary_var", "score", "source"]),
        "pair_summary": {},
        "perturb_indices_csv": "",
    }

    enabled = {str(x).strip().lower() for x in cfg_obj.steps}

    if "auto_pairs" in enabled:
        artifacts = _run_step(
            result,
            step_name="auto_pairs",
            failure_policy=cfg_obj.failure_policy,
            fn=lambda: build_pair_tables(
                ctx,
                top_k_pairs=int(cfg_obj.top_k_pairs),
                manual_pairs_txt=cfg_obj.manual_pairs_txt,
            ),
        )
        if isinstance(artifacts, dict):
            state["pair_summary"] = artifacts
            state["pairs_df"] = _load_pairs_df(str(artifacts.get("final_pairs_csv", "")))

    if "grn_export" in enabled:
        _run_step(
            result,
            step_name="grn_export",
            failure_policy=cfg_obj.failure_policy,
            fn=lambda: export_grn_tables(
                ctx,
                time_points=time_points,
                device=str(cfg_obj.grn_device),
                chunk_size=int(cfg_obj.grn_chunk_size),
                mapper_batch_size=int(cfg_obj.grn_mapper_batch_size),
                include_hessian_term=bool(cfg_obj.grn_include_hessian_term),
            ),
        )

    if "grn_focus" in enabled:
        _run_step(
            result,
            step_name="grn_focus",
            failure_policy=cfg_obj.failure_policy,
            fn=lambda: build_grn_focus_from_pairs(
                ctx,
                pairs_df=state["pairs_df"],
                time_points=time_points,
                focus_top_n=int(cfg_obj.focus_top_n),
            ),
        )

    if "perturb_select" in enabled:
        artifacts = _run_step(
            result,
            step_name="perturb_select",
            failure_policy=cfg_obj.failure_policy,
            fn=lambda: extract_perturbation_indices(
                ctx,
                target_group=str(cfg_obj.perturb_target_group),
                z_score=str(cfg_obj.perturb_z_score),
                time_point=str(cfg_obj.perturb_time),
                cell_type=str(cfg_obj.perturb_cell_type),
                label_rule="argmax",
                device=str(cfg_obj.grn_device),
                output_csv="",
            ),
        )
        if isinstance(artifacts, dict):
            state["perturb_indices_csv"] = str(artifacts.get("output_csv", ""))

    if "perturb_traj" in enabled:
        _run_step(
            result,
            step_name="perturb_traj",
            failure_policy=cfg_obj.failure_policy,
            fn=lambda: plot_perturbation_trajectories(
                ctx,
                indices_csv=str(state["perturb_indices_csv"]),
                scores=(cfg_obj.perturb_scores if cfg_obj.perturb_scores else None),
                sample_n=int(cfg_obj.perturb_sample_n),
                seed=int(cfg_obj.perturb_seed),
                device=str(cfg_obj.grn_device),
                output_dir="",
            ),
        )

    if "pair_timeseries" in enabled:
        _run_step(
            result,
            step_name="pair_timeseries",
            failure_policy=cfg_obj.failure_policy,
            fn=lambda: build_pair_timeseries(
                ctx,
                pairs_df=state["pairs_df"],
                time_points=time_points,
                max_lag_steps=int(cfg_obj.pair_max_lag_steps),
                line_dt=float(cfg_obj.pair_line_dt),
                key_time_step=float(cfg_obj.pair_key_time_step),
                plot_top_n=int(cfg_obj.pair_plot_top_n),
            ),
        )

    if "relation_heatmap" in enabled:
        direction = str(cfg_obj.relation_heatmap_direction).strip().lower()
        prefix = "grn_sub_main_all_relations_tf" if direction in {"secondary_to_primary", "sub_to_main", "2to1", "reverse"} else "grn_main_sub_all_relations_tf"
        _run_step(
            result,
            step_name="relation_heatmap",
            failure_policy=cfg_obj.failure_policy,
            fn=lambda: build_relation_heatmap(
                ctx,
                pairs_df=state["pairs_df"],
                time_points=time_points,
                prefix=prefix,
                norm=str(cfg_obj.relation_heatmap_norm),
                row_cluster=bool(cfg_obj.relation_heatmap_row_cluster),
                cluster_method=str(cfg_obj.relation_heatmap_cluster_method),
                cluster_metric=str(cfg_obj.relation_heatmap_cluster_metric),
                nan_strategy=str(cfg_obj.relation_heatmap_nan_strategy),
                export_reordered_matrix=bool(cfg_obj.relation_heatmap_export_reordered_matrix),
                time_dense_keep_real_threshold=int(cfg_obj.relation_heatmap_time_dense_keep_real_threshold),
                time_dense_points_between=int(cfg_obj.relation_heatmap_time_dense_points_between),
                time_dense_target_points=int(cfg_obj.relation_heatmap_time_dense_target_points),
                direction=str(cfg_obj.relation_heatmap_direction),
            ),
        )

    payload = result.to_dict()
    payload["config"] = {
        "failure_policy": cfg_obj.failure_policy,
        "pair_source": cfg_obj.pair_source,
        "manual_pairs_txt": list(cfg_obj.manual_pairs_txt),
        "steps": list(cfg_obj.steps),
        "time_points_used": [float(x) for x in time_points],
        "relation_heatmap": {
            "norm": str(cfg_obj.relation_heatmap_norm),
            "direction": str(cfg_obj.relation_heatmap_direction),
            "row_cluster": bool(cfg_obj.relation_heatmap_row_cluster),
            "cluster_method": str(cfg_obj.relation_heatmap_cluster_method),
            "cluster_metric": str(cfg_obj.relation_heatmap_cluster_metric),
            "nan_strategy": str(cfg_obj.relation_heatmap_nan_strategy),
            "export_reordered_matrix": bool(cfg_obj.relation_heatmap_export_reordered_matrix),
            "time_dense_keep_real_threshold": int(cfg_obj.relation_heatmap_time_dense_keep_real_threshold),
            "time_dense_points_between": int(cfg_obj.relation_heatmap_time_dense_points_between),
            "time_dense_target_points": int(cfg_obj.relation_heatmap_time_dense_target_points),
        },
    }
    payload["context"] = ctx.to_dict()

    manifest_path = ctx.output_figure_dir / "tf_pipeline_manifest.json"
    _write_json(manifest_path, payload)
    payload["manifest_json"] = str(manifest_path)

    ctx.close_backed()
    return payload


def run_tf_pipeline_for_summary(workflow_summary_path: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path = Path(workflow_summary_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    runs = summary.get("runs", []) if isinstance(summary, dict) else []

    out_rows: List[Dict[str, Any]] = []
    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        run_dir = str(run.get("run_dir", "")).strip()
        if not run_dir:
            continue
        try:
            res = run_tf_pipeline_for_run(run_dir, cfg=cfg)
            out_rows.append({"run_index": i, "run_dir": run_dir, "status": "success", "manifest_json": res.get("manifest_json", "")})
        except Exception as exc:
            out_rows.append({"run_index": i, "run_dir": run_dir, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})

    payload = {
        "workflow_summary_path": str(path),
        "num_runs": int(len(out_rows)),
        "runs": out_rows,
    }
    out_path = path.with_name("tf_collection_summary.json")
    _write_json(out_path, payload)
    payload["collection_json"] = str(out_path)
    return payload
