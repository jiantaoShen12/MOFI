from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_STEPS: List[str] = [
    "auto_pairs",
    "grn_export",
    "grn_focus",
    "perturb_select",
    "perturb_traj",
    "pair_timeseries",
    "relation_heatmap",
]


@dataclass
class TFPipelineConfig:
    failure_policy: str = "best_effort"
    pair_source: str = "auto"
    manual_pairs_txt: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=lambda: list(DEFAULT_STEPS))
    time_points: Optional[List[float]] = None

    output_table_dir: str = ""
    output_figure_dir: str = ""

    top_k_pairs: int = 50
    focus_top_n: int = 8

    grn_device: str = "cpu"
    grn_chunk_size: int = 256
    grn_mapper_batch_size: int = 256
    grn_include_hessian_term: bool = True

    pair_max_lag_steps: int = 2
    pair_line_dt: float = 0.25
    pair_key_time_step: float = 1.0
    pair_plot_top_n: int = 12
    relation_heatmap_norm: str = "subset"
    relation_heatmap_row_cluster: bool = True
    relation_heatmap_cluster_method: str = "average"
    relation_heatmap_cluster_metric: str = "euclidean"
    relation_heatmap_nan_strategy: str = "row_mean"
    relation_heatmap_export_reordered_matrix: bool = True
    relation_heatmap_direction: str = "primary_to_secondary"
    relation_heatmap_time_dense_keep_real_threshold: int = 10
    relation_heatmap_time_dense_points_between: int = 0
    relation_heatmap_time_dense_target_points: int = 0

    perturb_sample_n: int = 30
    perturb_seed: int = 42
    perturb_scores: List[float] = field(default_factory=list)

    perturb_target_group: str = "auto"
    perturb_z_score: str = "auto"
    perturb_time: str = "auto"
    perturb_cell_type: str = "auto"


@dataclass
class TFStepResult:
    step: str
    status: str
    reason: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TFPipelineResult:
    run_dir: str
    steps: List[TFStepResult] = field(default_factory=list)
    output_table_dir: str = ""
    output_figure_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "output_table_dir": self.output_table_dir,
            "output_figure_dir": self.output_figure_dir,
            "steps": [
                {
                    "step": s.step,
                    "status": s.status,
                    "reason": s.reason,
                    "artifacts": s.artifacts,
                }
                for s in self.steps
            ],
        }


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def resolve_tf_config(raw: Optional[Dict[str, Any]] = None) -> TFPipelineConfig:
    cfg = TFPipelineConfig()
    if not isinstance(raw, dict):
        return cfg

    if "failure_policy" in raw:
        cfg.failure_policy = str(raw["failure_policy"]).strip() or "best_effort"
    if "pair_source" in raw:
        cfg.pair_source = str(raw["pair_source"]).strip() or "auto"
    if "manual_pairs_txt" in raw:
        cfg.manual_pairs_txt = [str(x) for x in _as_list(raw.get("manual_pairs_txt")) if str(x).strip()]
    if "steps" in raw:
        parsed = [str(x).strip() for x in _as_list(raw.get("steps")) if str(x).strip()]
        if parsed:
            cfg.steps = parsed
    if "time_points" in raw and raw.get("time_points") is not None:
        cfg.time_points = [float(x) for x in _as_list(raw.get("time_points"))]

    for key in [
        "output_table_dir",
        "output_figure_dir",
        "grn_device",
        "relation_heatmap_norm",
        "relation_heatmap_direction",
        "relation_heatmap_cluster_method",
        "relation_heatmap_cluster_metric",
        "relation_heatmap_nan_strategy",
    ]:
        if key in raw and raw.get(key) is not None:
            setattr(cfg, key, str(raw.get(key)))

    int_keys = {
        "top_k_pairs",
        "focus_top_n",
        "grn_chunk_size",
        "grn_mapper_batch_size",
        "pair_max_lag_steps",
        "pair_plot_top_n",
        "perturb_sample_n",
        "perturb_seed",
        "relation_heatmap_time_dense_keep_real_threshold",
        "relation_heatmap_time_dense_points_between",
        "relation_heatmap_time_dense_target_points",
    }
    for key in int_keys:
        if key in raw and raw.get(key) is not None:
            setattr(cfg, key, int(raw.get(key)))

    float_keys = {"pair_line_dt", "pair_key_time_step"}
    for key in float_keys:
        if key in raw and raw.get(key) is not None:
            setattr(cfg, key, float(raw.get(key)))

    if "grn_include_hessian_term" in raw:
        cfg.grn_include_hessian_term = bool(raw.get("grn_include_hessian_term"))
    if "relation_heatmap_row_cluster" in raw:
        cfg.relation_heatmap_row_cluster = bool(raw.get("relation_heatmap_row_cluster"))
    if "relation_heatmap_export_reordered_matrix" in raw:
        cfg.relation_heatmap_export_reordered_matrix = bool(raw.get("relation_heatmap_export_reordered_matrix"))

    if "perturb_scores" in raw and raw.get("perturb_scores") is not None:
        cfg.perturb_scores = sorted({float(x) for x in _as_list(raw.get("perturb_scores"))})

    for key in ["perturb_target_group", "perturb_z_score", "perturb_time", "perturb_cell_type"]:
        if key in raw and raw.get(key) is not None:
            setattr(cfg, key, str(raw.get(key)))

    return cfg
