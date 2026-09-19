#!/usr/bin/env python3
"""Compare the joint HSPC candidate with the immediately preceding active run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / ".vendor", ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from CytoBridge.reproducibility import run_real_downstream  # noqa: E402
from audit_hspc_finetuned_downstream import (  # noqa: E402
    aligned_correlation,
    latent_mean_rmse,
    observed_time_rmse,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", type=Path)
    args = parser.parse_args()
    baseline_run = ROOT / "results" / "figure_4_velocity_tmap_base_audit" / "coarse_dense" / "sample"
    if args.candidate_run is None:
        fine_output = ROOT / "results" / "figure_4_velocity_tmap_candidate_audit"
        result = run_real_downstream(
            "hspc_31800",
            output_dir=fine_output,
            selected_figures=[
                "perturbation_zscore_sweep",
                "perturbation_sweep_heatmap_prob",
                "program_both",
                "key_molecule_volcano",
            ],
            device="cuda",
            sample_max_cells=None,
            reuse_cache=False,
            repo_root=ROOT,
        )
        fine_run = Path(result["run_dir"])
    else:
        fine_run = args.candidate_run.resolve()
    base_tables = baseline_run / "tables"
    fine_tables = fine_run / "tables"
    base_secondary = observed_time_rmse(
        base_tables / "sec_var_marker_pred.csv", base_tables / "sec_var_marker_real.csv",
        key="marker", value="mean_expr",
    )
    fine_secondary = observed_time_rmse(
        fine_tables / "sec_var_marker_pred.csv", fine_tables / "sec_var_marker_real.csv",
        key="marker", value="mean_expr",
    )
    base_fate = observed_time_rmse(
        base_tables / "cell_type_pred_main.csv", base_tables / "cell_type_real_main.csv",
        key="cell_type", value="prob",
    )
    fine_fate = observed_time_rmse(
        fine_tables / "cell_type_pred_main.csv", fine_tables / "cell_type_real_main.csv",
        key="cell_type", value="prob",
    )
    base_sweep = next(base_tables.glob("perturbation_zscore_sweep__*.csv"))
    fine_sweep = next(fine_tables.glob("perturbation_zscore_sweep__*.csv"))
    sweep_comparison = aligned_correlation(
        base_sweep, fine_sweep,
        keys=["target_group", "z_score", "cell_type", "space", "sweep_space"],
        value="delta_rate",
    )
    base_traj = np.load(baseline_run / "assets" / "traj_main.npy", allow_pickle=False)
    fine_traj = np.load(fine_run / "assets" / "traj_main.npy", allow_pickle=False)
    primary_max_abs = float(np.max(np.abs(base_traj - fine_traj)))
    base_latent_rmse = latent_mean_rmse(
        baseline_run, ROOT / "paper_data" / "processed" / "hspc_31800" / "protein.h5ad",
        time_key="time_point_processed",
    )
    fine_latent_rmse = latent_mean_rmse(
        fine_run, ROOT / "paper_data" / "processed" / "hspc_31800" / "protein.h5ad",
        time_key="time_point_processed",
    )
    base_by_marker = {row["marker"]: row["rmse"] for row in base_secondary["rows"]}
    fine_by_marker = {row["marker"]: row["rmse"] for row in fine_secondary["rows"]}
    marker_rows = [
        {
            "marker": marker,
            "baseline_rmse": float(base_by_marker[marker]),
            "candidate_rmse": float(fine_by_marker[marker]),
            "relative_change": float(fine_by_marker[marker] / max(base_by_marker[marker], 1e-8) - 1.0),
        }
        for marker in sorted(base_by_marker)
    ]
    non_target = [row for row in marker_rows if row["marker"] != "CD41"]
    non_target_max = max(row["relative_change"] for row in non_target)
    latent_change = float(fine_latent_rmse / max(base_latent_rmse, 1e-8) - 1.0)
    fate_change = float(fine_fate["macro_rmse"] / max(base_fate["macro_rmse"], 1e-8) - 1.0)
    accepted = bool(
        non_target_max <= 0.10
        and latent_change <= 0.05
        and fate_change <= 0.05
        and sweep_comparison["pearson"] >= 0.98
    )
    audit = {
        "status": "PASS" if accepted else "REVIEW",
        "baseline_run": str(baseline_run.relative_to(ROOT)),
        "candidate_run": str(fine_run.relative_to(ROOT)),
        "primary_trajectory_max_abs_difference": primary_max_abs,
        "secondary_latent_mean_rmse": {
            "baseline": base_latent_rmse,
            "candidate": fine_latent_rmse,
            "relative_change": latent_change,
        },
        "secondary_marker_rmse": marker_rows,
        "non_target_marker_max_degradation_fraction": float(non_target_max),
        "cell_fate_macro_rmse": {
            "baseline": base_fate["macro_rmse"],
            "candidate": fine_fate["macro_rmse"],
            "relative_change": fate_change,
        },
        "perturbation_delta_rate": sweep_comparison,
        "acceptance_thresholds": {
            "non_target_marker_max_degradation_fraction": 0.10,
            "secondary_latent_mean_rmse_max_degradation_fraction": 0.05,
            "cell_fate_macro_rmse_max_degradation_fraction": 0.05,
            "perturbation_delta_rate_min_pearson": 0.98,
        },
        "accepted_for_full_notebook_display": accepted,
    }
    output = ROOT / "validation" / "hspc_velocity_tmap_downstream_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
