#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.plotting.hc_author5_palette import AUTHOR5_CELLTYPE_ALIAS, AUTHOR5_COLORS, AUTHOR5_ORDER


TARGET_AUTHOR_TYPES = ["IN-CGE", "IN-MGE"]
TARGET_CELL_TYPES = TARGET_AUTHOR_TYPES + [AUTHOR5_CELLTYPE_ALIAS[x] for x in TARGET_AUTHOR_TYPES]


def _is_monotone(values: np.ndarray, *, tol: float = 1e-8) -> str:
    diff = np.diff(np.asarray(values, dtype=float))
    if diff.size == 0:
        return "flat"
    if np.all(diff >= -tol):
        return "nondecreasing"
    if np.all(diff <= tol):
        return "nonincreasing"
    return "mixed"


def _safe_slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text))
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "item"


def _resolve_default_inputs(run_dir: Path) -> tuple[Path, Path]:
    tables = run_dir / "tables"
    sweep = sorted(tables.glob("perturbation_zscore_sweep__*.csv"))
    focus = sorted(tables.glob("perturbation_timecurve_focus__*.csv"))
    if not sweep:
        raise FileNotFoundError(f"No perturbation_zscore_sweep__*.csv found under {tables}")
    if not focus:
        raise FileNotFoundError(f"No perturbation_timecurve_focus__*.csv found under {tables}")
    return sweep[0], focus[0]


def _build_summary_rows(sweep_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    use = sweep_df[sweep_df["cell_type"].astype(str).isin(TARGET_CELL_TYPES)].copy()
    if use.empty:
        return pd.DataFrame(
            columns=[
                "target_group",
                "target_genes",
                "IN-CGE_baseline_prob",
                "IN-CGE_slope",
                "IN-CGE_delta_slope",
                "IN-CGE_monotonicity",
                "IN-CGE_delta_monotonicity",
                "IN-MGE_baseline_prob",
                "IN-MGE_slope",
                "IN-MGE_delta_slope",
                "IN-MGE_monotonicity",
                "IN-MGE_delta_monotonicity",
                "competition_score",
                "direction_match_expected",
            ]
        )
    for group_name, group_df in use.groupby("target_group", sort=True):
        row: Dict[str, object] = {
            "target_group": str(group_name),
            "target_genes": "|".join(sorted(set(group_df["target_genes"].astype(str).tolist()))),
        }
        for author_label in TARGET_AUTHOR_TYPES:
            sub = group_df[group_df["cell_type"].astype(str).isin([author_label, AUTHOR5_CELLTYPE_ALIAS[author_label]])].sort_values("z_score")
            z = sub["z_score"].to_numpy(dtype=float)
            perturbed = sub["perturbed_prob"].to_numpy(dtype=float)
            delta = sub["delta_prob"].to_numpy(dtype=float)
            row[f"{author_label}_baseline_prob"] = float(sub["baseline_prob"].iloc[0]) if not sub.empty else float("nan")
            row[f"{author_label}_slope"] = float(np.polyfit(z, perturbed, 1)[0]) if len(sub) >= 2 else float("nan")
            row[f"{author_label}_delta_slope"] = float(np.polyfit(z, delta, 1)[0]) if len(sub) >= 2 else float("nan")
            row[f"{author_label}_monotonicity"] = _is_monotone(perturbed)
            row[f"{author_label}_delta_monotonicity"] = _is_monotone(delta)
            row[f"{author_label}_min_prob"] = float(np.min(perturbed)) if len(perturbed) else float("nan")
            row[f"{author_label}_max_prob"] = float(np.max(perturbed)) if len(perturbed) else float("nan")
        if pd.notna(row["IN-MGE_slope"]) and pd.notna(row["IN-CGE_slope"]):
            row["competition_score"] = float(row["IN-MGE_slope"]) - float(row["IN-CGE_slope"])
            row["direction_match_expected"] = bool(float(row["IN-MGE_slope"]) > 0.0 and float(row["IN-CGE_slope"]) < 0.0)
        else:
            row["competition_score"] = float("nan")
            row["direction_match_expected"] = False
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["direction_match_expected", "competition_score"], ascending=[False, False])


def _plot_final_sweep(sweep_df: pd.DataFrame, output_path: Path) -> None:
    use = sweep_df[sweep_df["cell_type"].astype(str).isin(TARGET_CELL_TYPES)].copy()
    groups = use["target_group"].astype(str).drop_duplicates().tolist()
    fig, axes = plt.subplots(len(groups), 1, figsize=(10.5, max(4.4, 3.0 * len(groups))), sharex=True, squeeze=False)
    for idx, group_name in enumerate(groups):
        ax = axes[idx][0]
        group_df = use[use["target_group"].astype(str) == str(group_name)].copy()
        for author_label in TARGET_AUTHOR_TYPES:
            sub = group_df[group_df["cell_type"].astype(str).isin([author_label, AUTHOR5_CELLTYPE_ALIAS[author_label]])].sort_values("z_score")
            if sub.empty:
                continue
            ax.plot(
                sub["z_score"].to_numpy(dtype=float),
                sub["perturbed_prob"].to_numpy(dtype=float),
                marker="o",
                linewidth=2.0,
                color=AUTHOR5_COLORS[author_label],
                label=author_label,
            )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("final prob")
        ax.set_title(str(group_name))
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[-1][0].set_xlabel("perturbation z score")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_timecourse_focus(focus_df: pd.DataFrame, output_dir: Path) -> List[str]:
    outputs: List[str] = []
    use = focus_df[focus_df["cell_type"].astype(str).isin(TARGET_CELL_TYPES)].copy()
    for group_name, group_df in use.groupby("target_group", sort=True):
        z_scores = sorted({float(x) for x in group_df["z_score"].tolist()})
        fig, axes = plt.subplots(1, len(z_scores), figsize=(4.2 * len(z_scores), 4.0), sharey=True, squeeze=False)
        for col, z in enumerate(z_scores):
            ax = axes[0][col]
            sub_z = group_df[np.isclose(group_df["z_score"].to_numpy(dtype=float), float(z))].copy()
            for author_label in TARGET_AUTHOR_TYPES:
                sub = sub_z[sub_z["cell_type"].astype(str).isin([author_label, AUTHOR5_CELLTYPE_ALIAS[author_label]])].sort_values("time")
                if sub.empty:
                    continue
                ax.plot(
                    sub["time"].to_numpy(dtype=float),
                    sub["perturbed_prob"].to_numpy(dtype=float),
                    marker="o",
                    linewidth=2.0,
                    color=AUTHOR5_COLORS[author_label],
                    label=author_label,
                )
            ax.set_title(f"z={z:g}")
            ax.set_xlabel("time")
            ax.grid(alpha=0.25, linewidth=0.6)
        axes[0][0].set_ylabel("probability")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=False)
        fig.tight_layout()
        out = output_dir / f"focus_timecourse__{_safe_slug(str(group_name))}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=320, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=320, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(out))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize HC author5 perturbation competition between IN-CGE and IN-MGE.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sweep-csv", type=Path, default=None)
    parser.add_argument("--focus-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.sweep_csv is None or args.focus_csv is None:
        default_sweep, default_focus = _resolve_default_inputs(args.run_dir)
        sweep_csv = args.sweep_csv or default_sweep
        focus_csv = args.focus_csv or default_focus
    else:
        sweep_csv = args.sweep_csv
        focus_csv = args.focus_csv

    output_dir = args.output_dir or (args.run_dir / "figures" / "author5_perturbation_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_df = pd.read_csv(sweep_csv)
    focus_df = pd.read_csv(focus_csv)
    summary_df = _build_summary_rows(sweep_df)
    summary_csv = output_dir / "author5_terminal_competition_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    sweep_plot = output_dir / "author5_terminal_competition_sweep.pdf"
    _plot_final_sweep(sweep_df, sweep_plot)
    timecourse_outputs = _plot_timecourse_focus(focus_df, output_dir)

    manifest = {
        "run_dir": str(args.run_dir),
        "sweep_csv": str(sweep_csv),
        "focus_csv": str(focus_csv),
        "summary_csv": str(summary_csv),
        "sweep_plot_pdf": str(sweep_plot),
        "timecourse_plots": timecourse_outputs,
        "target_author_cell_types": TARGET_AUTHOR_TYPES,
        "target_cell_types": TARGET_CELL_TYPES,
    }
    manifest_path = output_dir / "author5_perturbation_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
