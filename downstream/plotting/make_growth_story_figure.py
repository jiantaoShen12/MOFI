#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a growth-aware fate story figure from dense-time outputs.")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--pred-table", type=Path, required=True, help="cell_type_pred_main.csv")
    p.add_argument("--analysis-ready", type=Path, required=True, help="analysis_ready main h5ad with observed cell_type/time")
    p.add_argument("--output-figure", type=Path, required=True)
    p.add_argument("--output-summary", type=Path, required=True)
    p.add_argument("--time-key", default="time_point_processed")
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--space", default="joint")
    return p.parse_args()


def _load_pred_table(path: Path, space: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(path)
    if "source" in df.columns:
        df = df.loc[df["source"].astype(str) == "pred"].copy()
    if "space" in df.columns:
        keep = df["space"].astype(str) == str(space)
        if keep.any():
            df = df.loc[keep].copy()
    prob = df.pivot_table(index="time", columns="cell_type", values="prob", aggfunc="mean").sort_index()
    weight_sum = df[["time", "weight_sum"]].drop_duplicates().set_index("time")["weight_sum"].sort_index()
    occupancy = prob.mul(weight_sum, axis=0)
    return prob, weight_sum, occupancy


def _load_observed_counts(path: Path, time_key: str, cell_type_key: str) -> pd.DataFrame:
    adata = ad.read_h5ad(path)
    obs = adata.obs[[time_key, cell_type_key]].copy()
    obs[time_key] = pd.to_numeric(obs[time_key], errors="coerce")
    obs[cell_type_key] = obs[cell_type_key].astype(str)
    counts = (
        obs.groupby([time_key, cell_type_key], observed=False)
        .size()
        .rename("observed_count")
        .reset_index()
    )
    return counts


def _ordered_cell_types(prob: pd.DataFrame) -> list[str]:
    delta = (prob.iloc[-1] - prob.iloc[0]).sort_values(ascending=False)
    return delta.index.astype(str).tolist()


def main() -> None:
    args = parse_args()
    prob, weight_sum, occupancy = _load_pred_table(args.pred_table, args.space)
    observed = _load_observed_counts(args.analysis_ready, args.time_key, args.cell_type_key)
    observed_total = observed.groupby(args.time_key, observed=False)["observed_count"].sum().sort_index()
    observed_pivot = (
        observed.pivot_table(index=args.time_key, columns=args.cell_type_key, values="observed_count", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )

    cell_types = _ordered_cell_types(prob)
    colors = {ct: plt.get_cmap("tab10")(i % 10) for i, ct in enumerate(cell_types)}

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True, constrained_layout=True)

    axes[0].plot(weight_sum.index, weight_sum.values, color="black", lw=2.0, label="Predicted total mass")
    axes[0].scatter(observed_total.index, observed_total.values, color="black", s=34, zorder=3, label="Observed total cells")
    axes[0].set_ylabel("Total mass / cells")
    axes[0].set_title(f"{args.dataset_name}: growth-aware dense-time occupancy")
    axes[0].legend(frameon=False, loc="best")

    for ct in cell_types:
        if ct not in occupancy.columns:
            continue
        axes[1].plot(occupancy.index, occupancy[ct].values, color=colors[ct], lw=2.2, label=f"{ct} predicted")
        if ct in observed_pivot.columns:
            axes[1].scatter(
                observed_pivot.index,
                observed_pivot[ct].values,
                color=colors[ct],
                s=30,
                zorder=3,
            )

    axes[1].set_xlabel("Dense time")
    axes[1].set_ylabel("Weighted occupancy / observed cells")
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=250, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset_name": args.dataset_name,
        "pred_table": str(args.pred_table),
        "analysis_ready": str(args.analysis_ready),
        "weight_sum_start": float(weight_sum.iloc[0]),
        "weight_sum_end": float(weight_sum.iloc[-1]),
        "weight_sum_delta": float(weight_sum.iloc[-1] - weight_sum.iloc[0]),
        "cell_type_start_prob": {str(k): float(v) for k, v in prob.iloc[0].to_dict().items()},
        "cell_type_end_prob": {str(k): float(v) for k, v in prob.iloc[-1].to_dict().items()},
        "cell_type_start_occupancy": {str(k): float(v) for k, v in occupancy.iloc[0].to_dict().items()},
        "cell_type_end_occupancy": {str(k): float(v) for k, v in occupancy.iloc[-1].to_dict().items()},
        "cell_type_occupancy_delta": {str(k): float(v) for k, v in (occupancy.iloc[-1] - occupancy.iloc[0]).to_dict().items()},
        "observed_total_by_time": {str(k): int(v) for k, v in observed_total.to_dict().items()},
        "observed_cell_type_by_time": {
            str(time): {str(ct): int(val) for ct, val in row.dropna().to_dict().items()}
            for time, row in observed_pivot.iterrows()
        },
        "cell_type_order": cell_types,
    }
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(args.output_figure), "summary": str(args.output_summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
