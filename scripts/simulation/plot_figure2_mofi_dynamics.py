#!/usr/bin/env python3
"""Regenerate the MOFI dynamics and growth panels used by Figure 2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from CytoBridge.pl import plot as cb_pl  # noqa: E402
import CytoBridge.utils as cb_utils  # noqa: E402
from CytoBridge.utils.utils import set_seed  # noqa: E402


def true_growth_rate(b_values: np.ndarray) -> np.ndarray:
    b_values = np.asarray(b_values, dtype=np.float64)
    return 0.05 * (b_values**2 / (1.0 + b_values**2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adata",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "figure2_mofi"
            / "adata.h5ad"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures" / "figure2_reproduced",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(22)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(args.adata)
    model = cb_utils.load_model_from_adata(adata)
    model.to(args.device)

    for name in ["predicted_growth.png", "predicted_growth.pdf"]:
        cb_pl.plot_growth(
            adata,
            dim_reduction="none",
            output_path=str(args.output_dir / name),
        )

    cb_pl.plot_ode_trajectories_v2(
        adata=adata,
        model=model,
        output_path=str(args.output_dir),
        n_trajectories=25,
        n_bins=40,
        dim_reduction="none",
        device=args.device,
    )
    cb_pl.plot_landscape(
        adata,
        model,
        output_path=str(args.output_dir),
        dim_reduction="none",
        device=args.device,
    )
    if "velocity_latent" in adata.obsm:
        cb_pl.plot_velocity_stream(
            adata,
            model,
            str(args.output_dir),
            dim_reduction="none",
            device=args.device,
        )

    df = pd.read_csv(
        REPO_ROOT / "datasets" / "simulation" / "simulation_gene.csv"
    )
    x_csv = df[["x1", "x2"]].to_numpy(dtype=np.float64)
    x_adata = np.asarray(adata.X, dtype=np.float64)
    if x_csv.shape != x_adata.shape or np.max(np.abs(x_csv - x_adata)) > 1e-8:
        raise ValueError("CSV and AnnData rows are not aligned")
    adata_true = adata.copy()
    adata_true.obsm["growth_rate"] = true_growth_rate(
        df["x2"].to_numpy()
    )[:, None].astype(np.float32)
    for name in ["true_growth_from_generator.png", "true_growth_from_generator.pdf"]:
        cb_pl.plot_growth(
            adata_true,
            dim_reduction="none",
            output_path=str(args.output_dir / name),
        )
    print(f"Figure 2 dynamics outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
