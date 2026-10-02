#!/usr/bin/env python3
"""Build the Figure 2 AnnData inputs from the generated simulation CSV."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


SIMULATION_DIR = Path(__file__).resolve().parents[2] / "datasets" / "simulation"
CSV_PATH = SIMULATION_DIR / "simulation_gene.csv"


def make_adata(df: pd.DataFrame, columns: list[str]) -> ad.AnnData:
    obs = pd.DataFrame(index=pd.RangeIndex(len(df)).astype(str))
    obs["samples"] = df["samples"].to_numpy(dtype=np.float64)
    return ad.AnnData(
        X=df[columns].to_numpy(dtype=np.float64),
        obs=obs,
    )


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    expected_columns = ["samples", "x1", "x2", "x3", "x4"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")

    adata_2d = make_adata(df, ["x1", "x2"])
    adata_3d = make_adata(df, ["x1", "x2", "x4"])

    adata_2d.write_h5ad(SIMULATION_DIR / "simulation4_2d.h5ad")
    adata_3d.write_h5ad(SIMULATION_DIR / "simulation4_3d.h5ad")

    x_merged = np.hstack(
        [
            np.asarray(adata_2d.X, dtype=np.float64),
            np.asarray(adata_3d.X, dtype=np.float64),
        ]
    )
    merged = ad.AnnData(X=x_merged, obs=adata_2d.obs.copy())
    merged.var_names = [
        "mod1_0",
        "mod1_1",
        "mod2_0",
        "mod2_1",
        "mod2_2",
    ]
    merged.var["feature_modality"] = ["mod1", "mod1", "mod2", "mod2", "mod2"]
    merged.write_h5ad(SIMULATION_DIR / "simulation4_merged.h5ad")

    counts = (
        df["samples"].value_counts().sort_index().astype(int).to_dict()
    )
    expected_counts = {0.0: 400, 0.75: 426, 1.5: 464, 2.25: 539, 3.0: 623}
    if counts != expected_counts:
        raise RuntimeError(f"Figure 2 cell counts changed: {counts}")

    print(f"CSV: {CSV_PATH}")
    print(f"2D: {adata_2d.shape}")
    print(f"3D: {adata_3d.shape}")
    print(f"Merged: {merged.shape}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
