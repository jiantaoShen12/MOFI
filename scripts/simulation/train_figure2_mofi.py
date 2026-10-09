#!/usr/bin/env python3
"""Reproduce the MOFI dynamics model used in manuscript Figure 2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc
import yaml
from anndata import AnnData


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import CytoBridge.pp as cb_pp  # noqa: E402
from CytoBridge.tl.fit import fit as fit_model  # noqa: E402
from CytoBridge.utils.utils import set_seed  # noqa: E402


DATA_PATH = REPO_ROOT / "datasets" / "simulation" / "simulation4_2d.h5ad"
SIMULATION_CSV = REPO_ROOT / "datasets" / "simulation" / "simulation_gene.csv"
CONFIG_PATH = REPO_ROOT / "configs" / "training" / "simulation_figure2.yaml"
TIME_MAPPING = {
    0.0: 0.0,
    0.75: 1.0,
    1.5: 2.0,
    2.25: 3.0,
    3.0: 4.0,
}
SEED = 22


def preprocess(adata):
    return cb_pp.preprocess(
        adata,
        time_key="samples",
        time_mapping=TIME_MAPPING,
        dim_reduction="none",
        normalization=False,
        log1p=False,
        select_hvg=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figure2_mofi",
    )
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Diagnostic-only override. Omit it for the paper configuration.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    adata = preprocess(sc.read_h5ad(DATA_PATH))

    # Preserve the exact preparation sequence used for experiment_simu_05.
    simulation_df = pd.read_csv(SIMULATION_CSV)
    adata_t = AnnData(
        simulation_df[["x1", "x2", "x4"]].to_numpy(dtype="float32")
    )
    adata_t.obs["samples"] = simulation_df["samples"].to_numpy()
    _ = preprocess(adata_t)

    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config["ckpt_dir"] = str(args.output_dir)

    paper_parameters = args.epochs_override is None
    if args.epochs_override is not None:
        if args.epochs_override <= 0:
            raise ValueError("--epochs-override must be positive")
        for stage in config["training"]["plan"]:
            stage["epochs"] = int(args.epochs_override)

    print(f"Figure 2 paper parameters: {paper_parameters}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(json.dumps(config, indent=2))

    adata = fit_model(
        adata,
        config=config,
        batch_size=400,
        device=args.device,
    )
    output_adata = args.output_dir / "adata.h5ad"
    adata.write_h5ad(output_adata)
    with (args.output_dir / "reproduction_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "paper_parameters": paper_parameters,
                "seed": SEED,
                "data_path": str(DATA_PATH),
                "simulation_csv": str(SIMULATION_CSV),
                "config_path": str(CONFIG_PATH),
                "output_adata": str(output_adata),
                "device": args.device,
                "time_mapping": TIME_MAPPING,
                "resolved_config": config,
            },
            stream,
            indent=2,
        )
    print(f"Saved reproducible MOFI AnnData: {output_adata}")


if __name__ == "__main__":
    main()
