#!/usr/bin/env python3
"""Build the checksum manifest for local paper-data release assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


DATASETS = {
    "gse213152": {
        "accession": "GSE213152",
        "files": {
            "processed_rna": "paper_data/processed/gse213152/rna.h5ad",
            "processed_atac": "paper_data/processed/gse213152/atac.h5ad",
            "dynamics": "paper_data/dynamics/gse213152/adata.h5ad",
            "training_config": "paper_data/dynamics/gse213152/config.yaml",
            "training_map": "paper_data/maps/gse213152_training/best_model.pt",
            "analysis_map": "paper_data/maps/gse213152_analysis/best_model.pt",
            "paper_umap_primary": "paper_data/umap/gse213152/primary.pkl",
            "paper_umap_secondary": "paper_data/umap/gse213152/secondary.pkl",
            "paper_umap_parameters": "paper_data/umap/gse213152/parameters.json",
            "figure3_secondary_background": "paper_data/projected/gse213152_figure3/secondary_background.npy",
            "figure3_secondary_trajectories": "paper_data/projected/gse213152_figure3/secondary_trajectories.npy",
            "figure3_secondary_points": "paper_data/projected/gse213152_figure3/secondary_points.npy",
        },
    },
    "hspc_31800": {
        "accession": "GSE305370 donor 31800",
        "files": {
            "processed_rna": "paper_data/processed/hspc_31800/rna.h5ad",
            "processed_protein": "paper_data/processed/hspc_31800/protein.h5ad",
            "dynamics": "paper_data/dynamics/hspc_31800/adata.h5ad",
            "paper_dynamics_original_velocity": "paper_data/dynamics/hspc_31800/adata_original_velocity.h5ad",
            "velocity_finetuned_dynamics": "paper_data/dynamics/hspc_31800/adata_velocity_finetuned.h5ad",
            "velocity_tmap_base_dynamics": "paper_data/dynamics/hspc_31800/adata_velocity_tmap_base.h5ad",
            "velocity_tmap_candidate_dynamics": "paper_data/dynamics/hspc_31800/adata_velocity_tmap_finetuned.h5ad",
            "training_config": "paper_data/dynamics/hspc_31800/config.yaml",
            "map": "paper_data/maps/hspc_31800/best_model.pt",
            "paper_map_original": "paper_data/maps/hspc_31800/best_model_original.pt",
            "trajectory_finetuned_map": "paper_data/maps/hspc_31800/best_model_trajectory_finetuned.pt",
            "velocity_tmap_base_map": "paper_data/maps/hspc_31800/best_model_velocity_tmap_base.pt",
            "velocity_tmap_candidate_map": "paper_data/maps/hspc_31800/best_model_velocity_tmap_finetuned.pt",
            "perturb_umap_primary": "paper_data/umap/hspc_31800/perturb_primary.pkl",
            "perturb_umap_secondary": "paper_data/umap/hspc_31800/perturb_secondary.pkl",
            "tfrc_cd71_real_curve": "paper_data/archived/hspc_figure4/tfrc_cd71/real_curve.csv",
            "tfrc_cd71_z_m40": "paper_data/archived/hspc_figure4/tfrc_cd71/sim_curve__z-m40.csv",
            "tfrc_cd71_z_m20": "paper_data/archived/hspc_figure4/tfrc_cd71/sim_curve__z-m20.csv",
            "tfrc_cd71_z_0": "paper_data/archived/hspc_figure4/tfrc_cd71/sim_curve__z-0.csv",
            "tfrc_cd71_z_20": "paper_data/archived/hspc_figure4/tfrc_cd71/sim_curve__z-20.csv",
            "tfrc_cd71_z_40": "paper_data/archived/hspc_figure4/tfrc_cd71/sim_curve__z-40.csv",
            "itga2b_cd41_real_curve": "paper_data/archived/hspc_figure4/itga2b_cd41/real_curve.csv",
            "itga2b_cd41_z_m10": "paper_data/archived/hspc_figure4/itga2b_cd41/sim_curve__z-m10.csv",
            "itga2b_cd41_z_m5": "paper_data/archived/hspc_figure4/itga2b_cd41/sim_curve__z-m5.csv",
            "itga2b_cd41_z_0": "paper_data/archived/hspc_figure4/itga2b_cd41/sim_curve__z-0.csv",
            "itga2b_cd41_z_5": "paper_data/archived/hspc_figure4/itga2b_cd41/sim_curve__z-5.csv",
            "itga2b_cd41_z_10": "paper_data/archived/hspc_figure4/itga2b_cd41/sim_curve__z-10.csv",
        },
    },
    "ho": {
        "accession": "E-MTAB-12001; E-MTAB-11998",
        "label_note": "Release copies correct early nepi->nect and late nect->nepi.",
        "files": {
            "processed_rna": "paper_data/processed/ho/rna.h5ad",
            "processed_atac": "paper_data/processed/ho/atac.h5ad",
            "dynamics": "paper_data/dynamics/ho/adata.h5ad",
            "training_config": "paper_data/dynamics/ho/config.yaml",
            "map": "paper_data/maps/ho/best_model.pt",
            "paper_umap_primary": "paper_data/umap/ho/primary.pkl",
            "paper_umap_secondary": "paper_data/umap/ho/secondary.pkl",
            "figure5_primary_background": "paper_data/projected/ho_figure5/primary_background.npy",
            "figure5_primary_trajectories": "paper_data/projected/ho_figure5/primary_trajectories.npy",
            "figure5_primary_points": "paper_data/projected/ho_figure5/primary_points.npy",
            "figure5_primary_trajectories_raw": "paper_data/projected/ho_figure5/primary_trajectories_raw.npy",
            "figure5_primary_points_raw": "paper_data/projected/ho_figure5/primary_points_raw.npy",
            "figure5_primary_trajectories_original": "paper_data/projected/ho_figure5/primary_trajectories_original.npy",
            "figure5_primary_points_original": "paper_data/projected/ho_figure5/primary_points_original.npy",
            "figure5_secondary_background": "paper_data/projected/ho_figure5/secondary_background.npy",
            "figure5_secondary_trajectories": "paper_data/projected/ho_figure5/secondary_trajectories.npy",
            "figure5_secondary_points": "paper_data/projected/ho_figure5/secondary_points.npy",
            "figure5_sampling_manifest": "paper_data/projected/ho_figure5/sampling_manifest.json",
            "figure5_de_rna": "paper_data/archived/ho_figure5/de_main_var_path_wilcoxon.csv",
            "figure5_de_atac": "paper_data/archived/ho_figure5/de_sec_var_path_wilcoxon.csv",
            "figure5_rna_trajectory_original_pdf": "paper_data/archived/ho_figure5/ode_trajectories_v2_original.pdf",
            "figure5_rna_trajectory_original_png": "paper_data/archived/ho_figure5/ode_trajectories_v2_original.png",
        },
    },
    "hc_author5": {
        "accession": "GSE204684",
        "files": {
            "processed_rna": "paper_data/processed/hc_author5/rna.h5ad",
            "processed_atac": "paper_data/processed/hc_author5/atac.h5ad",
            "dynamics": "paper_data/dynamics/hc_author5/adata.h5ad",
            "training_config": "paper_data/dynamics/hc_author5/config.yaml",
            "map": "paper_data/maps/hc_author5/best_model.pt",
            "erbb4_umap_primary": "paper_data/umap/hc_author5/erbb4_primary.pkl",
            "erbb4_umap_secondary": "paper_data/umap/hc_author5/erbb4_secondary.pkl",
        },
    },
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_name(dataset_id: str, role: str, path: Path) -> str:
    return f"{dataset_id}-{role}{path.suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data_manifest.yaml")
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "release_tag": "paper-data-v1",
        "release_base_url": (
            "https://github.com/jiantaoShen12/MOFI/releases/download/paper-data-v1"
        ),
        "datasets": {},
    }
    for dataset_id, dataset in DATASETS.items():
        entry = {key: value for key, value in dataset.items() if key != "files"}
        entry["files"] = {}
        for role, rel in dataset["files"].items():
            path = REPO_ROOT / rel
            if not path.exists():
                raise FileNotFoundError(path)
            entry["files"][role] = {
                "path": rel.replace("\\", "/"),
                "asset": asset_name(dataset_id, role, path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
        payload["datasets"][dataset_id] = entry
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
