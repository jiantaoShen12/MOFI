#!/usr/bin/env python3
"""Fine-tune the HSPC RNA-to-protein map on the integrated CD41 trajectory.

This is model training, not curve calibration. The released ODE is integrated
from the observed HSPC time-0 RNA cells; only the final layer of the protein
decoder is optimized. The objective combines decoded-CD41 trajectory error,
paired latent reconstruction and retention to the paper map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from CytoBridge.tl.map_finetune import fine_tune_primary_to_secondary_feature_trajectory  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def read_slim_processed(path: Path, *, include_pca: bool) -> ad.AnnData:
    """Read only latent/time/PCA fields from a processed AnnData HDF5 file."""
    with h5py.File(path, "r") as handle:
        latent = np.asarray(handle["obsm/X_latent"], dtype=np.float32)
        obs_names = _decode_strings(np.asarray(handle["obs/_index"]))
        times = np.asarray(handle["obs/time_point_processed"], dtype=np.float32)
        pca = None
        original_names = None
        if include_pca:
            pca = {
                "input_mean": np.asarray(handle["uns/pca/input_mean"], dtype=np.float32),
                "projection_matrix": np.asarray(
                    handle["uns/pca/projection_matrix"], dtype=np.float32
                ),
                "reconstruction_matrix": np.asarray(
                    handle["uns/pca/reconstruction_matrix"], dtype=np.float32
                ),
            }
            original_names = _decode_strings(
                np.asarray(handle["uns/original_gene_info/var_names"])
            )
    result = ad.AnnData(
        X=latent.copy(),
        obs=pd.DataFrame({"time_point_processed": times}, index=obs_names),
    )
    result.obsm["X_latent"] = latent
    if include_pca:
        result.uns["pca"] = pca
        result.uns["original_gene_info"] = {"var_names": np.asarray(original_names, dtype=object)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--curve-weight", type=float, default=8.0)
    parser.add_argument("--paired-weight", type=float, default=0.25)
    parser.add_argument("--retention-weight", type=float, default=0.60)
    parser.add_argument("--non-target-feature-weight", type=float, default=2.0)
    parser.add_argument("--retention-features", default="CD42b,CD71")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-trajectory-cells-per-split", type=int, default=512)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--minimum-late-gain", type=float, default=0.25)
    parser.add_argument("--maximum-paired-degradation", type=float, default=0.03)
    parser.add_argument("--trajectory-npy", type=Path)
    parser.add_argument("--weights-npy", type=Path)
    parser.add_argument("--trajectory-times", default="")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    data_dir = ROOT / "paper_data" / "processed" / "hspc_31800"
    dynamics_path = ROOT / "paper_data" / "dynamics" / "hspc_31800" / "adata.h5ad"
    map_dir = ROOT / "paper_data" / "maps" / "hspc_31800"
    checkpoint_path = map_dir / "best_model.pt"
    original_path = map_dir / "best_model_original.pt"
    candidate_path = map_dir / "best_model_trajectory_finetuned.pt"
    audit_path = ROOT / "validation" / "hspc_map_finetune_audit.json"

    # Always restart from the immutable paper checkpoint when a previous run
    # has already activated a candidate. This makes notebook reruns idempotent.
    baseline_path = original_path if original_path.exists() else checkpoint_path
    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)

    trajectory = None
    trajectory_weights = None
    trajectory_times = None
    if args.trajectory_npy is not None or args.weights_npy is not None:
        if args.trajectory_npy is None or args.weights_npy is None:
            raise ValueError("--trajectory-npy and --weights-npy must be supplied together")
        trajectory = np.load(args.trajectory_npy, allow_pickle=False)
        trajectory_weights = np.load(args.weights_npy, allow_pickle=False)
        if str(args.trajectory_times).strip():
            trajectory_times = [float(value) for value in str(args.trajectory_times).split(",")]
        else:
            trajectory_times = np.linspace(0.0, 3.0, trajectory.shape[0]).tolist()
        primary = read_slim_processed(data_dir / "rna.h5ad", include_pca=False)
        secondary = read_slim_processed(data_dir / "protein.h5ad", include_pca=True)
        dynamics = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    else:
        primary = ad.read_h5ad(data_dir / "rna.h5ad")
        secondary = ad.read_h5ad(data_dir / "protein.h5ad")
        dynamics = ad.read_h5ad(dynamics_path)

    retention_features = [
        value.strip() for value in str(args.retention_features).split(",") if value.strip()
    ]
    result = fine_tune_primary_to_secondary_feature_trajectory(
        primary=primary,
        secondary=secondary,
        dynamics=dynamics,
        checkpoint=baseline_checkpoint,
        feature_name="CD41",
        time_key="time_point_processed",
        times=(0.0, 1.0, 2.0, 3.0),
        dt=0.05,
        device=str(args.device),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        curve_weight=float(args.curve_weight),
        paired_weight=float(args.paired_weight),
        retention_weight=float(args.retention_weight),
        non_target_feature_weight=float(args.non_target_feature_weight),
        retention_feature_names=retention_features,
        batch_size=int(args.batch_size),
        max_trajectory_cells_per_split=int(args.max_trajectory_cells_per_split),
        patience=int(args.patience),
        seed=int(args.seed),
        precomputed_trajectory=trajectory,
        precomputed_weights=trajectory_weights,
        precomputed_times=trajectory_times,
    )

    candidate_checkpoint = dict(baseline_checkpoint)
    candidate_checkpoint["model_state_dict"] = dict(result.state_dict)
    candidate_checkpoint["epoch"] = int(result.best_epoch)
    candidate_checkpoint["fine_tune_metadata"] = {
        "method": "trajectory-guided decoded-CD41 fine-tuning",
        "feature": "CD41",
        "trainable_modules": ["decoder2.final_linear"],
        "dynamics_model_changed": False,
        "posthoc_curve_adjustment": False,
        "seed": int(args.seed),
    }
    torch.save(candidate_checkpoint, candidate_path)

    late_gain = float(result.audit["heldout_late_curve_rmse_gain_fraction"])
    paired_change = float(result.audit["heldout_paired_latent_mse"]["relative_change"])
    accepted = bool(
        late_gain >= float(args.minimum_late_gain)
        and paired_change <= float(args.maximum_paired_degradation)
    )
    activated = False
    if bool(args.activate) and accepted:
        if not original_path.exists():
            shutil.copy2(checkpoint_path, original_path)
        shutil.copy2(candidate_path, checkpoint_path)
        activated = True

    audit = dict(result.audit)
    parameter_record = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    audit.update(
        {
            "dataset": "HSPC 31800",
            "objective": "reduce integrated z=0 CD41 error after RNA-to-protein mapping and protein PCA reconstruction",
            "inputs": {
                "primary": str((data_dir / "rna.h5ad").relative_to(ROOT)),
                "secondary": str((data_dir / "protein.h5ad").relative_to(ROOT)),
                "dynamics": str(dynamics_path.relative_to(ROOT)),
                "baseline_checkpoint": str(baseline_path.relative_to(ROOT)),
                "baseline_checkpoint_sha256": sha256(baseline_path),
            },
            "parameters": parameter_record,
            "acceptance_rule": {
                "minimum_heldout_late_curve_rmse_gain_fraction": float(args.minimum_late_gain),
                "maximum_heldout_paired_latent_mse_degradation_fraction": float(
                    args.maximum_paired_degradation
                ),
            },
            "accepted": accepted,
            "activated": activated,
            "candidate_checkpoint": str(candidate_path.relative_to(ROOT)),
            "candidate_checkpoint_sha256": sha256(candidate_path),
            "active_checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "active_checkpoint_sha256": sha256(checkpoint_path),
        }
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": accepted,
                "activated": activated,
                "heldout_late_curve_rmse_gain_fraction": late_gain,
                "heldout_paired_latent_mse_relative_change": paired_change,
                "best_epoch": int(result.best_epoch),
                "candidate": str(candidate_path),
                "audit": str(audit_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
