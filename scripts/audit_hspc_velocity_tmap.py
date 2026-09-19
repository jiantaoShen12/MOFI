#!/usr/bin/env python3
"""Audit the joint HSPC velocity+tmap candidate against immutable baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finetune_hspc_velocity import _load_dynamics, _load_fixed_map  # noqa: E402
from finetune_hspc_velocity_tmap import (  # noqa: E402
    _all_feature_bundle,
    _evaluate,
    _feature_bundle,
    _observed_target,
)


def _metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return {
        "rmse_all": float(np.sqrt(np.mean(residual**2))),
        "rmse_late": float(np.sqrt(np.mean(residual[1:] ** 2))),
        "t0_abs_error": float(abs(residual[0])),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dyn_dir = ROOT / "paper_data" / "dynamics" / "hspc_31800"
    map_dir = ROOT / "paper_data" / "maps" / "hspc_31800"
    data_dir = ROOT / "paper_data" / "processed" / "hspc_31800"
    baseline_dyn = ad.read_h5ad(dyn_dir / "adata_velocity_tmap_base.h5ad")
    candidate_dyn = ad.read_h5ad(dyn_dir / "adata_velocity_tmap_finetuned.h5ad")
    active_dyn = ad.read_h5ad(dyn_dir / "adata.h5ad")
    rna = ad.read_h5ad(data_dir / "rna.h5ad")
    protein = ad.read_h5ad(data_dir / "protein.h5ad")
    assets = ROOT / "results" / "figure_4_conservative_finetuned_paper" / "coarse_dense" / "sample" / "assets"
    x0 = np.asarray(np.load(assets / "traj_main.npy", allow_pickle=False)[0], dtype=np.float32)
    times = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    primary_col, primary_mean, _ = _feature_bundle(rna, "ITGA2B")
    secondary_col, secondary_mean, _ = _feature_bundle(protein, "CD41")
    reconstruction, pca_mean, names = _all_feature_bundle(protein)

    base_dyn_model = _load_dynamics(baseline_dyn, device).eval()
    candidate_dyn_model = _load_dynamics(candidate_dyn, device).eval()
    base_map_model = _load_fixed_map(map_dir / "best_model_velocity_tmap_base.pt", device)
    candidate_map_model = _load_fixed_map(map_dir / "best_model_velocity_tmap_finetuned.pt", device)
    active_map_model = _load_fixed_map(map_dir / "best_model.pt", device)
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    primary_col = primary_col.to(device)
    primary_mean = primary_mean.to(device)
    secondary_col = secondary_col.to(device)
    secondary_mean = secondary_mean.to(device)
    reconstruction = reconstruction.to(device)
    pca_mean = pca_mean.to(device)

    base_traj, base_map, base_primary, base_secondary = _evaluate(
        base_dyn_model,
        base_map_model,
        x0_t,
        times,
        dt=0.05,
        primary_column=primary_col,
        primary_mean=primary_mean,
        secondary_column=secondary_col,
        secondary_mean=secondary_mean,
        batch_size=1024,
    )
    candidate_traj, candidate_map, candidate_primary, candidate_secondary = _evaluate(
        candidate_dyn_model,
        candidate_map_model,
        x0_t,
        times,
        dt=0.05,
        primary_column=primary_col,
        primary_mean=primary_mean,
        secondary_column=secondary_col,
        secondary_mean=secondary_mean,
        batch_size=1024,
    )
    active_traj, active_map, active_primary, active_secondary = _evaluate(
        _load_dynamics(active_dyn, device).eval(),
        active_map_model,
        x0_t,
        times,
        dt=0.05,
        primary_column=primary_col,
        primary_mean=primary_mean,
        secondary_column=secondary_col,
        secondary_mean=secondary_mean,
        batch_size=1024,
    )
    base_protein = base_map @ reconstruction + pca_mean
    candidate_protein = candidate_map @ reconstruction + pca_mean
    active_protein = active_map @ reconstruction + pca_mean
    observed_by_feature = {
        name: _observed_target(protein, name, times)
        for name in names
    }
    base_protein_np = base_protein.cpu().numpy()
    candidate_protein_np = candidate_protein.cpu().numpy()
    marker_rows = []
    for index, name in enumerate(names):
        base_curve = base_protein_np[:, :, index].mean(axis=1)
        candidate_curve = candidate_protein_np[:, :, index].mean(axis=1)
        active_curve = active_protein.cpu().numpy()[:, :, index].mean(axis=1)
        observed = observed_by_feature[name]
        base_metric = _metrics(base_curve, observed)
        candidate_metric = _metrics(candidate_curve, observed)
        marker_rows.append({
            "marker": name,
            "baseline": base_metric,
            "candidate": candidate_metric,
            "relative_rmse_change": float(candidate_metric["rmse_all"] / max(base_metric["rmse_all"], 1e-8) - 1.0),
            "curve_max_abs_change": float(np.max(np.abs(candidate_curve - base_curve))),
            "candidate_vs_active_curve_max_abs_change": float(np.max(np.abs(candidate_curve - active_curve))),
        })

    base_primary_np = base_primary.cpu().numpy()
    candidate_primary_np = candidate_primary.cpu().numpy()
    base_secondary_np = base_secondary.cpu().numpy()
    candidate_secondary_np = candidate_secondary.cpu().numpy()
    target_itga2b = _observed_target(rna, "ITGA2B", times)
    target_cd41 = _observed_target(protein, "CD41", times)
    non_target = [row for row in marker_rows if row["marker"] != "CD41"]
    non_target_max = max(row["relative_rmse_change"] for row in non_target)
    growth_unchanged = all(
        torch.equal(value, base_dyn_model.growth_net.state_dict()[key])
        for key, value in candidate_dyn_model.growth_net.state_dict().items()
    )
    map_non_decoder_unchanged = all(
        torch.equal(value, base_map_model.state_dict()[key])
        for key, value in candidate_map_model.state_dict().items()
        if not key.startswith("decoder2.final_linear.")
    )
    audit = {
        "status": "PASS" if (
            _metrics(candidate_primary_np, target_itga2b)["rmse_all"] < _metrics(base_primary_np, target_itga2b)["rmse_all"]
            and _metrics(candidate_secondary_np, target_cd41)["rmse_all"] < _metrics(base_secondary_np, target_cd41)["rmse_all"]
            and non_target_max <= 0.10
        ) else "REVIEW",
        "device": str(device),
        "times": times.tolist(),
        "target_features": ["ITGA2B", "CD41"],
        "primary": {
            "baseline_curve": base_primary_np.tolist(),
            "candidate_curve": candidate_primary_np.tolist(),
            "observed_curve": target_itga2b.tolist(),
            "baseline_metrics": _metrics(base_primary_np, target_itga2b),
            "candidate_metrics": _metrics(candidate_primary_np, target_itga2b),
        },
        "secondary": {
            "baseline_curve": base_secondary_np.tolist(),
            "candidate_curve": candidate_secondary_np.tolist(),
            "observed_curve": target_cd41.tolist(),
            "baseline_metrics": _metrics(base_secondary_np, target_cd41),
            "candidate_metrics": _metrics(candidate_secondary_np, target_cd41),
        },
        "previous_active": {
            "primary_curve": active_primary.cpu().numpy().tolist(),
            "secondary_curve": active_secondary.cpu().numpy().tolist(),
            "primary_metrics": _metrics(active_primary.cpu().numpy(), target_itga2b),
            "secondary_metrics": _metrics(active_secondary.cpu().numpy(), target_cd41),
        },
        "marker_rows": marker_rows,
        "non_target_marker_max_relative_rmse_change": float(non_target_max),
        "trajectory_relative_l2_change": float(
            torch.linalg.vector_norm(candidate_traj - base_traj)
            / torch.clamp(torch.linalg.vector_norm(base_traj), min=1e-8)
        ),
        "growth_unchanged": bool(growth_unchanged),
        "map_non_decoder_unchanged": bool(map_non_decoder_unchanged),
        "posthoc_curve_adjustment": False,
        "candidate_dynamics": "paper_data/dynamics/hspc_31800/adata_velocity_tmap_finetuned.h5ad",
        "candidate_map": "paper_data/maps/hspc_31800/best_model_velocity_tmap_finetuned.pt",
    }
    output = ROOT / "validation" / "hspc_velocity_tmap_reconstruction_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
