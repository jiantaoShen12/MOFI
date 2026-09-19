"""Render the HSPC z=0 ITGA2B/CD41 curve from the velocity candidate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finetune_hspc_velocity import _integrate, _load_dynamics, _load_fixed_map


def _feature_bundle(proc: ad.AnnData, feature: str):
    names = [str(x) for x in proc.uns["original_gene_info"]["var_names"]]
    idx = names.index(feature)
    pca = proc.uns["pca"]
    col = np.asarray(pca["reconstruction_matrix"], dtype=np.float32)[:, idx]
    mean = float(np.asarray(pca["input_mean"], dtype=np.float32)[idx])
    return col, mean


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dyn_path = ROOT / "paper_data/dynamics/hspc_31800/adata.h5ad"
    dyn = ad.read_h5ad(dyn_path)
    rna = ad.read_h5ad(ROOT / "paper_data/processed/hspc_31800/rna.h5ad")
    protein = ad.read_h5ad(ROOT / "paper_data/processed/hspc_31800/protein.h5ad")
    model = _load_dynamics(dyn, device)
    mapper = _load_fixed_map(ROOT / "paper_data/maps/hspc_31800/best_model.pt", device)
    assets = ROOT / "results/figure_4_conservative_finetuned_paper/coarse_dense/sample/assets"
    x0_np = np.asarray(np.load(assets / "traj_main.npy", allow_pickle=False)[0], dtype=np.float32)
    times = np.linspace(0.0, 3.0, 10, dtype=np.float32)
    x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        latent = _integrate(model, x0, times, dt=0.05)
        primary_col, primary_mean = _feature_bundle(rna, "ITGA2B")
        secondary_col, secondary_mean = _feature_bundle(protein, "CD41")
        primary_col_t = torch.as_tensor(primary_col, dtype=torch.float32, device=device)
        secondary_col_t = torch.as_tensor(secondary_col, dtype=torch.float32, device=device)
        primary = (latent @ primary_col_t + primary_mean).mean(dim=1).cpu().numpy()
        secondary_parts = []
        for i in range(latent.shape[0]):
            vals = []
            for start in range(0, latent.shape[1], 1024):
                mapped = mapper.forward_single(latent[i, start : start + 1024])
                vals.append(mapped @ secondary_col_t + secondary_mean)
            secondary_parts.append(torch.cat(vals).mean().cpu().item())
        secondary = np.asarray(secondary_parts, dtype=float)
    out_dir = ROOT / "results/figure_4_velocity_finetuned_paper/coarse_dense/sample/tables/tf/reconstruction_primary/primary/batch/a_itga2b_cd41"
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = pd.DataFrame({
        "task_id": "a_itga2b_cd41", "tier": "A", "label": "ITGA2B->CD41", "perturb_domain": "primary",
        "z_score": 0.0, "time": times, "primary_value": primary, "secondary_value": secondary,
        "primary_vars": "ITGA2B", "secondary_vars": "CD41", "n_primary_vars": 1, "n_secondary_vars": 1,
    })
    sim.to_csv(out_dir / "sim_curve__z-0.csv", index=False)
    real = pd.read_csv(ROOT / "paper_data/archived/hspc_figure4/itga2b_cd41/real_curve.csv")
    real.to_csv(out_dir / "real_curve.csv", index=False)
    # Reuse the released plotting routine for the visible figure.
    from CytoBridge.pl.tf.perturb_pipeline import plot_saved_perturbation_batch_curves
    fig_dir = ROOT / "results/figure_4_velocity_finetuned_paper/coarse_dense/sample/figures/tf/paper_primary/primary/batch"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = plot_saved_perturbation_batch_curves(
        out_dir, fig_dir / "perturb_batch_curve__task-a-itga2b-cd41__z0_velocity_finetuned.png",
        scores=[0], primary_var="ITGA2B", secondary_var="CD41",
        title_suffix="ITGA2B->CD41 (velocity-only fine-tuned z=0)",
    )
    print(json.dumps({"table": str(out_dir / "sim_curve__z-0.csv"), "figure": str(out["output"]), "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
