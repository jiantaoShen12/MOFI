"""Recreate the original HO ``umap_adata`` ODE-v3 projection.

The legacy paper call fits UMAP jointly on observed latent cells, saved ODE
trajectory states, and saved ODE sample points.  This utility keeps that
coordinate construction explicit and routes the final drawing through the
original ``CytoBridge.pl.plot_ode_v3`` implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from joblib import parallel_backend

from CytoBridge.pl.plot_ode_v3 import plot_ode_v3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    dynamics = ad.read_h5ad(root / "paper_data" / "dynamics" / "ho" / "adata.h5ad")
    assets = root / "paper_data" / "projected" / "ho_figure5"
    traj = np.load(assets / "primary_trajectories_raw.npy")
    points = np.load(assets / "primary_points_raw.npy")
    if traj.ndim != 3 or points.ndim != 3:
        raise ValueError((traj.shape, points.shape))
    x_raw = np.asarray(dynamics.obsm["X_latent"], dtype=np.float32)
    flat_traj = traj.reshape(-1, traj.shape[-1]).astype(np.float32)
    flat_points = points.reshape(-1, points.shape[-1]).astype(np.float32)
    joint = ad.AnnData(X=np.concatenate([x_raw, flat_traj, flat_points], axis=0))
    # A single worker is deterministic and avoids Windows joblib worker ACLs.
    sc.settings.n_jobs = 1
    with parallel_backend("threading", n_jobs=1):
        sc.pp.neighbors(joint, n_neighbors=40, random_state=42)
        sc.tl.umap(joint, random_state=42)
    emb = np.asarray(joint.obsm["X_umap"], dtype=np.float32)
    n = x_raw.shape[0]
    t_end = n + flat_traj.shape[0]
    background = emb[:n]
    traj_2d = emb[n:t_end].reshape(traj.shape[0], traj.shape[1], 2)
    points_2d = emb[t_end:].reshape(points.shape[0], points.shape[1], 2)
    np.save(out / "primary_background.npy", background)
    np.save(out / "primary_trajectories.npy", traj_2d)
    np.save(out / "primary_points.npy", points_2d)
    times = np.sort(np.asarray(dynamics.obs["time_point_processed"].unique(), dtype=float))
    X = [background[np.isclose(np.asarray(dynamics.obs["time_point_processed"], dtype=float), t)] for t in times]
    labels = [dynamics.obs.loc[np.isclose(np.asarray(dynamics.obs["time_point_processed"], dtype=float), t), "cell_type"].astype(str).to_numpy() for t in times]
    plot_ode_v3(
        X=X,
        traj_array=traj_2d,
        save_path=str(out / "ode_trajectories_primary.png"),
        traj_times=np.linspace(float(times[0]), float(times[-1]), traj_2d.shape[0]),
        time_ticks=times,
        bg_obs_values=labels,
        bg_obs_name="cell_type",
        bg_palette="tab10",
        show_axes=True,
    )
    print({"status": "PASS", "background_shape": list(background.shape), "trajectory_shape": list(traj_2d.shape), "out_dir": str(out)})


if __name__ == "__main__":
    main()
