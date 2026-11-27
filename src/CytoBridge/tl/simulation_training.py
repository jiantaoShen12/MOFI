"""Training implementation for the bundled two-dimensional simulation only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import ot
from CytoBridge.tl.analysis import simulate_trajectory
import torch
import yaml

from .losses import calc_mass_loss, calc_ot_loss
from .methods import MongeMap_simu, ODEFunc, neural_ode_step
from .models import DynamicalModel
from CytoBridge.utils.utils import sample


class SimulationTrainer:
    def __init__(self, model, config, batch_size, device):
        self.model = model.to(device)
        self.config, self.batch_size, self.device = config, batch_size, device
        opts = config["model"]["multi"]
        if model.latent_dim != 2 or not opts.get("simu_multi_use"):
            raise ValueError("Only the bundled two-dimensional simulation is supported")
        transform = MongeMap_simu(**opts.get("simu_map_params", {}))
        self.ode = ODEFunc(model=self.model, tranmap_simu=transform,
                           multi_alpha=opts.get("multi_alpha", 1), use_mass=True)

    def epoch(self, stage, data, times, optimizer):
        x0 = sample(data[0], self.batch_size).to(self.device)
        lnw0 = torch.log(torch.ones(self.batch_size, 1) / self.batch_size).to(self.device)
        initial_mass = data[0].shape[0]
        total_loss = 0.0
        for idx in range(1, len(times)):
            optimizer.zero_grad()
            target = sample(data[idx], self.batch_size).to(self.device)
            x1, lnw1, energy = neural_ode_step(self.ode, x0, lnw0, times[idx - 1], times[idx], self.device)
            loss = (stage["lambda_ot"] * calc_ot_loss(x1, target, lnw1, stage["OT_loss"])
                    + stage["lambda_mass"] * calc_mass_loss(x1, target, lnw1, data[idx].shape[0] / initial_mass, stage.get("global_mass", False))
                    + stage["lambda_energy"] * energy.mean())
            loss.backward()
            optimizer.step()
            x0, lnw0 = x1.detach().clone(), lnw1.detach().clone()
            total_loss += loss.item()
        return total_loss / (len(times) - 1)

    def train(self, data, times):
        for item in self.config["training"]["plan"]:
            stage = {**self.config["training"]["defaults"], **item}
            if stage["mode"] != "neural_ode":
                raise ValueError("Only simulation neural ODE training is supported")
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=stage["lr"])
            best_loss, best_state = float("inf"), copy.deepcopy(self.model.state_dict())
            for epoch in range(stage["epochs"]):
                loss = self.epoch(stage, data, times, optimizer)
                if not np.isfinite(loss):
                    raise RuntimeError(f"Non-finite simulation loss at epoch {epoch}")
                if loss < best_loss:
                    best_loss, best_state = loss, copy.deepcopy(self.model.state_dict())
                if epoch % 10 == 0:
                    print(f"{stage['name']} {epoch + 1}/{stage['epochs']}: {loss:.6f}", flush=True)
            if stage.get("save_strategy", "best") == "best":
                self.model.load_state_dict(best_state)
                filename = "best.pth"
            else:
                self.epoch(stage, data, times, optimizer)
                filename = "last_model.pth"
            dest = Path(self.config["ckpt_dir"]) / stage["name"]
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), dest / filename)
        return self.model


def fit_simulation(adata, *, config, batch_size=400, device="cpu"):
    times = sorted(adata.obs["time_point_processed"].unique())
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    if latent.shape[1] != 2:
        raise ValueError("This training entry point supports only the simulation's 2D coordinates")
    data = [torch.tensor(latent[adata.obs["time_point_processed"] == t], device=device) for t in times]
    model = SimulationTrainer(DynamicalModel(2, config["model"]), config, batch_size, device).train(data, times)
    with torch.no_grad():
        x = torch.as_tensor(latent, device=device)
        t = torch.as_tensor(np.asarray(adata.obs["time_point_processed"], dtype=np.float32), device=device)[:, None]
        out = model(t, x, torch.zeros((len(x), 1), device=device))
    adata.obsm["velocity_latent"] = out["velocity"].cpu().numpy()
    adata.obsm["growth_rate"] = out["growth"].cpu().numpy()
    adata.uns["all_model"] = {"model_config": config["model"],
                              "training_config": {"defaults": config["training"]["defaults"],
                                                  "plan": json.dumps(config["training"]["plan"])},
                              "model_state_dict": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}
    dest = Path(config["ckpt_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(dest / "adata.h5ad")
    (dest / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return adata


def evaluate_simulation(model, adata, data, time_points, device, val=None):
    """Evaluate trained model using Wasserstein-1 distance and Total Mass Variation (TMV)

    Args:
        data: List of tensors where each element represents samples at a specific time point
        time_points: List of time values corresponding to each element in 'data'

    Returns:
        list: List of Wasserstein-1 distances for each time point (excluding initial time)
    """
    print(f"\n--- Starting Evaluation ---")
    # Get initial time point data (t=0)
    if isinstance(val, (int, float)):   # 纯 Python 数字
        n = int(val)
        T = len(data)
        N0 = data[0].shape[0]
        n0 = min(n, N0)
        ratio = n0 / N0 if N0 > 0 else 1.0
        generator = torch.Generator()
        generator.manual_seed(42)
        idx0 = torch.randperm(N0, generator=generator)[:n0]
        x0 = data[0][idx0].to(device)
        new_data = [None] * T
        new_data[0] = data[0][idx0]
        for t in range(1, T):
            Nt = data[t].shape[0]
            nt = min(max(int(round(Nt * ratio)), 1), Nt)
            gen_t = torch.Generator()
            gen_t.manual_seed(42 + t)
            idx_t = torch.randperm(Nt, generator=gen_t)[:nt]
            new_data[t] = data[t][idx_t]
        data = new_data
    else:
        x0 = data[0].to(device)
    # Freeze model parameters during evaluation (disable gradient computation)
    for param in model.parameters():
        param.requires_grad = False

    # Get sigma parameter (use stored value or default to 0.05 if not available)
    sigma = 0

    # Simulate trajectory using the trained model
    point, weight = simulate_trajectory(
        adata,
        model,
        x0,
        sigma,
        time_points,
        dt=0.1,  # Time step for ODE simulation
        device=x0.device
    )

    # Calculate Wasserstein-1 distance for each time point (excluding initial time)
    wasserstein_scores = []
    tmv_scores = []
    for idx in range(1, len(time_points)):
        t0, t1 = time_points[0], time_points[idx]
        # Get target data at current time point (convert to numpy for OT computation)
        data_t1 = data[idx].detach().cpu().numpy()
        # Get predicted positions and weights from simulated trajectory
        x1 = point[idx]
        m1 = weight[idx]

        # Calculate Total Mass Variation (TMV) between predicted and true mass
        tmv = np.abs(m1.sum() - data[idx].shape[0] / data[0].shape[0])
        # Normalize predicted weights to sum to 1 (required for OT)
        m1 = m1 / m1.sum()

        # Create uniform weights for target data (sum to 1)
        m2 = np.ones(data_t1.shape[0]) / data_t1.shape[0]
        # Compute Euclidean distance matrix between target and predicted points
        cost_matrix = ot.dist(data_t1, x1, metric='euclidean')

        # Calculate Wasserstein-1 distance using Earth Mover's Distance (EMD)
        w1 = ot.emd2(
            m2,
            m1.reshape(-1),  # Reshape to 1D array (required by ot.emd2)
            cost_matrix,
            numItermax=1e7  # Increase max iterations for convergence
        )

        # Store results and print progress
        wasserstein_scores.append(w1)
        tmv_scores.append(tmv)
        print(f"  Time Point {t1}: Wasserstein-1 Distance = {w1:.4f}")
        print(f"  Time Point {t1}: TMV = {tmv:.4f}")

    return {
        'w1_scores': wasserstein_scores,
        'tmv_scores': tmv_scores
    }
