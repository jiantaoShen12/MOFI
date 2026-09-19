#!/usr/bin/env python3
"""Jointly fine-tune the HSPC velocity field and RNA-to-protein tmap.

The candidate always starts from the immutable paper dynamics and map.  Only
``velocity_net.output_layer`` and ``decoder2.final_linear`` are trainable.
The loss fits the all-time-0, z=0 reconstruction of the paired ITGA2B/CD41
curve while retaining the released latent trajectories, paired protein map,
and non-target protein features.  No curve is edited after inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune_hspc_velocity import (  # noqa: E402
    _integrate,
    _load_dynamics,
    _load_fixed_map,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_bundle(processed: ad.AnnData, feature: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    names = [str(value) for value in processed.uns["original_gene_info"]["var_names"]]
    index = names.index(str(feature))
    pca = processed.uns["pca"]
    column = np.asarray(pca["reconstruction_matrix"], dtype=np.float32)[:, index]
    mean = float(np.asarray(pca["input_mean"], dtype=np.float32)[index])
    return torch.as_tensor(column), torch.as_tensor(mean), index


def _all_feature_bundle(processed: ad.AnnData) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    names = [str(value) for value in processed.uns["original_gene_info"]["var_names"]]
    pca = processed.uns["pca"]
    reconstruction = torch.as_tensor(
        np.asarray(pca["reconstruction_matrix"], dtype=np.float32)
    )
    mean = torch.as_tensor(np.asarray(pca["input_mean"], dtype=np.float32))
    return reconstruction, mean, names


def _map_latent(mapper, trajectory: torch.Tensor, *, batch_size: int = 1024) -> torch.Tensor:
    flat = trajectory.reshape(-1, trajectory.shape[-1])
    chunks = []
    for start in range(0, int(flat.shape[0]), int(batch_size)):
        chunks.append(mapper.forward_single(flat[start : start + int(batch_size)]))
    return torch.cat(chunks, dim=0).reshape(trajectory.shape[0], trajectory.shape[1], -1)


def _integrate_with_weights(
    model,
    x0: torch.Tensor,
    times: np.ndarray,
    *,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable deterministic counterpart of the release ODE runner.

    The notebook aggregates trajectories with the growth-derived particle
    weights.  The former fine-tune objective used an unweighted mean, so its
    audit could look good while the actual perturbation notebook still showed
    a large CD41 error.  Keep the same Euler drift and log-weight update used
    by ``simulate_trajectory(..., sigma=0)``.
    """
    state = x0
    n_cells = int(x0.shape[0])
    lnw = torch.full(
        (n_cells, 1),
        -math.log(max(n_cells, 1)),
        dtype=x0.dtype,
        device=x0.device,
    )
    states = [state]
    log_weights = [lnw]
    current = float(times[0])
    # The lightweight _Dynamics loader intentionally does not define the
    # full DynamicalModel.components list.  Presence of growth_net is the
    # authoritative check for this HSPC model and matches simulate_trajectory.
    has_growth = hasattr(model, "growth_net")
    for target in times[1:]:
        target = float(target)
        n_steps = max(1, int(np.ceil((target - current) / float(dt))))
        h = (target - current) / n_steps
        for _ in range(n_steps):
            t_col = torch.full((state.shape[0], 1), current, dtype=state.dtype, device=state.device)
            velocity = model.velocity_net(torch.cat([state, t_col], dim=1))
            if has_growth:
                growth = model.growth_net(torch.cat([state, t_col], dim=1))
            else:
                growth = torch.zeros_like(lnw)
            state = state + float(h) * velocity
            lnw = lnw + float(h) * growth
            current += h
        states.append(state)
        log_weights.append(lnw)
    return torch.stack(states), torch.stack(log_weights)


def _normalized_weights(log_weights: torch.Tensor) -> torch.Tensor:
    return torch.softmax(log_weights.squeeze(-1), dim=1)


def _curve_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    late = residual[1:] if residual.size > 1 else residual
    return {
        "rmse_all": float(np.sqrt(np.mean(residual**2))),
        "rmse_late": float(np.sqrt(np.mean(late**2))),
        "t0_abs_error": float(abs(residual[0])),
    }


def _evaluate(
    dynamics,
    mapper,
    x0: torch.Tensor,
    times: np.ndarray,
    *,
    dt: float,
    primary_column: torch.Tensor,
    primary_mean: torch.Tensor,
    secondary_column: torch.Tensor,
    secondary_mean: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        trajectory, log_weights = _integrate_with_weights(dynamics, x0, times, dt=float(dt))
        mapped = _map_latent(mapper, trajectory, batch_size=batch_size)
        weights = _normalized_weights(log_weights)
        primary_values = trajectory @ primary_column + primary_mean
        secondary_values = mapped @ secondary_column + secondary_mean
        primary_curve = (primary_values * weights).sum(dim=1)
        secondary_curve = (secondary_values * weights).sum(dim=1)
    return trajectory, mapped, primary_curve, secondary_curve


def _observed_target(
    processed: ad.AnnData,
    feature: str,
    times: np.ndarray,
) -> np.ndarray:
    column, mean, _ = _feature_bundle(processed, feature)
    latent = np.asarray(processed.obsm["X_latent"], dtype=np.float32)
    values = latent @ column.numpy() + float(mean.item())
    observed_times = np.asarray(processed.obs["time_point_processed"], dtype=float)
    return np.asarray(
        [float(values[np.isclose(observed_times, float(time_value))].mean()) for time_value in times],
        dtype=np.float32,
    )


def _state_to_numpy(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-cells", type=int, default=2048)
    parser.add_argument("--eval-cells", type=int, default=2048)
    parser.add_argument("--learning-rate-velocity", type=float, default=2e-5)
    parser.add_argument("--learning-rate-tmap", type=float, default=3e-4)
    parser.add_argument("--primary-weight", type=float, default=8.0)
    parser.add_argument("--secondary-weight", type=float, default=8.0)
    parser.add_argument("--trajectory-retention-weight", type=float, default=0.40)
    parser.add_argument("--velocity-retention-weight", type=float, default=0.30)
    parser.add_argument("--map-retention-weight", type=float, default=0.60)
    parser.add_argument("--non-target-retention-weight", type=float, default=4.0)
    parser.add_argument("--primary-non-target-retention-weight", type=float, default=4.0)
    parser.add_argument("--paired-weight", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dyn_dir = ROOT / "paper_data" / "dynamics" / "hspc_31800"
    map_dir = ROOT / "paper_data" / "maps" / "hspc_31800"
    data_dir = ROOT / "paper_data" / "processed" / "hspc_31800"
    active_dyn = dyn_dir / "adata.h5ad"
    baseline_dyn = dyn_dir / "adata_velocity_tmap_base.h5ad"
    candidate_dyn = dyn_dir / "adata_velocity_tmap_finetuned.h5ad"
    active_map = map_dir / "best_model.pt"
    baseline_map = map_dir / "best_model_velocity_tmap_base.pt"
    candidate_map = map_dir / "best_model_velocity_tmap_finetuned.pt"
    audit_path = ROOT / "validation" / "hspc_velocity_tmap_finetune_audit.json"
    assets = ROOT / "results" / "figure_4_conservative_finetuned_paper" / "coarse_dense" / "sample" / "assets"

    if not baseline_dyn.exists():
        shutil.copy2(active_dyn, baseline_dyn)
    if not baseline_map.exists():
        shutil.copy2(active_map, baseline_map)

    times = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    x0_np = np.asarray(np.load(assets / "traj_main.npy", allow_pickle=False)[0], dtype=np.float32)
    rna = ad.read_h5ad(data_dir / "rna.h5ad")
    protein = ad.read_h5ad(data_dir / "protein.h5ad")
    target_itga2b = _observed_target(rna, "ITGA2B", times)
    target_cd41 = _observed_target(protein, "CD41", times)
    primary_column_cpu, primary_mean_cpu, _ = _feature_bundle(rna, "ITGA2B")
    secondary_column_cpu, secondary_mean_cpu, secondary_index = _feature_bundle(protein, "CD41")
    secondary_reconstruction_cpu, secondary_input_mean_cpu, secondary_names = _all_feature_bundle(protein)
    retention_indices = [index for index, name in enumerate(secondary_names) if index != secondary_index]
    retention_reconstruction_cpu = secondary_reconstruction_cpu[:, retention_indices]
    primary_reconstruction_cpu, primary_input_mean_cpu, primary_names = _all_feature_bundle(rna)
    primary_retention_indices = [index for index, name in enumerate(primary_names) if name != "ITGA2B"]
    primary_retention_reconstruction_cpu = primary_reconstruction_cpu[:, primary_retention_indices]

    print(
        f"[velocity-tmap-finetune] device={device} x0={x0_np.shape} "
        f"target_ITGA2B={target_itga2b.tolist()} target_CD41={target_cd41.tolist()}",
        flush=True,
    )

    base_dyn_adata = ad.read_h5ad(baseline_dyn)
    base_dyn = _load_dynamics(base_dyn_adata, device).eval()
    candidate_dyn_model = _load_dynamics(base_dyn_adata, device).eval()
    base_map = _load_fixed_map(baseline_map, device)
    candidate_map_model = _load_fixed_map(baseline_map, device)
    for parameter in candidate_dyn_model.parameters():
        parameter.requires_grad = False
    for parameter in candidate_map_model.parameters():
        parameter.requires_grad = False
    for parameter in candidate_dyn_model.velocity_net.output_layer.parameters():
        parameter.requires_grad = True
    for parameter in candidate_map_model.decoder2.final_linear.parameters():
        parameter.requires_grad = True

    primary_column = primary_column_cpu.to(device=device, dtype=torch.float32)
    primary_mean = primary_mean_cpu.to(device=device, dtype=torch.float32)
    secondary_column = secondary_column_cpu.to(device=device, dtype=torch.float32)
    secondary_mean = secondary_mean_cpu.to(device=device, dtype=torch.float32)
    retention_reconstruction = retention_reconstruction_cpu.to(device=device, dtype=torch.float32)
    primary_retention_reconstruction = primary_retention_reconstruction_cpu.to(device=device, dtype=torch.float32)
    x0_all = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
    rng = np.random.default_rng(int(args.seed))
    positions = np.arange(x0_np.shape[0], dtype=np.int64)
    rng.shuffle(positions)
    train_pos = np.sort(positions[: min(int(args.train_cells), positions.size)])
    eval_pos = np.sort(positions[: min(int(args.eval_cells), positions.size)])
    x_train = x0_all.index_select(0, torch.as_tensor(train_pos, dtype=torch.long, device=device))
    x_eval = x0_all.index_select(0, torch.as_tensor(eval_pos, dtype=torch.long, device=device))

    with torch.no_grad():
        baseline_full_map0 = base_map.forward_single(x0_all).detach()
        baseline_train_traj, baseline_train_log_weights = _integrate_with_weights(
            base_dyn, x_train, times, dt=float(args.dt)
        )
        baseline_train_weights = _normalized_weights(baseline_train_log_weights)
        baseline_train_map = _map_latent(base_map, baseline_train_traj, batch_size=int(args.batch_size))
        baseline_train_primary_features = baseline_train_traj @ primary_retention_reconstruction
        baseline_eval_traj, baseline_eval_map, baseline_primary, baseline_secondary = _evaluate(
            base_dyn,
            base_map,
            x_eval,
            times,
            dt=float(args.dt),
            primary_column=primary_column,
            primary_mean=primary_mean,
            secondary_column=secondary_column,
            secondary_mean=secondary_mean,
            batch_size=int(args.batch_size),
        )
        baseline_velocity = base_dyn.velocity_net(
            torch.cat([x_eval, torch.zeros(x_eval.shape[0], 1, device=device)], dim=1)
        )
        baseline_train_pair = base_map.forward_single(x_train).detach()
    target_primary_t = torch.as_tensor(target_itga2b, dtype=torch.float32, device=device)
    target_secondary_t = torch.as_tensor(target_cd41, dtype=torch.float32, device=device)

    trainable_velocity = list(candidate_dyn_model.velocity_net.output_layer.parameters())
    trainable_tmap = list(candidate_map_model.decoder2.final_linear.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_velocity, "lr": float(args.learning_rate_velocity)},
            {"params": trainable_tmap, "lr": float(args.learning_rate_tmap)},
        ],
        weight_decay=1e-6,
    )

    def objective(
        xb: torch.Tensor,
        baseline_traj: torch.Tensor,
        baseline_map: torch.Tensor,
    ):
        candidate_traj, candidate_log_weights = _integrate_with_weights(
            candidate_dyn_model, xb, times, dt=float(args.dt)
        )
        candidate_map = _map_latent(candidate_map_model, candidate_traj, batch_size=int(args.batch_size))
        candidate_weights = _normalized_weights(candidate_log_weights)
        primary_values = candidate_traj @ primary_column + primary_mean
        secondary_values = candidate_map @ secondary_column + secondary_mean
        primary_curve = (primary_values * candidate_weights).sum(dim=1)
        secondary_curve = (secondary_values * candidate_weights).sum(dim=1)
        curve_loss = (
            float(args.primary_weight) * torch.mean((primary_curve - target_primary_t) ** 2)
            + float(args.secondary_weight) * torch.mean((secondary_curve - target_secondary_t) ** 2)
        )
        velocity_input = torch.cat([xb, torch.zeros(xb.shape[0], 1, device=device)], dim=1)
        baseline_v = base_dyn.velocity_net(velocity_input).detach()
        candidate_v = candidate_dyn_model.velocity_net(velocity_input)
        velocity_retention = torch.mean((candidate_v - baseline_v) ** 2)
        trajectory_retention = torch.mean((candidate_traj - baseline_traj) ** 2)
        baseline_primary_features = baseline_traj @ primary_retention_reconstruction
        candidate_primary_features = candidate_traj @ primary_retention_reconstruction
        primary_non_target_retention = torch.mean(
            (candidate_primary_features - baseline_primary_features) ** 2
        )
        map_retention = torch.mean((candidate_map - baseline_map) ** 2)
        baseline_non_target = baseline_map @ retention_reconstruction
        candidate_non_target = candidate_map @ retention_reconstruction
        non_target_retention = torch.mean((candidate_non_target - baseline_non_target) ** 2)
        candidate_map0 = candidate_map_model.forward_single(x0_all)
        candidate_secondary0 = (candidate_map0 @ secondary_column + secondary_mean).mean()
        baseline_non_target0 = baseline_full_map0 @ retention_reconstruction
        candidate_non_target0 = candidate_map0 @ retention_reconstruction
        t0_secondary_loss = (candidate_secondary0 - target_secondary_t[0]) ** 2
        t0_non_target_retention = torch.mean((candidate_non_target0 - baseline_non_target0) ** 2)
        paired_prediction = candidate_map_model.forward_single(xb)
        paired_baseline = base_map.forward_single(xb).detach()
        paired_loss = torch.mean((paired_prediction - paired_baseline) ** 2)
        loss = (
            curve_loss
            + float(args.velocity_retention_weight) * velocity_retention
            + float(args.trajectory_retention_weight) * trajectory_retention
            + float(args.primary_non_target_retention_weight) * primary_non_target_retention
            + float(args.map_retention_weight) * map_retention
            + float(args.non_target_retention_weight) * non_target_retention
            + float(args.secondary_weight) * t0_secondary_loss
            + float(args.non_target_retention_weight) * t0_non_target_retention
            + float(args.paired_weight) * paired_loss
        )
        return loss, {
            "curve_loss": curve_loss,
            "velocity_retention": velocity_retention,
            "trajectory_retention": trajectory_retention,
            "primary_non_target_retention": primary_non_target_retention,
            "map_retention": map_retention,
            "non_target_retention": non_target_retention,
            "t0_secondary_loss": t0_secondary_loss,
            "t0_non_target_retention": t0_non_target_retention,
            "paired_loss": paired_loss,
            "primary_curve": primary_curve,
            "secondary_curve": secondary_curve,
            "candidate_traj": candidate_traj,
            "candidate_map": candidate_map,
        }

    best_state_dyn = copy.deepcopy({key: value.detach().cpu() for key, value in candidate_dyn_model.state_dict().items()})
    best_state_map = copy.deepcopy({key: value.detach().cpu() for key, value in candidate_map_model.state_dict().items()})
    best_score = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        take = min(int(args.batch_size), x_train.shape[0])
        batch_pos = rng.choice(x_train.shape[0], size=take, replace=False)
        batch_idx = torch.as_tensor(batch_pos, dtype=torch.long, device=device)
        xb = x_train.index_select(0, batch_idx)
        baseline_batch_traj = baseline_train_traj[:, batch_pos]
        baseline_batch_map = baseline_train_map[:, batch_pos]
        loss, details = objective(xb, baseline_batch_traj, baseline_batch_map)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_velocity + trainable_tmap, 1.0)
        optimizer.step()

        if epoch == 1 or epoch % max(1, int(args.eval_every)) == 0 or epoch == int(args.epochs):
            _, _, eval_primary, eval_secondary = _evaluate(
                candidate_dyn_model,
                candidate_map_model,
                x_eval,
                times,
                dt=float(args.dt),
                primary_column=primary_column,
                primary_mean=primary_mean,
                secondary_column=secondary_column,
                secondary_mean=secondary_mean,
                batch_size=int(args.batch_size),
            )
            eval_primary_np = eval_primary.cpu().numpy()
            eval_secondary_np = eval_secondary.cpu().numpy()
            score = float(
                np.mean((eval_primary_np - target_itga2b) ** 2)
                + np.mean((eval_secondary_np - target_cd41) ** 2)
            )
            if score < best_score - 1e-8:
                best_score = score
                best_state_dyn = copy.deepcopy({key: value.detach().cpu() for key, value in candidate_dyn_model.state_dict().items()})
                best_state_map = copy.deepcopy({key: value.detach().cpu() for key, value in candidate_map_model.state_dict().items()})
                stale = 0
            else:
                stale += 1
            record = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "curve_loss": float(details["curve_loss"].detach().cpu()),
                "velocity_retention": float(details["velocity_retention"].detach().cpu()),
                "trajectory_retention": float(details["trajectory_retention"].detach().cpu()),
                "primary_non_target_retention": float(details["primary_non_target_retention"].detach().cpu()),
                "map_retention": float(details["map_retention"].detach().cpu()),
                "non_target_retention": float(details["non_target_retention"].detach().cpu()),
                "t0_secondary_loss": float(details["t0_secondary_loss"].detach().cpu()),
                "t0_non_target_retention": float(details["t0_non_target_retention"].detach().cpu()),
                "paired_loss": float(details["paired_loss"].detach().cpu()),
                "eval_primary_rmse": float(np.sqrt(np.mean((eval_primary_np - target_itga2b) ** 2))),
                "eval_secondary_rmse": float(np.sqrt(np.mean((eval_secondary_np - target_cd41) ** 2))),
            }
            history.append(record)
            print(
                f"epoch={epoch:03d} primary_rmse={record['eval_primary_rmse']:.6f} "
                f"secondary_rmse={record['eval_secondary_rmse']:.6f} "
                f"drift={record['trajectory_retention']:.6f}",
                flush=True,
            )
            if stale >= int(args.patience):
                break

    candidate_dyn_model.load_state_dict(best_state_dyn, strict=True)
    candidate_map_model.load_state_dict(best_state_map, strict=True)
    candidate_dyn_model.eval()
    candidate_map_model.eval()
    candidate_eval_traj, candidate_eval_map, candidate_primary, candidate_secondary = _evaluate(
        candidate_dyn_model,
        candidate_map_model,
        x_eval,
        times,
        dt=float(args.dt),
        primary_column=primary_column,
        primary_mean=primary_mean,
        secondary_column=secondary_column,
        secondary_mean=secondary_mean,
        batch_size=int(args.batch_size),
    )
    with torch.no_grad():
        candidate_velocity = candidate_dyn_model.velocity_net(
            torch.cat([x_eval, torch.zeros(x_eval.shape[0], 1, device=device)], dim=1)
        )
        baseline_non_target = baseline_eval_map @ retention_reconstruction
        candidate_non_target = candidate_eval_map @ retention_reconstruction
        baseline_pair = base_map.forward_single(x_eval)
        candidate_pair = candidate_map_model.forward_single(x_eval)
    velocity_relative_change = float(
        torch.linalg.vector_norm(candidate_velocity - baseline_velocity)
        / torch.clamp(torch.linalg.vector_norm(baseline_velocity), min=1e-8)
    )
    trajectory_relative_change = float(
        torch.linalg.vector_norm(candidate_eval_traj - baseline_eval_traj)
        / torch.clamp(torch.linalg.vector_norm(baseline_eval_traj), min=1e-8)
    )
    map_relative_change = float(
        torch.linalg.vector_norm(candidate_eval_map - baseline_eval_map)
        / torch.clamp(torch.linalg.vector_norm(baseline_eval_map), min=1e-8)
    )
    non_target_relative_change = float(
        torch.linalg.vector_norm(candidate_non_target - baseline_non_target)
        / torch.clamp(torch.linalg.vector_norm(baseline_non_target), min=1e-8)
    )
    paired_relative_change = float(
        torch.linalg.vector_norm(candidate_pair - baseline_pair)
        / torch.clamp(torch.linalg.vector_norm(baseline_pair), min=1e-8)
    )
    candidate_primary_np = candidate_primary.cpu().numpy()
    candidate_secondary_np = candidate_secondary.cpu().numpy()
    baseline_primary_np = baseline_primary.cpu().numpy()
    baseline_secondary_np = baseline_secondary.cpu().numpy()
    growth_unchanged = all(
        torch.equal(value, base_dyn.growth_net.state_dict()[key])
        for key, value in candidate_dyn_model.growth_net.state_dict().items()
    )
    map_non_decoder_unchanged = all(
        torch.equal(value, base_map.state_dict()[key])
        for key, value in candidate_map_model.state_dict().items()
        if not key.startswith("decoder2.final_linear.")
    )
    audit = {
        "method": "joint velocity and RNA-to-protein tmap fine-tuning for HSPC z=0 ITGA2B/CD41 reconstruction",
        "dataset": "HSPC 31800",
        "trainable_modules": ["velocity_net.output_layer", "decoder2.final_linear"],
        "frozen_modules": ["velocity hidden layers", "growth_net", "tmap encoders/decoder layers", "protein PCA reconstruction"],
        "posthoc_curve_adjustment": False,
        "device": str(device),
        "times": times.tolist(),
        "target_itga2b": target_itga2b.tolist(),
        "target_cd41": target_cd41.tolist(),
        "baseline_curve": {"ITGA2B": baseline_primary_np.tolist(), "CD41": baseline_secondary_np.tolist()},
        "candidate_curve": {"ITGA2B": candidate_primary_np.tolist(), "CD41": candidate_secondary_np.tolist()},
        "baseline_metrics": {
            "ITGA2B": _curve_metrics(baseline_primary_np, target_itga2b),
            "CD41": _curve_metrics(baseline_secondary_np, target_cd41),
        },
        "candidate_metrics": {
            "ITGA2B": _curve_metrics(candidate_primary_np, target_itga2b),
            "CD41": _curve_metrics(candidate_secondary_np, target_cd41),
        },
        "retention_metrics": {
            "velocity_relative_change": velocity_relative_change,
            "trajectory_relative_change": trajectory_relative_change,
            "map_relative_change": map_relative_change,
            "non_target_protein_relative_change": non_target_relative_change,
            "paired_map_relative_change": paired_relative_change,
        },
        "growth_unchanged": bool(growth_unchanged),
        "map_non_decoder_unchanged": bool(map_non_decoder_unchanged),
        "train_cells": int(train_pos.size),
        "eval_cells": int(eval_pos.size),
        "history": history,
        "baseline_dynamics_path": str(baseline_dyn.relative_to(ROOT)),
        "baseline_map_path": str(baseline_map.relative_to(ROOT)),
        "candidate_dynamics_path": str(candidate_dyn.relative_to(ROOT)),
        "candidate_map_path": str(candidate_map.relative_to(ROOT)),
    }

    dyn_out = ad.read_h5ad(baseline_dyn)
    dyn_out.uns["all_model"]["model_state_dict"] = _state_to_numpy(candidate_dyn_model)
    dyn_out.uns["all_model"]["velocity_tmap_finetune"] = {
        "method": audit["method"],
        "trainable_modules": audit["trainable_modules"],
        "target_features": ["ITGA2B", "CD41"],
        "audit_path": str(audit_path.relative_to(ROOT)),
    }
    dyn_out.write_h5ad(candidate_dyn)
    original_checkpoint = torch.load(baseline_map, map_location="cpu", weights_only=False)
    candidate_checkpoint = dict(original_checkpoint)
    candidate_checkpoint["model_state_dict"] = {
        key: value.detach().cpu() for key, value in candidate_map_model.state_dict().items()
    }
    candidate_checkpoint["fine_tune_metadata"] = {
        "method": audit["method"],
        "target_features": ["ITGA2B", "CD41"],
        "trainable_modules": ["decoder2.final_linear"],
        "dynamics_model_changed": True,
        "posthoc_curve_adjustment": False,
        "seed": int(args.seed),
    }
    torch.save(candidate_checkpoint, candidate_map)
    audit["candidate_dynamics_sha256"] = _sha256(candidate_dyn)
    audit["candidate_map_sha256"] = _sha256(candidate_map)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    activated = False
    if bool(args.activate):
        shutil.copy2(candidate_dyn, active_dyn)
        shutil.copy2(candidate_map, active_map)
        activated = True
    audit["activated"] = activated
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "baseline": audit["baseline_metrics"],
                "candidate": audit["candidate_metrics"],
                "retention": audit["retention_metrics"],
                "activated": activated,
                "candidate_dynamics": str(candidate_dyn),
                "candidate_map": str(candidate_map),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
