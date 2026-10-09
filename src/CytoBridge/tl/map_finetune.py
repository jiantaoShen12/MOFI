"""Trajectory-guided fine-tuning for paired-modality latent maps.

The routine in this module keeps the learned dynamics fixed.  It integrates the
primary-modality initial cells with the released ODE, maps every integrated
state to the secondary latent space, and optimizes the selected secondary
feature after the stored PCA reconstruction.  A paired-latent loss and a
baseline-retention loss limit changes outside the requested trajectory readout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import anndata as ad
import numpy as np
import torch

from CytoBridge.Map.tl.models import SplicedAutoEncoder
from CytoBridge.tl.analysis_dense_time import _build_ode_trajectory, _load_pca_bundle
from CytoBridge.utils import load_model_from_adata


@dataclass(frozen=True)
class TrajectoryMapFineTuneResult:
    state_dict: Mapping[str, torch.Tensor]
    best_epoch: int
    history: list[dict[str, float]]
    audit: Dict[str, Any]


def _latent(adata: ad.AnnData) -> np.ndarray:
    values = adata.obsm["X_latent"] if "X_latent" in adata.obsm else adata.X
    return np.asarray(values, dtype=np.float32)


def _stratified_split(times: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    parts: list[list[int]] = [[], [], []]
    for value in np.sort(np.unique(times)):
        indices = np.flatnonzero(np.isclose(times, value))
        rng.shuffle(indices)
        n_test = max(1, int(round(0.10 * len(indices))))
        n_valid = max(1, int(round(0.10 * len(indices))))
        parts[2].extend(indices[:n_test].tolist())
        parts[1].extend(indices[n_test : n_test + n_valid].tolist())
        parts[0].extend(indices[n_test + n_valid :].tolist())
    return tuple(np.asarray(part, dtype=np.int64) for part in parts)  # type: ignore[return-value]


def _make_map_model(checkpoint: Mapping[str, Any], dim1: int, dim2: int, device: torch.device) -> SplicedAutoEncoder:
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    hidden = int(state["encoder1.encode2.weight"].shape[0])
    model = SplicedAutoEncoder(dim1, dim2, hidden_dim=hidden, flat_mode=True)
    model.load_state_dict(state, strict=True)
    return model.to(device)


@torch.no_grad()
def _predict_paired(
    model: SplicedAutoEncoder,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(x), int(batch_size)):
        xb = torch.as_tensor(x[start : start + int(batch_size)], dtype=torch.float32, device=device)
        chunks.append(model.forward_single(xb, in_domain=1, out_domain=2)[0].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _weighted_feature_curve(
    model: SplicedAutoEncoder,
    trajectory: torch.Tensor,
    weights: torch.Tensor,
    positions: np.ndarray,
    reconstruction_column: torch.Tensor,
    reconstruction_mean: torch.Tensor,
) -> torch.Tensor:
    pos = torch.as_tensor(positions, dtype=torch.long, device=trajectory.device)
    values = []
    for t_idx in range(int(trajectory.shape[0])):
        secondary_latent = model.forward_single(
            trajectory[t_idx].index_select(0, pos), in_domain=1, out_domain=2
        )[0]
        decoded = secondary_latent @ reconstruction_column + reconstruction_mean
        wt = torch.clamp(weights[t_idx].index_select(0, pos), min=0.0)
        denominator = torch.sum(wt)
        if bool((denominator <= 1e-12).detach().cpu()):
            values.append(torch.mean(decoded))
        else:
            values.append(torch.sum(decoded * wt) / denominator)
    return torch.stack(values)


def _curve_metrics(predicted: np.ndarray, observed: np.ndarray) -> Dict[str, float]:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    late = residual[1:] if residual.size > 1 else residual
    return {
        "rmse_all": float(np.sqrt(np.mean(residual**2))),
        "rmse_late": float(np.sqrt(np.mean(late**2))),
        "mse_all": float(np.mean(residual**2)),
        "mse_late": float(np.mean(late**2)),
    }


def fine_tune_primary_to_secondary_feature_trajectory(
    *,
    primary: ad.AnnData,
    secondary: ad.AnnData,
    dynamics: ad.AnnData,
    checkpoint: Mapping[str, Any],
    feature_name: str,
    time_key: str,
    times: Sequence[float],
    dt: float,
    device: str = "cpu",
    epochs: int = 160,
    learning_rate: float = 3e-4,
    curve_weight: float = 8.0,
    paired_weight: float = 0.10,
    retention_weight: float = 0.20,
    non_target_feature_weight: float = 0.0,
    retention_feature_names: Sequence[str] | None = None,
    batch_size: int = 4096,
    max_trajectory_cells_per_split: int | None = None,
    patience: int = 30,
    seed: int = 42,
    precomputed_trajectory: np.ndarray | None = None,
    precomputed_weights: np.ndarray | None = None,
    precomputed_times: Sequence[float] | None = None,
) -> TrajectoryMapFineTuneResult:
    """Fine-tune the protein decoder using an integrated feature trajectory.

    Only ``decoder2.final_linear`` is trainable.  The velocity/growth model,
    PCA reconstruction, RNA encoder and the rest of the paired map remain fixed.
    Target means and trajectory-source cells use deterministic train/validation/
    test partitions; test cells are never used for optimization or selection.
    """

    if primary.n_obs != secondary.n_obs or not np.array_equal(primary.obs_names, secondary.obs_names):
        raise ValueError("primary and secondary AnnData objects must be one-to-one paired")
    if str(time_key) not in primary.obs or str(time_key) not in secondary.obs:
        raise KeyError(f"time key not found in paired AnnData: {time_key}")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device_t = torch.device(device)
    print("[map-finetune] loading paired latent arrays", flush=True)
    x = _latent(primary)
    y = _latent(secondary)
    observed_times = np.asarray(primary.obs[str(time_key)], dtype=float)
    train_idx, valid_idx, test_idx = _stratified_split(observed_times, int(seed))

    print("[map-finetune] loading protein PCA reconstruction", flush=True)
    secondary_bundle = _load_pca_bundle(secondary, space_name="secondary_processed")
    names = [str(value) for value in secondary_bundle.var_names]
    if str(feature_name) not in names:
        raise KeyError(f"secondary feature not found: {feature_name}")
    feature_idx = names.index(str(feature_name))
    reconstruction_column = torch.as_tensor(
        np.asarray(secondary_bundle.reconstruction[:, feature_idx], dtype=np.float32),
        dtype=torch.float32,
        device=device_t,
    )
    reconstruction_mean = torch.tensor(
        float(np.asarray(secondary_bundle.mean, dtype=np.float32)[feature_idx]),
        dtype=torch.float32,
        device=device_t,
    )
    if retention_feature_names is None:
        retention_indices = [idx for idx, name in enumerate(names) if name != str(feature_name)]
    else:
        missing = [str(name) for name in retention_feature_names if str(name) not in names]
        if missing:
            raise KeyError(f"secondary retention features not found: {missing}")
        retention_indices = [names.index(str(name)) for name in retention_feature_names if str(name) != str(feature_name)]
    retention_reconstruction = torch.as_tensor(
        np.asarray(secondary_bundle.reconstruction[:, retention_indices], dtype=np.float32),
        dtype=torch.float32,
        device=device_t,
    )

    t_eval = np.asarray([float(value) for value in times], dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2 or np.any(np.diff(t_eval) <= 0):
        raise ValueError("times must be a strictly increasing one-dimensional sequence")
    t0 = float(np.min(observed_times))
    source_global = np.flatnonzero(np.isclose(observed_times, t0))
    source_position = {int(global_idx): pos for pos, global_idx in enumerate(source_global.tolist())}
    split_global = {"train": train_idx, "validation": valid_idx, "test": test_idx}
    source_splits = {
        name: np.asarray([source_position[int(i)] for i in indices if int(i) in source_position], dtype=np.int64)
        for name, indices in split_global.items()
    }
    if max_trajectory_cells_per_split is not None:
        split_rng = np.random.default_rng(int(seed) + 17)
        for split_name, positions in source_splits.items():
            if positions.size > int(max_trajectory_cells_per_split):
                source_splits[split_name] = np.sort(
                    split_rng.choice(
                        positions,
                        size=int(max_trajectory_cells_per_split),
                        replace=False,
                    )
                )

    integration_source = "fresh_original_library_integration"
    if precomputed_trajectory is None or precomputed_weights is None:
        print("[map-finetune] integrating time-0 RNA states with the released ODE", flush=True)
        dynamics_model = load_model_from_adata(dynamics)
        integrated = _build_ode_trajectory(
            model=dynamics_model,
            x0=x[source_global],
            times=t_eval,
            adata=dynamics,
            dt=float(dt),
            device=str(device_t),
        )
        trajectory_np = np.asarray(integrated["main_latent"], dtype=np.float32)
        weights_np = np.asarray(integrated["weights_abs"], dtype=np.float32)
    else:
        print("[map-finetune] validating original-library trajectory cache", flush=True)
        if precomputed_times is None:
            raise ValueError("precomputed_times is required with a precomputed trajectory")
        trajectory_full = np.asarray(precomputed_trajectory, dtype=np.float32)
        weights_full = np.asarray(precomputed_weights, dtype=np.float32)
        cached_times = np.asarray(precomputed_times, dtype=float)
        if trajectory_full.ndim != 3 or weights_full.shape[:2] != trajectory_full.shape[:2]:
            raise ValueError(
                "precomputed trajectory/weights must have shapes (time, cell, latent) and (time, cell)"
            )
        if trajectory_full.shape[1] != source_global.size:
            raise ValueError(
                f"precomputed trajectory has {trajectory_full.shape[1]} cells; expected {source_global.size}"
            )
        selected_time_indices = []
        for time_value in t_eval:
            matches = np.flatnonzero(np.isclose(cached_times, time_value))
            if matches.size == 0:
                raise ValueError(f"requested time {time_value} missing from precomputed trajectory")
            selected_time_indices.append(int(matches[0]))
        trajectory_np = trajectory_full[selected_time_indices]
        weights_np = weights_full[selected_time_indices]
        integration_source = "original_library_integration_cache"
    trajectory = torch.as_tensor(trajectory_np, dtype=torch.float32, device=device_t)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device_t)

    secondary_feature = (
        y @ np.asarray(secondary_bundle.reconstruction[:, feature_idx], dtype=np.float32)
        + float(np.asarray(secondary_bundle.mean, dtype=np.float32)[feature_idx])
    )
    target_by_split: Dict[str, np.ndarray] = {}
    for split_name, indices in split_global.items():
        target_by_split[split_name] = np.asarray(
            [
                float(np.mean(secondary_feature[indices[np.isclose(observed_times[indices], time_value)]]))
                for time_value in t_eval
            ],
            dtype=np.float32,
        )
    target_by_split["all"] = np.asarray(
        [float(np.mean(secondary_feature[np.isclose(observed_times, time_value)])) for time_value in t_eval],
        dtype=np.float32,
    )

    print("[map-finetune] building baseline and candidate map models", flush=True)
    base = _make_map_model(checkpoint, x.shape[1], y.shape[1], device_t).eval()
    model = _make_map_model(checkpoint, x.shape[1], y.shape[1], device_t).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.decoder2.final_linear.parameters():
        parameter.requires_grad = True

    print("[map-finetune] computing baseline curve and retention targets", flush=True)
    with torch.no_grad():
        base_curve_by_split = {
            name: _weighted_feature_curve(
                base,
                trajectory,
                weights,
                positions,
                reconstruction_column,
                reconstruction_mean,
            ).cpu().numpy()
            for name, positions in source_splits.items()
        }
        base_curve_by_split["all"] = _weighted_feature_curve(
            base,
            trajectory,
            weights,
            np.arange(source_global.size, dtype=np.int64),
            reconstruction_column,
            reconstruction_mean,
        ).cpu().numpy()
        base_train_latent = base.forward_single(
            trajectory[:, source_splits["train"]].reshape(-1, trajectory.shape[-1]),
            in_domain=1,
            out_domain=2,
        )[0].detach()
        base_train_features = (base_train_latent @ retention_reconstruction).detach()
        base_valid_latent = base.forward_single(
            trajectory[:, source_splits["validation"]].reshape(-1, trajectory.shape[-1]),
            in_domain=1,
            out_domain=2,
        )[0].detach()
        base_valid_features = (base_valid_latent @ retention_reconstruction).detach()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=1e-5,
    )
    curve_time_weights = torch.as_tensor(
        np.linspace(0.35, 1.75, t_eval.size, dtype=np.float32),
        dtype=torch.float32,
        device=device_t,
    )
    target_train = torch.as_tensor(target_by_split["train"], dtype=torch.float32, device=device_t)
    target_valid = torch.as_tensor(target_by_split["validation"], dtype=torch.float32, device=device_t)
    rng = np.random.default_rng(int(seed))
    best_score = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []

    print("[map-finetune] optimizing protein decoder", flush=True)
    for epoch in range(1, int(epochs) + 1):
        model.eval()
        curve_train = _weighted_feature_curve(
            model,
            trajectory,
            weights,
            source_splits["train"],
            reconstruction_column,
            reconstruction_mean,
        )
        curve_loss = torch.mean(curve_time_weights * (curve_train - target_train) ** 2)

        take = min(int(batch_size), int(train_idx.size))
        paired_indices = rng.choice(train_idx, size=take, replace=False)
        paired_x = torch.as_tensor(x[paired_indices], dtype=torch.float32, device=device_t)
        paired_y = torch.as_tensor(y[paired_indices], dtype=torch.float32, device=device_t)
        paired_prediction = model.forward_single(paired_x, in_domain=1, out_domain=2)[0]
        paired_loss = torch.mean((paired_prediction - paired_y) ** 2)

        trajectory_prediction = model.forward_single(
            trajectory[:, source_splits["train"]].reshape(-1, trajectory.shape[-1]),
            in_domain=1,
            out_domain=2,
        )[0]
        retention_loss = torch.mean((trajectory_prediction - base_train_latent) ** 2)
        non_target_feature_loss = torch.mean(
            (trajectory_prediction @ retention_reconstruction - base_train_features) ** 2
        )
        loss = (
            float(curve_weight) * curve_loss
            + float(paired_weight) * paired_loss
            + float(retention_weight) * retention_loss
            + float(non_target_feature_weight) * non_target_feature_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 2.0
        )
        optimizer.step()

        with torch.no_grad():
            curve_valid = _weighted_feature_curve(
                model,
                trajectory,
                weights,
                source_splits["validation"],
                reconstruction_column,
                reconstruction_mean,
            )
            valid_curve_mse = torch.mean((curve_valid[1:] - target_valid[1:]) ** 2)
            valid_prediction = model.forward_single(
                torch.as_tensor(x[valid_idx], dtype=torch.float32, device=device_t),
                in_domain=1,
                out_domain=2,
            )[0]
            valid_paired_mse = torch.mean(
                (valid_prediction - torch.as_tensor(y[valid_idx], dtype=torch.float32, device=device_t)) ** 2
            )
            valid_trajectory_prediction = model.forward_single(
                trajectory[:, source_splits["validation"]].reshape(-1, trajectory.shape[-1]),
                in_domain=1,
                out_domain=2,
            )[0]
            valid_non_target_feature_mse = torch.mean(
                (valid_trajectory_prediction @ retention_reconstruction - base_valid_features) ** 2
            )
            score = float(
                (
                    valid_curve_mse
                    + 0.01 * valid_paired_mse
                    + float(non_target_feature_weight) * valid_non_target_feature_mse
                ).cpu()
            )
        record = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "curve_loss": float(curve_loss.detach().cpu()),
            "paired_loss": float(paired_loss.detach().cpu()),
            "retention_loss": float(retention_loss.detach().cpu()),
            "non_target_feature_loss": float(non_target_feature_loss.detach().cpu()),
            "validation_curve_mse_late": float(valid_curve_mse.cpu()),
            "validation_paired_mse": float(valid_paired_mse.cpu()),
            "validation_non_target_feature_mse": float(valid_non_target_feature_mse.cpu()),
        }
        history.append(record)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} curve={record['curve_loss']:.6f} "
                f"val_curve_late={record['validation_curve_mse_late']:.6f} "
                f"paired={record['validation_paired_mse']:.6f}",
                flush=True,
            )
        if score < best_score - 1e-7:
            best_score = score
            best_epoch = int(epoch)
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("trajectory-guided fine-tuning did not produce a candidate")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        candidate_curve_by_split = {
            name: _weighted_feature_curve(
                model,
                trajectory,
                weights,
                positions,
                reconstruction_column,
                reconstruction_mean,
            ).cpu().numpy()
            for name, positions in source_splits.items()
        }
        candidate_curve_by_split["all"] = _weighted_feature_curve(
            model,
            trajectory,
            weights,
            np.arange(source_global.size, dtype=np.int64),
            reconstruction_column,
            reconstruction_mean,
        ).cpu().numpy()

    baseline_test_prediction = _predict_paired(base, x[test_idx], device=device_t, batch_size=batch_size)
    candidate_test_prediction = _predict_paired(model, x[test_idx], device=device_t, batch_size=batch_size)
    baseline_paired_mse = float(np.mean((baseline_test_prediction - y[test_idx]) ** 2))
    candidate_paired_mse = float(np.mean((candidate_test_prediction - y[test_idx]) ** 2))
    curve_audit = {}
    for split_name in ("train", "validation", "test", "all"):
        curve_audit[split_name] = {
            "observed": [float(value) for value in target_by_split[split_name]],
            "baseline": [float(value) for value in base_curve_by_split[split_name]],
            "candidate": [float(value) for value in candidate_curve_by_split[split_name]],
            "baseline_metrics": _curve_metrics(base_curve_by_split[split_name], target_by_split[split_name]),
            "candidate_metrics": _curve_metrics(candidate_curve_by_split[split_name], target_by_split[split_name]),
        }
    late_test_gain = 1.0 - (
        curve_audit["test"]["candidate_metrics"]["rmse_late"]
        / curve_audit["test"]["baseline_metrics"]["rmse_late"]
    )
    paired_change = candidate_paired_mse / baseline_paired_mse - 1.0
    audit = {
        "method": "trajectory-guided RNA-to-protein map fine-tuning",
        "feature": str(feature_name),
        "times": [float(value) for value in t_eval],
        "trainable_modules": ["decoder2.final_linear"],
        "retention_features": [names[idx] for idx in retention_indices],
        "non_target_feature_weight": float(non_target_feature_weight),
        "max_trajectory_cells_per_split": (
            None if max_trajectory_cells_per_split is None else int(max_trajectory_cells_per_split)
        ),
        "dynamics_model_changed": False,
        "pca_reconstruction_changed": False,
        "posthoc_curve_adjustment": False,
        "integration_source": integration_source,
        "split_counts": {
            name: {
                "paired_cells": int(indices.size),
                "time0_trajectory_cells": int(source_splits[name].size),
            }
            for name, indices in split_global.items()
        },
        "curve_by_split": curve_audit,
        "heldout_paired_latent_mse": {
            "baseline": baseline_paired_mse,
            "candidate": candidate_paired_mse,
            "relative_change": float(paired_change),
        },
        "heldout_late_curve_rmse_gain_fraction": float(late_test_gain),
        "best_epoch": int(best_epoch),
        "history": history,
    }
    return TrajectoryMapFineTuneResult(
        state_dict=best_state,
        best_epoch=int(best_epoch),
        history=history,
        audit=audit,
    )
