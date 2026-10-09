"""Hold-out training for aligned multi-omics time series.

This module keeps missing targets out of the loss while preserving a union
time grid when either modality remains observed. Normalized time 0 is the
required initial condition; times 1--3 are interpolation tests and time 4 can
be held out as an endpoint-extrapolation test.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse

from CytoBridge.tl.losses import calc_mass_loss, calc_ot_loss
from CytoBridge.tl.methods import neural_ode_step
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.trainer import TrainingPipeline
from CytoBridge.utils.config import load_config
from CytoBridge.utils.utils import sample, sample_sec


class HoldoutTrainingPipeline(TrainingPipeline):
    """Neural-ODE trainer that accepts per-time missing modality targets."""

    def _prepare_df(self, data):
        frames = []
        for time_idx, values in enumerate(data):
            if values is None:
                continue
            array = values.detach().cpu().numpy()
            frames.append(
                pd.DataFrame(
                    {
                        "x1": array[:, 0],
                        "x2": array[:, 1],
                        "samples": np.full(array.shape[0], time_idx, dtype=float),
                    }
                )
            )
        if not frames:
            raise ValueError("Hold-out training requires primary observations.")
        return pd.concat(frames, ignore_index=True)

    def train_neural_ode_epoch(
        self,
        stage_params,
        data,
        time_points,
        ode_func,
        data_sec_torch=None,
    ):
        if stage_params.get("use_density_loss", False):
            raise ValueError("Density loss is not implemented for missing primary targets.")
        if stage_params.get("use_pinn_loss", False):
            raise ValueError("PINN loss is not implemented for hold-out training.")
        if stage_params.get("use_cycle_OT", False):
            raise ValueError("Cycle OT is not implemented for hold-out training.")
        if data[0] is None:
            raise ValueError("The initial primary time point cannot be held out.")

        lambda_ot = float(stage_params["lambda_ot"])
        lambda_mass = float(stage_params["lambda_mass"])
        lambda_energy = float(stage_params["lambda_energy"])
        lambda_ot_sec = float(stage_params.get("lambda_ot_sec", 0.0))
        lambda_cross_map = float(stage_params.get("lambda_cross_map", 0.0))
        ot_loss_type = stage_params["OT_loss"]
        global_mass = bool(stage_params.get("global_mass", False))

        x0 = sample(data[0], self.batch_size).to(self.device)
        lnw0 = torch.log(
            torch.ones(self.batch_size, 1, device=self.device) / self.batch_size
        )
        mass_0 = int(data[0].shape[0])
        total_loss = 0.0

        for idx in range(1, len(time_points)):
            self.optimizer.zero_grad()
            t0, t1 = time_points[idx - 1], time_points[idx]
            primary_raw = data[idx]
            secondary_raw = (
                data_sec_torch[idx] if data_sec_torch is not None else None
            )
            has_primary = primary_raw is not None
            has_secondary = secondary_raw is not None and lambda_ot_sec > 0
            if not has_primary and not has_secondary:
                raise ValueError(f"No modality target remains at time {t1}.")

            primary_target = None
            secondary_target = None
            if (
                has_primary
                and has_secondary
                and primary_raw.shape[0] == secondary_raw.shape[0]
            ):
                primary_target, secondary_target = sample_sec(
                    primary_raw, secondary_raw, self.batch_size
                )
                primary_target = primary_target.to(self.device)
                secondary_target = secondary_target.to(self.device)
            else:
                if has_primary:
                    primary_target = sample(
                        primary_raw, self.batch_size
                    ).to(self.device)
                if has_secondary:
                    secondary_target = sample(
                        secondary_raw, self.batch_size
                    ).to(self.device)

            target_count = int(
                primary_raw.shape[0] if has_primary else secondary_raw.shape[0]
            )
            relative_mass = target_count / mass_0
            x1, lnw1, e1 = neural_ode_step(
                ode_func, x0, lnw0, t0, t1, self.device
            )

            loss_primary = torch.zeros((), dtype=x1.dtype, device=x1.device)
            loss_secondary = torch.zeros((), dtype=x1.dtype, device=x1.device)
            if has_primary:
                loss_primary = calc_ot_loss(
                    x1, primary_target, lnw1, ot_loss_type
                )
            if has_secondary:
                mapped = self._map_to_secondary(x1)
                loss_secondary = calc_ot_loss(
                    mapped, secondary_target, lnw1, ot_loss_type
                )

            mass_target = primary_target
            if mass_target is None:
                if secondary_target.shape[1] < x1.shape[1]:
                    raise ValueError(
                        "Secondary-only target cannot define primary local mass."
                    )
                mass_target = secondary_target[:, : x1.shape[1]]
            loss_mass = (
                calc_mass_loss(
                    x1, mass_target, lnw1, relative_mass, global_mass
                )
                if self.use_mass
                else torch.zeros((), dtype=x1.dtype, device=x1.device)
            )
            loss_energy = e1.mean()
            loss = (
                lambda_ot * loss_primary
                + lambda_ot_sec * loss_secondary
                + lambda_mass * loss_mass
                + lambda_energy * loss_energy
            )

            if (
                self.update_transport
                and has_primary
                and has_secondary
                and lambda_cross_map > 0
            ):
                mapped_target = self._map_to_secondary(primary_target)
                cross_loss = torch.nn.functional.mse_loss(
                    mapped_target, secondary_target
                )
                loss = loss + lambda_cross_map * cross_loss

            loss.backward()
            self.optimizer.step()
            x0 = x1.detach().clone()
            lnw0 = lnw1.detach().clone()
            total_loss += float(loss.item())

        return total_loss / max(len(time_points) - 1, 1)


def _float_set(values: Iterable[float] | None) -> set[float]:
    if values is None:
        return set()
    return {float(value) for value in values}


def _latent_tensor(adata, time_key: str, time_value: float, device):
    mask = np.isclose(
        adata.obs[time_key].astype(float).to_numpy(), float(time_value)
    )
    subset = adata[mask]
    latent = subset.obsm["X_latent"]
    if issparse(latent):
        latent = latent.toarray()
    return torch.as_tensor(latent, dtype=torch.float32, device=device)


def fit_holdout(
    adata,
    config: Dict[str, Any] | str,
    *,
    adata_sec=None,
    hold_out_primary: Iterable[float] | None = None,
    hold_out_secondary: Iterable[float] | None = None,
    batch_size: int | None = None,
    device: str = "cuda",
    progress_callback=None,
):
    """Train Sync-UOT with only explicitly observed modality/time targets."""
    resolved = load_config(config)
    device_obj = torch.device(device)
    time_key = "time_point_processed"

    primary_all = _float_set(adata.obs[time_key].unique())
    secondary_all = (
        _float_set(adata_sec.obs[time_key].unique())
        if adata_sec is not None
        else set()
    )
    held_primary = _float_set(hold_out_primary)
    held_secondary = _float_set(hold_out_secondary)
    # Hold-out support is data-driven.  The previous fixed set {1,2,3,4}
    # silently rejected valid interior times in real datasets such as HO
    # (whose normalized grid extends to t=9).  Keep t=0 reserved as the fixed
    # initial condition, while allowing any non-initial time present in the
    # supplied primary/secondary data.  The caller still enforces that the
    # selected time is interior (not the final endpoint).
    allowed_holdout_times = (primary_all | secondary_all) - {0.0}
    invalid = (held_primary | held_secondary) - allowed_holdout_times
    if invalid:
        raise ValueError(
            "Only normalized times 1, 2, 3, 4 may be held out; "
            f"time 0 is the fixed initial condition: {invalid}"
        )

    observed_primary = primary_all - held_primary
    observed_secondary = secondary_all - held_secondary
    time_points = sorted(observed_primary | observed_secondary)
    if not time_points or time_points[0] not in observed_primary:
        raise ValueError("Initial primary time 0 must remain observed.")

    primary_tensors = [
        _latent_tensor(adata, time_key, t, device_obj)
        if t in observed_primary
        else None
        for t in time_points
    ]
    secondary_tensors = [
        _latent_tensor(adata_sec, time_key, t, device_obj)
        if adata_sec is not None and t in observed_secondary
        else None
        for t in time_points
    ]
    available = [
        values
        for values in primary_tensors + secondary_tensors
        if values is not None
    ]
    if batch_size is None:
        batch_size = min(min(values.shape[0] for values in available), 256)
    if any(values.shape[0] < batch_size for values in available):
        raise ValueError("batch_size exceeds an observed modality/time sample count.")

    first_primary = next(values for values in primary_tensors if values is not None)
    model = DynamicalModel(first_primary.shape[1], resolved["model"])
    trainer = HoldoutTrainingPipeline(
        model,
        resolved,
        batch_size,
        device_obj,
        data=primary_tensors,
        progress_callback=progress_callback,
    )
    secondary_arg = (
        secondary_tensors
        if any(values is not None for values in secondary_tensors)
        else None
    )
    model = trainer.train(
        primary_tensors,
        time_points,
        data_sec_torch=secondary_arg,
    )

    result = adata.copy()
    all_times = torch.as_tensor(
        result.obs[time_key].to_numpy(),
        dtype=torch.float32,
        device=device_obj,
    ).unsqueeze(1)
    all_data = torch.as_tensor(
        result.obsm["X_latent"], dtype=torch.float32, device=device_obj
    )
    net_input = torch.cat([all_data, all_times], dim=1)
    result.obsm["velocity_latent"] = (
        model.velocity_net(net_input).detach().cpu().numpy()
    )
    if "growth" in model.components:
        result.obsm["growth_rate"] = (
            model.growth_net(net_input).detach().cpu().numpy()
        )

    protocol = {
        "allowed_holdout_times": sorted(allowed_holdout_times),
        "held_primary": sorted(held_primary),
        "held_secondary": sorted(held_secondary),
        "removed_primary_observations": sorted(held_primary & primary_all),
        "removed_secondary_observations": sorted(held_secondary & secondary_all),
        "secondary_evaluation_only_times": sorted(held_secondary - secondary_all),
        "observed_primary": sorted(observed_primary),
        "observed_secondary": sorted(observed_secondary),
        "training_time_grid": time_points,
        "task_kind": (
            "endpoint_extrapolation"
            if 4.0 in (held_primary | held_secondary)
            and 4.0 not in (observed_primary | observed_secondary)
            else "interpolation_or_cross_modal_reconstruction"
        ),
    }
    result.uns["holdout_protocol"] = protocol
    result.uns["all_model"] = {
        "model_config": resolved["model"],
        "training_config": {
            "defaults": resolved["training"]["defaults"],
            "plan": json.dumps(resolved["training"]["plan"]),
        },
        "model_state_dict": {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        },
    }

    output_dir = pathlib.Path(resolved["ckpt_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_dir / "adata.h5ad")
    (output_dir / "holdout_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    return result
