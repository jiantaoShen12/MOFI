"""Load a MOFI dynamics model without loading a fitted AnnData object.

The model bundle consists of JSON architecture metadata and an NPZ archive of
numeric state-dictionary arrays. It contains neither observations nor model
predictions. Bundle provenance must be documented separately by the publisher.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_dynamics(bundle: str | Path, *, device: str = "cpu"):
    """Load an exported model bundle on the requested device in evaluation mode.

    ``bundle`` is a directory with ``model.json`` and ``weights.npz``. The
    metadata must contain ``latent_dim`` and ``model_config``. A CUDA request
    fails clearly when CUDA is unavailable instead of silently changing device.
    """
    from .tl.models import DynamicalModel

    bundle = Path(bundle)
    with (bundle / "model.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    latent_dim = int(metadata["latent_dim"])
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in PyTorch")
    model = DynamicalModel(latent_dim, metadata["model_config"])
    with np.load(bundle / "weights.npz", allow_pickle=False) as weights:
        state = {key: torch.from_numpy(weights[key].copy()) for key in weights.files}
    model.load_state_dict(state, strict=True)
    return model.to(requested_device).eval()


def load_mapper(weights: str | Path, *, input_dim1: int, input_dim2: int,
                hidden_dim: int, device: str = "cpu"):
    """Load an exported cross-modality mapper's numeric state dictionary."""
    from .Map.tl.models import SplicedAutoEncoder

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in PyTorch")
    model = SplicedAutoEncoder(input_dim1=input_dim1, input_dim2=input_dim2,
                               hidden_dim=hidden_dim, flat_mode=True)
    with np.load(weights, allow_pickle=False) as archive:
        state = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    model.load_state_dict(state, strict=True)
    return model.to(requested_device).eval()


def materialize_legacy_inputs(bundle: str | Path, output: str | Path, *, device: str = "cpu"):
    """Build transient legacy inference inputs from the minimal published bundle.

    Per-cell velocity and growth are computed locally, and no saved trajectory
    or original fitted AnnData is required. These files are generated results.
    """
    import anndata as ad

    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = ad.read_h5ad(bundle / "rna.h5ad")
    model = load_dynamics(bundle, device=device)
    times = np.asarray(data.obs["time_point_processed"], dtype=np.float32)
    latent = np.asarray(data.obsm["X_latent"], dtype=np.float32)
    velocities, growth = [], []
    with torch.no_grad():
        for start in range(0, len(data), 4096):
            x = torch.as_tensor(latent[start:start + 4096], device=device)
            t = torch.as_tensor(times[start:start + 4096, None], device=device)
            result = model(t, x, torch.zeros((len(x), 1), device=device))
            velocities.append(result["velocity"].cpu().numpy())
            growth.append(result["growth"].cpu().numpy())
    data.obsm["velocity_latent"] = np.concatenate(velocities)
    data.obsm["growth_rate"] = np.concatenate(growth)
    metadata = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
    with np.load(bundle / "weights.npz", allow_pickle=False) as weights:
        state = {key: weights[key].copy() for key in weights.files}
    data.uns["all_model"] = {"model_config": metadata["model_config"],
                             "training_config": {"plan": "[]"},
                             "model_state_dict": state}
    dynamic = output / "dynamics.h5ad"
    data.write_h5ad(dynamic, compression="gzip")
    maps = {}
    for archive_path in bundle.glob("map_*.npz"):
        with np.load(archive_path, allow_pickle=False) as archive:
            state = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
        target = output / (archive_path.stem + ".pt")
        torch.save({"model_state_dict": state}, target)
        maps[archive_path.stem] = target
    return {"dynamics": dynamic, "maps": maps}
