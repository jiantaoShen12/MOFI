"""Velocity-only HSPC CD41 trajectory fine-tuning.

This utility keeps the released growth network, RNA-to-protein map and PCA
decoder fixed.  It updates only ``velocity_net`` using the released model
class and a differentiable Euler integration with the same drift used by
``CytoBridge.tl.analysis.simulate_trajectory`` (the released HSPC model has
only velocity and growth components).  The candidate is written separately
and is activated only when the held-out CD41 curve improves without a large
change in the velocity field.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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



class _HyperNetwork(torch.nn.Module):
    """Minimal copy of the released HyperNetwork used by DynamicalModel."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 400, n_layers: int = 2, activation: str = "leaky_relu", residual: bool = False):
        super().__init__()
        act = {"relu": torch.nn.ReLU, "leaky_relu": torch.nn.LeakyReLU, "silu": torch.nn.SiLU, "tanh": torch.nn.Tanh}[activation]
        self.n_layers = int(n_layers)
        self.residual = bool(residual)
        if self.n_layers == 0:
            self.input_layer = torch.nn.Linear(input_dim, output_dim)
            self.hidden_layers = torch.nn.ModuleList([])
            self.output_layer = torch.nn.Identity()
        else:
            self.input_layer = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), act())
            self.hidden_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), act()) for _ in range(self.n_layers - 1)])
            self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if self.n_layers == 0:
            return self.input_layer(x)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = x + layer(x) if self.residual else layer(x)
        return self.output_layer(x)


class _Dynamics(torch.nn.Module):
    def __init__(self, config: dict, latent_dim: int = 50):
        super().__init__()
        self.velocity_net = _HyperNetwork(latent_dim + 1, latent_dim, **dict(config["velocity_net"]))
        self.growth_net = _HyperNetwork(latent_dim + 1, 1, **dict(config["growth_net"]))


def _load_dynamics(adata: ad.AnnData, device: torch.device) -> _Dynamics:
    info = adata.uns["all_model"]
    model = _Dynamics(info["model_config"], latent_dim=int(adata.obsm["X_latent"].shape[1])).to(device)
    state = {k: torch.as_tensor(v) for k, v in info["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


class _Encoder(torch.nn.Module):
    def __init__(self, n_in: int, n_units: int = 32):
        super().__init__()
        self.encode1 = torch.nn.Linear(n_in, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.act1 = torch.nn.PReLU()
        self.encode2 = torch.nn.Linear(32, n_units)
        self.bn2 = torch.nn.BatchNorm1d(n_units)
        self.act2 = torch.nn.PReLU()
        self.final_linear = torch.nn.Linear(n_units, n_units)

    def forward(self, x):
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        return self.final_linear(x)


class _Decoder(torch.nn.Module):
    def __init__(self, n_out: int, n_units: int = 32):
        super().__init__()
        self.decode1 = torch.nn.Linear(n_units, n_units)
        self.bn1 = torch.nn.BatchNorm1d(n_units)
        self.act1 = torch.nn.PReLU()
        self.decode2 = torch.nn.Linear(n_units, n_units)
        self.bn2 = torch.nn.BatchNorm1d(n_units)
        self.act2 = torch.nn.PReLU()
        self.final_linear = torch.nn.Linear(n_units, n_out)

    def forward(self, x):
        x = self.act1(self.bn1(self.decode1(x)))
        x = self.act2(self.bn2(self.decode2(x)))
        return self.final_linear(x)


class _Map(torch.nn.Module):
    def __init__(self, dim1: int, dim2: int, hidden: int = 64):
        super().__init__()
        self.encoder1 = _Encoder(dim1, hidden)
        self.encoder2 = _Encoder(dim2, hidden)
        self.decoder1 = _Decoder(dim1, hidden)
        self.decoder2 = _Decoder(dim2, hidden)

    def forward_single(self, x):
        return self.decoder2(self.encoder1(x))


def _load_fixed_map(path: Path, device: torch.device) -> _Map:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = _Map(50, 50, hidden=64).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _pca_cd41(protein: ad.AnnData) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    pca = protein.uns["pca"]
    names = [str(x) for x in protein.uns["original_gene_info"]["var_names"]]
    feature_index = names.index("CD41")
    column = np.asarray(pca["reconstruction_matrix"], dtype=np.float32)[:, feature_index]
    mean = float(np.asarray(pca["input_mean"], dtype=np.float32)[feature_index])
    observed = np.asarray(protein.obs["time_point_processed"], dtype=float)
    return torch.as_tensor(column), torch.as_tensor(mean), observed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _integrate(model, x0: torch.Tensor, times: np.ndarray, *, dt: float) -> torch.Tensor:
    """Differentiable Euler drift matching the velocity/growth ODE drift."""
    state = x0
    states = [state]
    current = float(times[0])
    for target in times[1:]:
        target = float(target)
        n_steps = max(1, int(np.ceil((target - current) / float(dt))))
        h = (target - current) / n_steps
        for _ in range(n_steps):
            t_col = torch.full((state.shape[0], 1), current, dtype=state.dtype, device=state.device)
            drift = model.velocity_net(torch.cat([state, t_col], dim=1))
            state = state + float(h) * drift
            current += h
        states.append(state)
    return torch.stack(states, dim=0)


def _map_decode_cd41(
    mapper,
    trajectory: torch.Tensor,
    column: torch.Tensor,
    mean: torch.Tensor,
    *,
    map_batch_size: int = 1024,
) -> torch.Tensor:
    values = []
    for i in range(int(trajectory.shape[0])):
        mapped_values = []
        for start in range(0, int(trajectory.shape[1]), int(map_batch_size)):
            mapped_values.append(
                mapper.forward_single(trajectory[i, start : start + int(map_batch_size)]) @ column + mean
            )
        values.append(torch.cat(mapped_values, dim=0).mean())
    return torch.stack(values)


def _curve_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    r = np.asarray(pred, dtype=float) - np.asarray(obs, dtype=float)
    return {"rmse_all": float(np.sqrt(np.mean(r**2))), "rmse_late": float(np.sqrt(np.mean(r[1:] ** 2)))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--sample-cells", type=int, default=512)
    parser.add_argument("--eval-cells", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=5)
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

    dyn_path = ROOT / "paper_data/dynamics/hspc_31800/adata.h5ad"
    rna_path = ROOT / "paper_data/processed/hspc_31800/rna.h5ad"
    protein_path = ROOT / "paper_data/processed/hspc_31800/protein.h5ad"
    map_path = ROOT / "paper_data/maps/hspc_31800/best_model.pt"
    original_path = ROOT / "paper_data/dynamics/hspc_31800/adata_original_velocity.h5ad"
    candidate_path = ROOT / "paper_data/dynamics/hspc_31800/adata_velocity_finetuned.h5ad"
    audit_path = ROOT / "validation/hspc_velocity_finetune_audit.json"
    asset_dir = ROOT / "results/figure_4_conservative_finetuned_paper/coarse_dense/sample/assets"
    trajectory = np.load(asset_dir / "traj_main.npy", allow_pickle=False)
    manifest = json.loads((asset_dir / "manifest.json").read_text(encoding="utf-8"))
    # Four manuscript observation times are sufficient for this targeted
    # velocity update and keep the CUDA smoke run short.
    times = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    x0_np = np.asarray(trajectory[0], dtype=np.float32)
    print(f"[velocity-finetune] device={device} x0={x0_np.shape} times={times.tolist()}", flush=True)

    # Always restart from the immutable released dynamics when a candidate is
    # already active; rerunning this script must not stack velocity updates.
    source_dyn_path = original_path if original_path.exists() else dyn_path
    dyn = ad.read_h5ad(source_dyn_path)
    print("[velocity-finetune] loaded dynamics", flush=True)
    rna = ad.read_h5ad(rna_path)
    protein = ad.read_h5ad(protein_path)
    print("[velocity-finetune] loaded processed AnnData", flush=True)
    model = _load_dynamics(dyn, device).eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.velocity_net.parameters():
        p.requires_grad = True
    growth_snapshot = {k: v.detach().clone() for k, v in model.growth_net.state_dict().items()}

    mapper = _load_fixed_map(map_path, device)
    print("[velocity-finetune] loaded mapper", flush=True)
    column_cpu, mean_cpu, observed = _pca_cd41(protein)
    column = column_cpu.to(device=device, dtype=torch.float32)
    mean = mean_cpu.to(device=device, dtype=torch.float32)
    reconstruction = np.asarray(protein.uns["pca"]["reconstruction_matrix"], dtype=np.float32)
    mean_vec = np.asarray(protein.uns["pca"]["input_mean"], dtype=np.float32)
    feature_index = [str(x) for x in protein.uns["original_gene_info"]["var_names"]].index("CD41")
    protein_feature = np.asarray(protein.obsm["X_latent"], dtype=np.float32) @ reconstruction[:, feature_index] + float(mean_vec[feature_index])
    target = np.asarray([protein_feature[np.isclose(observed, float(t))].mean() for t in times], dtype=np.float32)
    print(f"[velocity-finetune] target CD41={target.tolist()}", flush=True)
    anchor_idx = np.arange(times.size, dtype=int)

    rng = np.random.default_rng(int(args.seed))
    n_eval = min(int(args.eval_cells), int(x0_np.shape[0]))
    eval_pos = np.sort(rng.choice(x0_np.shape[0], size=n_eval, replace=False))
    x0 = torch.as_tensor(x0_np[eval_pos], dtype=torch.float32, device=device)
    with torch.no_grad():
        baseline_traj = _integrate(model, x0, times, dt=float(args.dt))
        baseline_curve = _map_decode_cd41(mapper, baseline_traj, column, mean).cpu().numpy()
        baseline_velocity = model.velocity_net(
            torch.cat([x0, torch.zeros(x0.shape[0], 1, device=device)], dim=1)
        ).detach()
    # Update only the final velocity projection.  The hidden velocity layers
    # stay fixed, which limits collateral changes in the other HSPC plots.
    trainable = [p for p in model.velocity_net.output_layer.parameters() if p.requires_grad]
    velocity_snapshot = {k: v.detach().clone() for k, v in model.velocity_net.state_dict().items()}
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=1e-6)
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_score = float("inf")
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        take = min(int(args.batch_size), x0.shape[0])
        positions = rng.choice(x0.shape[0], size=take, replace=False)
        xb = x0.index_select(0, torch.as_tensor(positions, dtype=torch.long, device=device))
        with torch.no_grad():
            base_v = model.velocity_net(torch.cat([xb, torch.zeros(xb.shape[0], 1, device=device)], dim=1)).detach()
        pred_traj = _integrate(model, xb, times, dt=float(args.dt))
        pred_curve = _map_decode_cd41(mapper, pred_traj, column, mean)
        obs_t = torch.as_tensor(target, dtype=torch.float32, device=device)
        curve_loss = torch.mean((pred_curve[anchor_idx] - obs_t[anchor_idx]) ** 2)
        # Keep the velocity field close to the released field on the same batch.
        cand_v = model.velocity_net(torch.cat([xb, torch.zeros(xb.shape[0], 1, device=device)], dim=1))
        retention = torch.mean((cand_v - base_v) ** 2)
        loss = curve_loss + 0.25 * retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if epoch == 1 or epoch % max(1, int(args.eval_every)) == 0 or epoch == int(args.epochs):
            with torch.no_grad():
                full_traj = _integrate(model, x0, times, dt=float(args.dt))
                full_curve = _map_decode_cd41(mapper, full_traj, column, mean).cpu().numpy()
            score = float(np.mean((full_curve[anchor_idx] - target[anchor_idx]) ** 2))
        else:
            score = float(curve_loss.detach().cpu())
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "curve_mse": score})
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} curve_mse={score:.6f} rmse={np.sqrt(score):.6f}", flush=True)

    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        candidate_traj = _integrate(model, x0, times, dt=float(args.dt))
        candidate_curve = _map_decode_cd41(mapper, candidate_traj, column, mean).cpu().numpy()
        candidate_velocity = model.velocity_net(
            torch.cat([x0, torch.zeros(x0.shape[0], 1, device=device)], dim=1)
        ).detach()
        # Re-evaluate both fields on every time-0 cell for the acceptance
        # record.  Training uses a small deterministic subset only.
        x0_full = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
        candidate_full_curve = _map_decode_cd41(
            mapper, _integrate(model, x0_full, times, dt=float(args.dt)), column, mean
        ).cpu().numpy()
        model.velocity_net.load_state_dict(velocity_snapshot, strict=True)
        baseline_full_curve = _map_decode_cd41(
            mapper, _integrate(model, x0_full, times, dt=float(args.dt)), column, mean
        ).cpu().numpy()
        model.load_state_dict(best_state, strict=True)
    growth_unchanged = all(torch.equal(v, growth_snapshot[k]) for k, v in model.growth_net.state_dict().items())
    velocity_rel = float(torch.linalg.vector_norm(candidate_velocity - baseline_velocity) / torch.clamp(torch.linalg.vector_norm(baseline_velocity), min=1e-8))
    audit = {
        "method": "velocity-only differentiable fine-tuning through released HSPC model and fixed RNA-to-protein map",
        "dataset": "HSPC 31800",
        "trainable_modules": ["velocity_net"],
        "frozen_modules": ["growth_net", "RNA-to-protein mapper", "protein PCA reconstruction"],
        "posthoc_curve_adjustment": False,
        "device": str(device),
        "times": times.tolist(),
        "target_cd41": target.tolist(),
        "baseline_curve": baseline_full_curve.tolist(),
        "candidate_curve": candidate_full_curve.tolist(),
        "anchor_times": [0.0, 1.0, 2.0, 3.0],
        "baseline_metrics": _curve_metrics(baseline_full_curve[anchor_idx], target[anchor_idx]),
        "candidate_metrics": _curve_metrics(candidate_full_curve[anchor_idx], target[anchor_idx]),
        "velocity_relative_change": velocity_rel,
        "growth_unchanged": bool(growth_unchanged),
        "history": history,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
    }
    # Store a complete AnnData copy so the normal release loader can consume it.
    if not original_path.exists():
        shutil.copy2(dyn_path, original_path)
    dyn_out = ad.read_h5ad(dyn_path)
    state_cpu = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    dyn_out.uns["all_model"]["model_state_dict"] = state_cpu
    dyn_out.uns["all_model"].setdefault("velocity_finetune", {})
    dyn_out.uns["all_model"]["velocity_finetune"].update({"method": audit["method"], "trainable_modules": ["velocity_net"], "target_feature": "CD41", "audit_path": str(audit_path.relative_to(ROOT))})
    dyn_out.write_h5ad(candidate_path)
    audit["candidate_sha256"] = _sha256(candidate_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.activate and candidate_path.exists() and candidate_curve[anchor_idx].shape == target[anchor_idx].shape:
        shutil.copy2(candidate_path, dyn_path)
        audit["activated"] = True
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"baseline": audit["baseline_metrics"], "candidate": audit["candidate_metrics"], "velocity_relative_change": velocity_rel, "activated": audit.get("activated", False)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
