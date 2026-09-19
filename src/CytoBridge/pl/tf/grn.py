from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from CytoBridge.tl.analysis_dense_time import (
    _batched_main_to_sub_terms_autograd,
    _batched_mapper_reverse_grad_autograd,
    _batched_velocity,
    _batched_velocity_jacobian,
)

from .context import TFRunContext


def _time_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _match_time_index(time_grid: np.ndarray, t: float) -> int:
    idx = np.where(np.isclose(time_grid, float(t), atol=1e-8, rtol=0.0))[0]
    if idx.size == 0:
        raise ValueError(f"time={t:g} not found in run time_grid={time_grid.tolist()}")
    return int(idx[0])


def _batched_velocity_jacobian_fast(
    model: torch.nn.Module,
    x_np: np.ndarray,
    t_val: float,
    *,
    device: str = "cpu",
    chunk_size: int = 256,
) -> np.ndarray:
    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"x_np must be 2D, got shape={x_np.shape}")
    if not hasattr(torch, "func"):
        return _batched_velocity_jacobian(model, x_np, float(t_val), device=device)

    n, d = x_np.shape
    if not hasattr(model, "velocity_net"):
        return np.zeros((n, d, d), dtype=np.float32)

    device_t = torch.device(device)
    model = model.to(device_t).eval()
    t_const = torch.tensor([float(t_val)], dtype=torch.float32, device=device_t)

    def _single_velocity(x_single: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_single, t_const], dim=0).unsqueeze(0)
        return model.velocity_net(inp).squeeze(0)

    jac_single = torch.func.jacrev(_single_velocity)
    out: List[np.ndarray] = []
    with torch.enable_grad():
        for start in range(0, n, int(chunk_size)):
            end = min(n, start + int(chunk_size))
            xb = torch.tensor(
                x_np[start:end],
                dtype=torch.float32,
                device=device_t,
                requires_grad=True,
            )
            jb = torch.vmap(jac_single)(xb)
            out.append(np.asarray(jb.detach().cpu().numpy(), dtype=np.float32))
    return np.concatenate(out, axis=0)


def export_grn_tables(
    ctx: TFRunContext,
    *,
    time_points: Sequence[float],
    device: str = "cpu",
    chunk_size: int = 256,
    mapper_batch_size: int = 256,
    include_hessian_term: bool = False,
) -> Dict[str, Any]:
    table_dir = ctx.output_table_dir
    table_dir.mkdir(parents=True, exist_ok=True)

    traj_primary, _, _ = ctx.load_traj_arrays()
    time_grid = np.asarray(ctx.time_grid, dtype=float)
    if traj_primary.shape[0] != time_grid.shape[0]:
        raise ValueError(
            f"time_grid length mismatch: traj_time={traj_primary.shape[0]} grid={time_grid.shape[0]}"
        )

    primary_proc, secondary_proc = ctx.load_primary_secondary_processed(backed=False)
    model = ctx.load_model()
    mapper = ctx.build_mapper(device=str(device))

    primary_bundle, secondary_bundle = ctx.load_primary_secondary_bundles()
    primary_var_names = [str(x) for x in primary_bundle.var_names]
    secondary_var_names = [str(x) for x in secondary_bundle.var_names]

    primary_forward = np.asarray(primary_bundle.forward, dtype=np.float32)
    primary_recon = np.asarray(primary_bundle.reconstruction, dtype=np.float32)
    secondary_forward = np.asarray(secondary_bundle.forward, dtype=np.float32)
    secondary_recon = np.asarray(secondary_bundle.reconstruction, dtype=np.float32)

    out: Dict[str, Any] = {
        "run_dir": str(ctx.run_dir),
        "time_points": [float(x) for x in time_points],
        "primary_var_count": int(len(primary_var_names)),
        "secondary_var_count": int(len(secondary_var_names)),
        "primary_to_primary_files": [],
        "primary_to_secondary_files": [],
        "secondary_to_primary_files": [],
        "include_hessian_term": bool(include_hessian_term),
    }

    start_all = time.time()
    for t in time_points:
        t_idx = _match_time_index(time_grid, float(t))
        x_primary = np.asarray(traj_primary[t_idx], dtype=np.float32)

        jac_z = _batched_velocity_jacobian_fast(
            model,
            x_primary,
            float(t),
            device=str(device),
            chunk_size=int(chunk_size),
        )
        xdot_primary = _batched_velocity(model, x_primary, float(t), device=str(device))

        _, term1, term2, y_secondary = _batched_main_to_sub_terms_autograd(
            mapper,
            x_primary,
            xdot_primary,
            jac_z,
            batch_size=int(mapper_batch_size),
            include_hessian_term=bool(include_hessian_term),
            device=str(device),
        )
        jac_map_reverse = _batched_mapper_reverse_grad_autograd(
            mapper,
            y_secondary,
            batch_size=int(mapper_batch_size),
            device=str(device),
        )

        p2s_latent = np.asarray(term1 + term2, dtype=np.float32) if include_hessian_term else np.asarray(term2, dtype=np.float32)
        s2p_latent = np.asarray(np.einsum("bij,bjk->bik", jac_z, jac_map_reverse), dtype=np.float32)

        jac_z_mean = np.asarray(jac_z.mean(axis=0), dtype=np.float32)
        p2s_mean = np.asarray(p2s_latent.mean(axis=0), dtype=np.float32)

        p2p = np.asarray(primary_forward @ jac_z_mean.T @ primary_recon, dtype=np.float32)
        p2s = np.asarray(primary_forward @ p2s_mean.T @ secondary_recon, dtype=np.float32)

        s2p_acc = np.zeros((len(secondary_var_names), len(primary_var_names)), dtype=np.float64)
        for i in range(s2p_latent.shape[0]):
            s2p_acc += np.asarray(
                secondary_forward @ s2p_latent[i].T @ primary_recon,
                dtype=np.float64,
            )
        s2p = np.asarray(s2p_acc / float(max(1, s2p_latent.shape[0])), dtype=np.float32)

        token = _time_token(float(t))
        p2p_path = table_dir / f"grn_main_to_main__t-{token}.csv"
        p2s_path = table_dir / f"grn_main_to_sub__t-{token}.csv"
        s2p_path = table_dir / f"grn_sub_to_main__t-{token}.csv"

        pd.DataFrame(p2p, index=primary_var_names, columns=primary_var_names).to_csv(
            p2p_path,
            index=True,
            index_label="source_primary_var",
        )
        pd.DataFrame(p2s, index=primary_var_names, columns=secondary_var_names).to_csv(
            p2s_path,
            index=True,
            index_label="source_primary_var",
        )
        pd.DataFrame(s2p, index=secondary_var_names, columns=primary_var_names).to_csv(
            s2p_path,
            index=True,
            index_label="source_secondary_var",
        )

        out["primary_to_primary_files"].append(str(p2p_path))
        out["primary_to_secondary_files"].append(str(p2s_path))
        out["secondary_to_primary_files"].append(str(s2p_path))

    out["elapsed_seconds"] = float(time.time() - start_all)
    summary_path = table_dir / "grn_full_export_summary_tf.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    out["summary_json"] = str(summary_path)
    return out


def _load_p2s_value_and_rank(csv_path: Path, source_primary: str, target_secondary: str) -> Tuple[float, int]:
    df = pd.read_csv(csv_path)
    if target_secondary not in df.columns:
        raise ValueError(f"{target_secondary} missing in {csv_path.name}")
    sources = df.iloc[:, 0].astype(str).to_numpy()
    idx = np.where(sources == str(source_primary))[0]
    if idx.size == 0:
        raise ValueError(f"{source_primary} missing in {csv_path.name}")

    col = df[target_secondary].to_numpy(dtype=float)
    i = int(idx[0])
    abs_order = np.argsort(np.abs(col))[::-1]
    rank_abs = int(np.where(abs_order == i)[0][0]) + 1
    return float(col[i]), int(rank_abs)


def _auto_support_pairs_from_p2p(p2p_csv: Path, focus_sources: Sequence[str], top_n: int = 8) -> List[Tuple[str, str]]:
    if not p2p_csv.exists():
        return []
    df = pd.read_csv(p2p_csv)
    genes = df.iloc[:, 0].astype(str).to_numpy()
    out: List[Tuple[str, str, float]] = []
    for src in focus_sources:
        idx = np.where(genes == str(src))[0]
        if idx.size == 0:
            continue
        row = df.iloc[int(idx[0]), 1:]
        arr = np.asarray(row.to_numpy(dtype=float), dtype=float)
        cols = [str(c) for c in row.index.tolist()]
        order = np.argsort(np.abs(arr))[::-1]
        for j in order:
            tgt = cols[int(j)]
            if tgt == str(src):
                continue
            out.append((str(src), str(tgt), float(abs(arr[int(j)]))))
            break
    out = sorted(out, key=lambda x: x[2], reverse=True)

    uniq: List[Tuple[str, str]] = []
    seen = set()
    for src, tgt, _ in out:
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((src, tgt))
        if len(uniq) >= int(top_n):
            break
    return uniq


def build_grn_focus_from_pairs(
    ctx: TFRunContext,
    *,
    pairs_df: pd.DataFrame,
    time_points: Sequence[float],
    focus_top_n: int = 8,
) -> Dict[str, Any]:
    out_dir = ctx.output_figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if pairs_df.empty:
        raise ValueError("pairs_df is empty")

    if "score" in pairs_df.columns:
        focus_pairs_df = pairs_df.sort_values("score", ascending=False).head(int(focus_top_n))
    else:
        focus_pairs_df = pairs_df.head(int(focus_top_n))

    focus_pairs = [
        (str(r["primary_var"]), str(r["secondary_var"]))
        for _, r in focus_pairs_df.iterrows()
    ]

    rows: List[Dict[str, Any]] = []
    for t in time_points:
        token = _time_token(float(t))
        p2s_path = ctx.output_table_dir / f"grn_main_to_sub__t-{token}.csv"
        if not p2s_path.exists():
            p2s_path = ctx.tables_dir / f"grn_main_to_sub__t-{token}.csv"
        if not p2s_path.exists():
            raise FileNotFoundError(f"missing grn table: {p2s_path}")

        for src, dst in focus_pairs:
            try:
                value, rank_abs = _load_p2s_value_and_rank(p2s_path, src, dst)
            except ValueError:
                continue
            rows.append(
                {
                    "time": float(t),
                    "pair": f"{src}->{dst}",
                    "source_primary_var": src,
                    "target_secondary_var": dst,
                    "weight": float(value),
                    "rank_abs": int(rank_abs),
                }
            )

    focus_df = pd.DataFrame(rows)
    focus_csv = out_dir / "grn_focus_pairs_values.csv"
    focus_df.to_csv(focus_csv, index=False)

    weight_png = out_dir / "grn_focus_pairs_weight_time.png"
    rank_png = out_dir / "grn_focus_pairs_rank_time.png"
    weight_png = out_dir / "grn_focus_pairs_weight_time.pdf"
    rank_png = out_dir / "grn_focus_pairs_rank_time.pdf"

    if not focus_df.empty:
        plt.figure(figsize=(8.8, 5.2), dpi=240)
        for pair, sub in focus_df.groupby("pair"):
            s = sub.sort_values("time")
            plt.plot(s["time"], s["weight"], marker="o", linewidth=2.0, label=pair)
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        plt.xlabel("time")
        plt.ylabel("primary->secondary Jacobian weight")
        plt.title("Focused Primary->Secondary Coupling")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(weight_png, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8.8, 5.2), dpi=240)
        for pair, sub in focus_df.groupby("pair"):
            s = sub.sort_values("time")
            plt.plot(s["time"], s["rank_abs"], marker="o", linewidth=2.0, label=pair)
        plt.gca().invert_yaxis()
        plt.xlabel("time")
        plt.ylabel("abs-rank (lower=stronger)")
        plt.title("Focused Coupling Rank Over Time")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(rank_png, bbox_inches="tight")
        plt.close()

    support_csv = out_dir / "grn_focus_primary_to_primary_support_values.csv"
    support_png = out_dir / "grn_focus_primary_to_primary_support_weight_time.png"
    support_png = out_dir / "grn_focus_primary_to_primary_support_weight_time.pdf"

    first_token = _time_token(float(time_points[0]))
    p2p_first = ctx.output_table_dir / f"grn_main_to_main__t-{first_token}.csv"
    if not p2p_first.exists():
        p2p_first = ctx.tables_dir / f"grn_main_to_main__t-{first_token}.csv"
    support_pairs = _auto_support_pairs_from_p2p(
        p2p_first,
        focus_sources=[x[0] for x in focus_pairs],
        top_n=max(1, int(focus_top_n)),
    )

    support_rows: List[Dict[str, Any]] = []
    for t in time_points:
        token = _time_token(float(t))
        p2p_path = ctx.output_table_dir / f"grn_main_to_main__t-{token}.csv"
        if not p2p_path.exists():
            p2p_path = ctx.tables_dir / f"grn_main_to_main__t-{token}.csv"
        if not p2p_path.exists():
            continue
        df = pd.read_csv(p2p_path)
        sources = df.iloc[:, 0].astype(str).to_numpy()
        for src, tgt in support_pairs:
            if tgt not in df.columns:
                continue
            idx = np.where(sources == str(src))[0]
            if idx.size == 0:
                continue
            value = float(df[tgt].to_numpy(dtype=float)[int(idx[0])])
            support_rows.append(
                {
                    "time": float(t),
                    "pair": f"{src}->{tgt}",
                    "source_primary_var": src,
                    "target_primary_var": tgt,
                    "weight": value,
                }
            )

    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(support_csv, index=False)

    if not support_df.empty:
        plt.figure(figsize=(8.8, 5.2), dpi=240)
        for pair, sub in support_df.groupby("pair"):
            s = sub.sort_values("time")
            plt.plot(s["time"], s["weight"], marker="o", linewidth=1.9, label=pair)
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        plt.xlabel("time")
        plt.ylabel("primary->primary Jacobian weight")
        plt.title("Selected Primary->Primary Support Relations")
        plt.legend(frameon=False, fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(support_png, bbox_inches="tight")
        plt.close()

    manifest = {
        "focus_pairs": [f"{a}->{b}" for a, b in focus_pairs],
        "support_pairs": [f"{a}->{b}" for a, b in support_pairs],
        "time_points": [float(x) for x in time_points],
        "focus_csv": str(focus_csv),
        "focus_weight_png": str(weight_png) if weight_png.exists() else None,
        "focus_rank_png": str(rank_png) if rank_png.exists() else None,
        "support_csv": str(support_csv),
        "support_png": str(support_png) if support_png.exists() else None,
    }
    out_path = out_dir / "grn_focus_manifest_tf.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifest_json"] = str(out_path)
    return manifest
