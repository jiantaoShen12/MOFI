from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CytoBridge.tl.analysis_dense_time import _reconstruct_latent_to_feature

from .context import TFRunContext


def _slug(text: Any) -> str:
    raw = str(text).strip().lower()
    out = []
    for ch in raw:
        out.append(ch if ch.isalnum() else "-")
    token = "".join(out).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "na"


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    std = float(np.nanstd(arr))
    if (not np.isfinite(std)) or std < 1e-12:
        return np.zeros_like(arr, dtype=float)
    mean = float(np.nanmean(arr))
    return (arr - mean) / std


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    if weights is None:
        return float(np.mean(x))
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != x.shape[0]:
        raise ValueError(f"weights length mismatch: {w.shape[0]} vs {x.shape[0]}")
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    if float(w.sum()) <= 1e-12:
        return float(np.mean(x))
    return float(np.average(x, weights=w))


def _align_with_lag(x: np.ndarray, y: np.ndarray, lag_step: int) -> Tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)
    if xa.shape[0] != ya.shape[0]:
        raise ValueError("x/y length mismatch")
    if lag_step == 0:
        return xa, ya
    if lag_step > 0:
        if lag_step >= xa.shape[0]:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return xa[:-lag_step], ya[lag_step:]
    shift = -lag_step
    if shift >= xa.shape[0]:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return xa[shift:], ya[:-shift]


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xs = pd.Series(np.asarray(x, dtype=float))
    ys = pd.Series(np.asarray(y, dtype=float))
    if xs.size < 2 or ys.size < 2:
        return float("nan")
    if xs.nunique(dropna=True) <= 1 or ys.nunique(dropna=True) <= 1:
        return float("nan")
    return float(xs.corr(ys, method="spearman"))


def _build_dense_time_grid(times: np.ndarray, line_dt: float) -> np.ndarray:
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size <= 1 or (not np.isfinite(line_dt)) or line_dt <= 0:
        return t
    t0 = float(t.min())
    t1 = float(t.max())
    if t1 <= t0:
        return t
    dense = np.arange(t0, t1 + 0.5 * float(line_dt), float(line_dt), dtype=float)
    dense = np.clip(dense, t0, t1)
    if dense.size == 0 or (not np.isclose(dense[-1], t1)):
        dense = np.append(dense, t1)
    return np.asarray(np.unique(dense), dtype=float)


def _pick_key_times(times: np.ndarray, key_time_step: float) -> np.ndarray:
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size <= 1 or (not np.isfinite(key_time_step)) or key_time_step <= 0:
        return t
    t0 = float(t.min())
    t1 = float(t.max())
    anchors = np.arange(t0, t1 + 0.5 * float(key_time_step), float(key_time_step), dtype=float)
    out_idx: List[int] = []
    for a in anchors:
        right = int(np.searchsorted(t, a, side="left"))
        if right <= 0:
            idx = 0
        elif right >= t.size:
            idx = int(t.size - 1)
        else:
            left = right - 1
            idx = left if abs(t[left] - a) <= abs(t[right] - a) else right
        out_idx.append(idx)
    out_idx = sorted(set(out_idx))
    return np.asarray([t[i] for i in out_idx], dtype=float)


def build_pair_timeseries(
    ctx: TFRunContext,
    *,
    pairs_df: pd.DataFrame,
    time_points: Sequence[float],
    max_lag_steps: int = 2,
    line_dt: float = 0.25,
    key_time_step: float = 1.0,
    plot_top_n: int = 12,
) -> Dict[str, Any]:
    if pairs_df.empty:
        raise ValueError("pairs_df is empty")

    out_dir = ctx.output_figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_primary, traj_secondary, weights = ctx.load_traj_arrays()
    full_t = np.asarray(ctx.time_grid, dtype=float)

    picked: List[Tuple[int, float]] = []
    for t in time_points:
        idx = np.where(np.isclose(full_t, float(t)))[0]
        if idx.size == 0:
            continue
        picked.append((int(idx[0]), float(full_t[int(idx[0])])) )
    if not picked:
        picked = [(i, float(x)) for i, x in enumerate(full_t.tolist())]

    primary_bundle, secondary_bundle = ctx.load_primary_secondary_bundles()
    p_lut = {str(v): i for i, v in enumerate(primary_bundle.var_names)}
    s_lut = {str(v): i for i, v in enumerate(secondary_bundle.var_names)}

    used_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    all_trend_rows: List[pd.DataFrame] = []
    all_dense_rows: List[pd.DataFrame] = []
    lag_rows: List[Dict[str, Any]] = []

    for _, pr in pairs_df.iterrows():
        p_var = str(pr["primary_var"])
        s_var = str(pr["secondary_var"])
        p_idx = p_lut.get(p_var, None)
        s_idx = s_lut.get(s_var, None)
        if p_idx is None or s_idx is None:
            missing_rows.append(
                {
                    "primary_var": p_var,
                    "secondary_var": s_var,
                    "primary_found": bool(p_idx is not None),
                    "secondary_found": bool(s_idx is not None),
                }
            )
            continue

        rows: List[Dict[str, Any]] = []
        for t_idx, t_val in picked:
            x_primary = np.asarray(traj_primary[t_idx], dtype=np.float32)
            x_secondary = np.asarray(traj_secondary[t_idx], dtype=np.float32)
            feat_primary = _reconstruct_latent_to_feature(x_primary, primary_bundle)
            feat_secondary = _reconstruct_latent_to_feature(x_secondary, secondary_bundle)
            wt = None if weights is None else np.asarray(weights[t_idx], dtype=float)
            p_value = _weighted_mean(feat_primary[:, int(p_idx)], wt)
            s_value = _weighted_mean(feat_secondary[:, int(s_idx)], wt)
            rows.append(
                {
                    "primary_var": p_var,
                    "secondary_var": s_var,
                    "time": float(t_val),
                    "primary_value": float(p_value),
                    "secondary_value": float(s_value),
                }
            )

        trend_df = pd.DataFrame(rows).sort_values("time")
        p_z = _zscore(trend_df["primary_value"].to_numpy(dtype=float))
        s_z = _zscore(trend_df["secondary_value"].to_numpy(dtype=float))
        trend_df["primary_z"] = p_z
        trend_df["secondary_z"] = s_z

        native_time = trend_df["time"].to_numpy(dtype=float)
        dense_time = _build_dense_time_grid(native_time, float(line_dt))
        dense_df = pd.DataFrame(
            {
                "primary_var": p_var,
                "secondary_var": s_var,
                "time": dense_time,
                "primary_z": np.interp(dense_time, native_time, p_z),
                "secondary_z": np.interp(dense_time, native_time, s_z),
            }
        )

        key_time = _pick_key_times(native_time, float(key_time_step))
        key_p = np.interp(key_time, native_time, p_z)
        key_s = np.interp(key_time, native_time, s_z)

        best_lag = 0
        best_rho = float("nan")
        best_n = 0
        for lag in range(-int(max_lag_steps), int(max_lag_steps) + 1):
            xa, ya = _align_with_lag(p_z, s_z, int(lag))
            if xa.shape[0] < 3:
                continue
            rho = _safe_spearman(xa, ya)
            if not np.isfinite(rho):
                continue
            if (not np.isfinite(best_rho)) or (abs(float(rho)) > abs(float(best_rho))):
                best_rho = float(rho)
                best_lag = int(lag)
                best_n = int(xa.shape[0])

        slug = f"{_slug(p_var)}__vs__{_slug(s_var)}"
        per_csv = out_dir / f"main_sub_pair_timeseries__{slug}.csv"
        per_dense = out_dir / f"main_sub_pair_timeseries_dense__{slug}.csv"
        trend_df.to_csv(per_csv, index=False)
        dense_df.to_csv(per_dense, index=False)

        used_rows.append(
            {
                "primary_var": p_var,
                "secondary_var": s_var,
                "best_spearman": float(best_rho),
                "best_lag_step": int(best_lag),
                "n_points": int(best_n),
                "csv": str(per_csv),
                "dense_csv": str(per_dense),
                "n_native_points": int(native_time.shape[0]),
                "n_dense_points": int(dense_df.shape[0]),
            }
        )
        lag_rows.append(
            {
                "primary_var": p_var,
                "secondary_var": s_var,
                "best_spearman": float(best_rho),
                "best_lag_step": int(best_lag),
                "n_points": int(best_n),
            }
        )

        all_trend_rows.append(trend_df)
        all_dense_rows.append(dense_df)

        # key points for optional plotting cache
        trend_df["_key_time"] = False
        if key_time.size > 0:
            trend_df.loc[np.isclose(trend_df["time"].to_numpy(dtype=float)[:, None], key_time[None, :]).any(axis=1), "_key_time"] = True

    used_df = pd.DataFrame(used_rows)
    missing_df = pd.DataFrame(missing_rows)
    trend_all = pd.concat(all_trend_rows, axis=0, ignore_index=True) if all_trend_rows else pd.DataFrame()
    dense_all = pd.concat(all_dense_rows, axis=0, ignore_index=True) if all_dense_rows else pd.DataFrame()
    lag_df = pd.DataFrame(lag_rows)

    used_csv = out_dir / "main_sub_pairs_used_tf.csv"
    missing_csv = out_dir / "main_sub_pairs_missing_tf.csv"
    trend_csv = out_dir / "main_sub_pairs_timeseries_all_tf.csv"
    dense_csv = out_dir / "main_sub_pairs_timeseries_dense_all_tf.csv"
    lag_csv = out_dir / "main_sub_pairs_lag_all_tf.csv"

    used_df.to_csv(used_csv, index=False)
    missing_df.to_csv(missing_csv, index=False)
    trend_all.to_csv(trend_csv, index=False)
    dense_all.to_csv(dense_csv, index=False)
    lag_df.to_csv(lag_csv, index=False)

    if not used_df.empty:
        plot_df = used_df.copy()
        plot_df["abs_rho"] = np.abs(np.asarray(plot_df["best_spearman"], dtype=float))
        plot_df = plot_df.sort_values("abs_rho", ascending=False).head(int(max(1, plot_top_n)))

        for _, row in plot_df.iterrows():
            p_var = str(row["primary_var"])
            s_var = str(row["secondary_var"])
            pair_sub = dense_all[
                (dense_all["primary_var"].astype(str) == p_var)
                & (dense_all["secondary_var"].astype(str) == s_var)
            ].copy()
            if pair_sub.empty:
                continue
            pair_sub = pair_sub.sort_values("time")
            png = out_dir / f"main_sub_pair_timeseries__{_slug(p_var)}__vs__{_slug(s_var)}.png"
            png = out_dir / f"main_sub_pair_timeseries__{_slug(p_var)}__vs__{_slug(s_var)}.pdf"

            plt.style.use("seaborn-v0_8-darkgrid")
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            ax.plot(pair_sub["time"], pair_sub["primary_z"], linewidth=2.0, marker="o", label=f"{p_var} (primary)")
            ax.plot(pair_sub["time"], pair_sub["secondary_z"], linewidth=2.0, marker="s", label=f"{s_var} (secondary)")
            ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_xlabel("time")
            ax.set_ylabel("variable z-score")
            ax.set_title(f"{p_var}_primary vs {s_var}_secondary")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(png, dpi=260, bbox_inches="tight")
            plt.close(fig)

    summary = {
        "run_dir": str(ctx.run_dir),
        "output_dir": str(out_dir),
        "requested_pairs": int(pairs_df.shape[0]),
        "used_pairs": int(used_df.shape[0]),
        "missing_pairs": int(missing_df.shape[0]),
        "time_points": [float(x) for _, x in picked],
        "line_dt": float(line_dt),
        "key_time_step": float(key_time_step),
        "used_csv": str(used_csv),
        "missing_csv": str(missing_csv),
        "combined_timeseries_csv": str(trend_csv),
        "combined_timeseries_dense_csv": str(dense_csv),
        "combined_lag_csv": str(lag_csv),
    }
    out_json = out_dir / "main_sub_pairs_summary_tf.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_json"] = str(out_json)
    return summary
