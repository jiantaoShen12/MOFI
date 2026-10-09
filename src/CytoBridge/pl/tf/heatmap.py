from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
except Exception:  # pragma: no cover - optional dependency fallback
    leaves_list = None
    linkage = None

from .context import TFRunContext


def _time_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _load_time_tables(
    table_dirs: Sequence[Path],
    times: Sequence[float],
    *,
    table_prefix: str = "grn_main_to_sub",
) -> Dict[float, pd.DataFrame]:
    out: Dict[float, pd.DataFrame] = {}
    for t in times:
        token = _time_token(float(t))
        found = None
        for d in table_dirs:
            p = d / f"{str(table_prefix)}__t-{token}.csv"
            if p.exists():
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"missing {str(table_prefix)} table for t={t:g}")
        out[float(t)] = pd.read_csv(found)
    return out


def _compute_pair_matrices(
    *,
    time_tables: Dict[float, pd.DataFrame],
    times: Sequence[float],
    pairs_df: pd.DataFrame,
    source_col: str = "primary_var",
    target_col: str = "secondary_var",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wide_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []

    for _, row in pairs_df.iterrows():
        src = str(row[str(source_col)])
        dst = str(row[str(target_col)])
        base = {"pair": f"{src}->{dst}", "primary_var": src, "secondary_var": dst}
        for t in times:
            df = time_tables[float(t)]
            genes = df.iloc[:, 0].astype(str).to_numpy()
            idx = np.where(genes == src)[0]
            if idx.size == 0 or dst not in df.columns:
                value = np.nan
            else:
                value = float(df[dst].to_numpy(dtype=float)[int(idx[0])])
            base[f"t={float(t):g}"] = value
            long_rows.append(
                {
                    "pair": f"{src}->{dst}",
                    "time": float(t),
                    "weight": value,
                    "primary_var": src,
                    "secondary_var": dst,
                }
            )
        wide_rows.append(base)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)

    if wide_df.empty:
        return wide_df, long_df

    ranker = (
        long_df.groupby("pair", as_index=False)["weight"]
        .apply(lambda x: float(np.nanmax(np.abs(x.to_numpy(dtype=float)))) if len(x) else 0.0)
        .rename(columns={"weight": "max_abs_weight"})
        .sort_values("max_abs_weight", ascending=False)
    )
    order = ranker["pair"].astype(str).tolist()
    wide_df["pair"] = pd.Categorical(wide_df["pair"], categories=order, ordered=True)
    wide_df = wide_df.sort_values("pair").reset_index(drop=True)
    return wide_df, long_df


def _subset_vmax(wide_df: pd.DataFrame, times: Sequence[float]) -> float:
    cols = [f"t={float(t):g}" for t in times]
    mat = wide_df[cols].to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    return max(vmax, 1e-8)


def _full_vmax(time_tables: Dict[float, pd.DataFrame]) -> float:
    vmax = 0.0
    for df in time_tables.values():
        vals = df.iloc[:, 1:].to_numpy(dtype=float)
        local = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 0.0
        if local > vmax:
            vmax = local
    return max(vmax, 1e-8)


def _row_zscore(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i]
        m = float(np.nanmean(row))
        s = float(np.nanstd(row))
        if (not np.isfinite(s)) or s < 1e-12:
            out[i] = np.zeros_like(row, dtype=float)
        else:
            out[i] = (row - m) / s
    return out


def _auto_points_between(n_real_points: int) -> int:
    n = int(max(n_real_points, 0))
    if n <= 3:
        return 3
    if n == 4:
        return 2
    if n < 10:
        return 1
    return 0


def _build_dense_time_grid(
    native_time: Sequence[float],
    *,
    points_between: int = 0,
    target_total_points: int = 0,
) -> np.ndarray:
    t = np.asarray(native_time, dtype=float).reshape(-1)
    t = np.asarray(np.unique(t), dtype=float)
    if t.size <= 1:
        return t

    n_intervals = int(t.size - 1)
    target_n = int(max(target_total_points, 0))
    if target_n > int(t.size):
        extra = int(target_n - int(t.size))
        base = int(extra // n_intervals)
        rem = int(extra % n_intervals)
        out: List[float] = []
        for i in range(n_intervals):
            t0 = float(t[i])
            t1 = float(t[i + 1])
            out.append(t0)
            k_i = base + (1 if i < rem else 0)
            if k_i > 0:
                gap = (t1 - t0) / float(k_i + 1)
                for j in range(1, k_i + 1):
                    out.append(t0 + float(j) * gap)
        out.append(float(t[-1]))
        return np.asarray(out, dtype=float)

    k = int(max(points_between, 0))
    if k <= 0:
        return t

    out: List[float] = []
    for i in range(n_intervals):
        t0 = float(t[i])
        t1 = float(t[i + 1])
        out.append(t0)
        gap = (t1 - t0) / float(k + 1)
        for j in range(1, k + 1):
            out.append(t0 + float(j) * gap)
    out.append(float(t[-1]))
    return np.asarray(out, dtype=float)


def _densify_wide_by_time(
    *,
    wide_df: pd.DataFrame,
    real_times: Sequence[float],
    dense_times: Sequence[float],
) -> pd.DataFrame:
    real = [float(x) for x in real_times]
    dense = [float(x) for x in dense_times]
    if len(real) == 0 or len(dense) == 0:
        return wide_df
    if len(real) == len(dense) and bool(np.allclose(np.asarray(real), np.asarray(dense))):
        return wide_df

    real_cols = [f"t={float(t):g}" for t in real]
    mat = wide_df[real_cols].to_numpy(dtype=float)
    dense_mat = np.full((mat.shape[0], len(dense)), np.nan, dtype=float)

    x_real = np.asarray(real, dtype=float)
    x_dense = np.asarray(dense, dtype=float)
    for i in range(mat.shape[0]):
        row = np.asarray(mat[i], dtype=float)
        finite = np.isfinite(row)
        n_finite = int(np.sum(finite))
        if n_finite == 0:
            continue
        if n_finite == 1:
            dense_mat[i, :] = float(row[finite][0])
            continue
        xp = x_real[finite]
        fp = row[finite]
        dense_mat[i, :] = np.interp(x_dense, xp, fp)

    out = wide_df[["pair", "primary_var", "secondary_var"]].copy()
    for j, t in enumerate(dense):
        out[f"t={float(t):g}"] = dense_mat[:, j]
    return out


def _wide_to_long(wide_df: pd.DataFrame, times: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    time_cols = [f"t={float(t):g}" for t in times]
    for _, row in wide_df.iterrows():
        pair = str(row["pair"])
        src = str(row["primary_var"])
        dst = str(row["secondary_var"])
        for t, c in zip(times, time_cols):
            rows.append(
                {
                    "pair": pair,
                    "time": float(t),
                    "weight": float(row[c]) if np.isfinite(float(row[c])) else np.nan,
                    "primary_var": src,
                    "secondary_var": dst,
                }
            )
    return pd.DataFrame(rows)


def _impute_row_nan(row: np.ndarray, *, nan_strategy: str, col_fill: np.ndarray) -> np.ndarray:
    x = np.asarray(row, dtype=float).copy()
    finite = np.isfinite(x)
    if bool(np.all(finite)):
        return x

    strategy = str(nan_strategy).strip().lower()
    if strategy == "zero":
        x[~finite] = 0.0
        return x

    if strategy == "col_mean":
        x[~finite] = col_fill[~finite]
        return x

    # default: row_mean
    row_mean = float(np.nanmean(x)) if bool(np.isfinite(x).any()) else 0.0
    if not np.isfinite(row_mean):
        row_mean = 0.0
    x[~finite] = row_mean
    return x


def _compute_row_order(
    *,
    wide_df: pd.DataFrame,
    times: Sequence[float],
    row_cluster: bool,
    cluster_method: str,
    cluster_metric: str,
    nan_strategy: str,
) -> Tuple[List[int], Dict[str, Any]]:
    cols = [f"t={float(t):g}" for t in times]
    mat = wide_df[cols].to_numpy(dtype=float)
    n_rows = int(mat.shape[0])
    default_order = list(range(n_rows))

    meta: Dict[str, Any] = {
        "enabled": bool(row_cluster),
        "applied": False,
        "fallback": False,
        "fallback_reason": "",
        "method": str(cluster_method),
        "metric": str(cluster_metric),
        "nan_strategy": str(nan_strategy),
        "n_rows": int(n_rows),
        "n_clustered_rows": 0,
        "n_constant_rows": 0,
        "n_all_nan_rows": 0,
    }

    if not bool(row_cluster):
        meta["fallback_reason"] = "row clustering disabled"
        return default_order, meta
    if n_rows < 2:
        meta["fallback_reason"] = "too few rows (<2)"
        return default_order, meta
    if linkage is None or leaves_list is None:
        meta["fallback"] = True
        meta["fallback_reason"] = "scipy unavailable"
        return default_order, meta

    col_means = np.nanmean(mat, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    cluster_idx: List[int] = []
    constant_idx: List[int] = []
    all_nan_idx: List[int] = []
    cluster_rows: List[np.ndarray] = []

    for i in range(n_rows):
        row = np.asarray(mat[i], dtype=float)
        finite = np.isfinite(row)
        if int(np.sum(finite)) == 0:
            all_nan_idx.append(i)
            continue

        filled = _impute_row_nan(row, nan_strategy=str(nan_strategy), col_fill=col_means)
        if bool(np.nanstd(filled) < 1e-12):
            constant_idx.append(i)
            continue

        cluster_idx.append(i)
        cluster_rows.append(np.asarray(filled, dtype=float))

    meta["n_clustered_rows"] = int(len(cluster_idx))
    meta["n_constant_rows"] = int(len(constant_idx))
    meta["n_all_nan_rows"] = int(len(all_nan_idx))

    if len(cluster_idx) < 2:
        meta["fallback_reason"] = "too few valid rows for clustering"
        return default_order, meta

    try:
        X = np.asarray(cluster_rows, dtype=float)
        Z = linkage(X, method=str(cluster_method), metric=str(cluster_metric), optimal_ordering=True)
        leaf_local = leaves_list(Z).astype(int).tolist()
        ordered_cluster = [int(cluster_idx[j]) for j in leaf_local]
        ordered = ordered_cluster + list(constant_idx) + list(all_nan_idx)
        if len(ordered) != n_rows:
            meta["fallback"] = True
            meta["fallback_reason"] = "internal row count mismatch"
            return default_order, meta
        meta["applied"] = True
        return ordered, meta
    except Exception as exc:
        meta["fallback"] = True
        meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return default_order, meta


def _plot_heatmap_abs(
    *,
    wide_df: pd.DataFrame,
    times: Sequence[float],
    output_png: Path,
    vmax: float,
    title: str = "Primary->Secondary Jacobian Heatmap (Absolute Weights)",
) -> None:
    cols = [f"t={float(t):g}" for t in times]
    mat = wide_df[cols].to_numpy(dtype=float)
    labels = wide_df["pair"].astype(str).tolist()

    fig_h = max(4.8, 0.33 * len(labels) + 2.0)
    plt.figure(figsize=(10.0, fig_h), dpi=240)
    im = plt.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-float(vmax), vmax=float(vmax), interpolation="nearest")
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)
    plt.xticks(np.arange(len(cols)), [f"{float(t):g}" for t in times], fontsize=9)
    plt.xlabel("time")
    plt.ylabel("relation")
    plt.title(str(title))
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.set_label("Jacobian weight")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()


def _plot_heatmap_row_zscore(
    *,
    wide_df: pd.DataFrame,
    times: Sequence[float],
    output_png: Path,
    title: str = "Primary->Secondary Jacobian Heatmap (Row-wise Z-score)",
) -> float:
    cols = [f"t={float(t):g}" for t in times]
    mat = wide_df[cols].to_numpy(dtype=float)
    mat_z = _row_zscore(mat)
    labels = wide_df["pair"].astype(str).tolist()
    vmax = float(np.nanmax(np.abs(mat_z))) if np.isfinite(mat_z).any() else 1.0
    vmax = max(vmax, 1e-8)

    fig_h = max(4.8, 0.33 * len(labels) + 2.0)
    plt.figure(figsize=(10.0, fig_h), dpi=240)
    im = plt.imshow(mat_z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)
    plt.xticks(np.arange(len(cols)), [f"{float(t):g}" for t in times], fontsize=9)
    plt.xlabel("time")
    plt.ylabel("relation")
    plt.title(str(title))
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.set_label("row z-score")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    return float(vmax)


def build_relation_heatmap(
    ctx: TFRunContext,
    *,
    pairs_df: pd.DataFrame,
    time_points: Sequence[float],
    prefix: str = "grn_main_sub_all_relations_tf",
    norm: str = "subset",
    row_cluster: bool = False,
    cluster_method: str = "average",
    cluster_metric: str = "euclidean",
    nan_strategy: str = "row_mean",
    export_reordered_matrix: bool = True,
    time_dense_keep_real_threshold: int = 10,
    time_dense_points_between: int = 0,
    time_dense_target_points: int = 0,
    direction: str = "primary_to_secondary",
) -> Dict[str, Any]:
    if pairs_df.empty:
        raise ValueError("pairs_df is empty")

    out_dir = ctx.output_figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    direction_norm = str(direction).strip().lower()
    if direction_norm in {"secondary_to_primary", "sub_to_main", "2to1", "reverse"}:
        table_prefix = "grn_sub_to_main"
        pairs_use = (
            pairs_df[["secondary_var", "primary_var"]]
            .rename(columns={"secondary_var": "primary_var", "primary_var": "secondary_var"})
            .copy()
        )
        direction_tag = "secondary_to_primary"
    else:
        table_prefix = "grn_main_to_sub"
        pairs_use = pairs_df[["primary_var", "secondary_var"]].copy()
        direction_tag = "primary_to_secondary"

    real_times = [float(x) for x in time_points]
    time_tables = _load_time_tables([ctx.output_table_dir, ctx.tables_dir], real_times, table_prefix=table_prefix)
    wide_df, long_df = _compute_pair_matrices(
        time_tables=time_tables,
        times=real_times,
        pairs_df=pairs_use,
    )
    if wide_df.empty:
        raise ValueError("no usable pairs for heatmap")

    dense_keep_thr = int(max(time_dense_keep_real_threshold, 2))
    dense_target = int(max(time_dense_target_points, 0))
    resolved_points_between = 0
    if len(real_times) >= dense_keep_thr:
        display_times = list(real_times)
    else:
        if dense_target <= 0:
            resolved_points_between = int(max(time_dense_points_between, 0))
            if resolved_points_between <= 0:
                resolved_points_between = _auto_points_between(len(real_times))
        display_times = _build_dense_time_grid(
            real_times,
            points_between=int(resolved_points_between),
            target_total_points=int(dense_target),
        ).astype(float).tolist()

    wide_df = _densify_wide_by_time(
        wide_df=wide_df,
        real_times=real_times,
        dense_times=display_times,
    )
    long_df = _wide_to_long(wide_df, display_times)

    row_order, cluster_meta = _compute_row_order(
        wide_df=wide_df,
        times=display_times,
        row_cluster=bool(row_cluster),
        cluster_method=str(cluster_method),
        cluster_metric=str(cluster_metric),
        nan_strategy=str(nan_strategy),
    )
    wide_df = wide_df.iloc[row_order].reset_index(drop=True)

    subset_vmax = _subset_vmax(wide_df, display_times)
    full_vmax = _full_vmax(time_tables)
    used_vmax = subset_vmax if str(norm) == "subset" else full_vmax

    abs_png = out_dir / f"{prefix}_heatmap.pdf"
    rowz_png = out_dir / f"{prefix}_heatmap_row_zscore.pdf"
    abs_png = out_dir / f"{prefix}_heatmap.pdf"
    rowz_png = out_dir / f"{prefix}_heatmap_row_zscore.pdf"
    wide_csv = out_dir / f"{prefix}_values_wide.csv"
    wide_reordered_csv = out_dir / f"{prefix}_values_wide_reordered.csv"
    long_csv = out_dir / f"{prefix}_values_long.csv"
    pairs_csv = out_dir / f"{prefix}_pairs_used.csv"
    manifest_json = out_dir / f"{prefix}_manifest.json"

    if direction_tag == "secondary_to_primary":
        abs_title = "Secondary->Primary Jacobian Heatmap (Absolute Weights)"
        rowz_title = "Secondary->Primary Jacobian Heatmap (Row-wise Z-score)"
    else:
        abs_title = "Primary->Secondary Jacobian Heatmap (Absolute Weights)"
        rowz_title = "Primary->Secondary Jacobian Heatmap (Row-wise Z-score)"

    _plot_heatmap_abs(
        wide_df=wide_df,
        times=display_times,
        output_png=abs_png,
        vmax=used_vmax,
        title=abs_title,
    )
    rowz_vmax = _plot_heatmap_row_zscore(
        wide_df=wide_df,
        times=display_times,
        output_png=rowz_png,
        title=rowz_title,
    )

    wide_df.to_csv(wide_csv, index=False)
    if bool(export_reordered_matrix):
        wide_df.to_csv(wide_reordered_csv, index=False)
    long_df.to_csv(long_csv, index=False)
    pairs_use.to_csv(pairs_csv, index=False)

    row_labels = wide_df["pair"].astype(str).tolist()

    manifest = {
        "run_dir": str(ctx.run_dir),
        "direction": direction_tag,
        "table_prefix": table_prefix,
        "times": [float(x) for x in display_times],
        "real_times": [float(x) for x in real_times],
        "time_densify": {
            "enabled": bool(len(display_times) != len(real_times) or not np.allclose(np.asarray(display_times), np.asarray(real_times))),
            "keep_real_threshold": int(dense_keep_thr),
            "target_points": int(dense_target),
            "configured_points_between": int(max(time_dense_points_between, 0)),
            "resolved_points_between": int(resolved_points_between),
            "real_points": int(len(real_times)),
            "display_points": int(len(display_times)),
        },
        "norm_mode": str(norm),
        "row_order": [int(i) for i in row_order],
        "row_labels_reordered": row_labels,
        "row_clustering": cluster_meta,
        "subset_vmax": float(subset_vmax),
        "full_vmax": float(full_vmax),
        "used_vmax": float(used_vmax),
        "row_zscore_vmax": float(rowz_vmax),
        "n_used_pairs": int(pairs_use.shape[0]),
        "outputs": {
            "heatmap_png": str(abs_png),
            "heatmap_row_zscore_png": str(rowz_png),
            "values_wide_csv": str(wide_csv),
            "values_wide_reordered_csv": str(wide_reordered_csv) if bool(export_reordered_matrix) else "",
            "values_long_csv": str(long_csv),
            "pairs_used_csv": str(pairs_csv),
        },
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifest_json"] = str(manifest_json)
    return manifest
