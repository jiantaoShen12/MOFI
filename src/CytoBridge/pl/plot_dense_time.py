from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from .plot_ode_v3 import _build_traj_cmap, _draw_background, _draw_endpoints


ALL_FIGURE_IDS: Sequence[str] = (
    "trajectory",
    "rna_bridging",
    "protein_bridging",
    "cell_type_main",
    "cell_type_heatmap",
    "program",
    "program_both",
    "program_heatmap",
    "cycle_consistency",
    "jacobian_grn",
    "perturbation_fate_shift",
    "perturbation_zscore_sweep",
    "perturbation_sweep_heatmap",
    "perturbation_sweep_heatmap_prob",
    "perturbation_timecurve_focus",
    "key_molecule_dotplot",
    "key_molecule_volcano",
)


DEFAULT_FIGURE_CFG: Dict[str, Any] = {
    "theme": "journal",
    "dpi": 300,
    "marker_top_n": 6,
    "cell_type_top_n": 8,
    "cell_type_exclude": ["HSC"],
    "program_top_n": 12,
    "perturb_max_groups": 4,
    "perturb_max_cell_types": 8,
    "perturb_curve_scores": [-8.0, 0.0, 8.0],
    "perturb_curve_cell_types": ["EryP", "MasP", "MkP", "MoP", "NeuP"],
    "perturb_heatmap_z_order": [],
    "min_fig_width": 8.6,
    "min_fig_height": 4.8,
    "fdr_alpha": 0.05,
    "min_abs_log2fc": 0.25,
    "trajectory_projection": "pca",
    "trajectory_umap_n_neighbors": 30,
    "trajectory_umap_min_dist": 0.2,
    "trajectory_umap_metric": "euclidean",
    "trajectory_umap_random_state": 42,
    "trajectory_n_show_traj": 120,
    "trajectory_traj_cmap": "sunset",
}


def _resolve_optional_z_order(raw: Any) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        vals = [x.strip() for x in raw.split(",") if x.strip()]
        return [float(x) for x in vals]
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [float(x) for x in raw]
    return []


class _FigureSkip(RuntimeError):
    """Expected skip reason for one figure."""


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_figure_cfg(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_FIGURE_CFG)
    if isinstance(raw, dict):
        for key, value in raw.items():
            cfg[key] = value
    return cfg


def _apply_journal_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
        }
    )


def _save_fig(path: str, *, dpi: int) -> None:
    out = Path(path)
    _ensure_parent(out)
    plt.tight_layout()
    plt.savefig(out, dpi=int(dpi), bbox_inches="tight")
    if out.suffix.lower() != ".png":
        plt.savefig(out.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight")
    if out.suffix.lower() != ".pdf":
        plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    if out.suffix.lower() != ".svg":
        plt.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def _pick_colors(n: int, *, cmap_name: str = "viridis") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    return cmap(np.linspace(0.08, 0.94, max(n, 2)))


def _resolve_cell_type_order(cell_types: Sequence[str], cfg: Dict[str, Any]) -> List[str]:
    raw_order = [str(x) for x in cfg.get("cell_type_order", []) if str(x).strip()]
    present = [str(x) for x in cell_types]
    if not raw_order:
        return present
    out = [ct for ct in raw_order if ct in present]
    out.extend([ct for ct in present if ct not in out])
    return out


def _resolve_cell_type_color_lut(cell_types: Sequence[str], cfg: Dict[str, Any], *, cmap_name: str) -> Dict[str, Any]:
    ordered = _resolve_cell_type_order(cell_types, cfg)
    palette_cfg = cfg.get("cell_type_palette", {})
    lut: Dict[str, Any] = {}
    if isinstance(palette_cfg, dict):
        for cell_type in ordered:
            color = palette_cfg.get(cell_type)
            if color is not None:
                lut[cell_type] = color
    unresolved = [ct for ct in ordered if ct not in lut]
    fallback = _pick_colors(len(unresolved), cmap_name=cmap_name)
    for color, cell_type in zip(fallback, unresolved):
        lut[cell_type] = color
    return lut


def _slugify_token(text: object, *, fallback: str = "na") -> str:
    raw = str(text).strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    raw = raw.strip("-")
    return raw if raw else fallback


def _signed_float_token(value: float) -> str:
    txt = f"{float(value):g}"
    txt = txt.replace("-", "m").replace("+", "p").replace(".", "d")
    return _slugify_token(txt, fallback="0")


def _compact_suffix_with_hash(text: str, *, max_len: int = 120) -> str:
    token = _slugify_token(text, fallback="all")
    if len(token) <= max_len:
        return token
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:10]
    head_len = max(16, max_len - len("__h-") - len(digest))
    head = token[:head_len].rstrip("-")
    return f"{head}__h-{digest}"


def _extract_sweep_suffix_from_stem(stem: str) -> str:
    prefix = "perturbation_zscore_sweep__"
    if not stem.startswith(prefix):
        return ""
    raw = stem[len(prefix) :].strip()
    if not raw:
        return ""
    raw = re.sub(r"[^a-zA-Z0-9._+-]+", "-", raw).strip("-._+")
    return raw.lower()


def _extract_focus_curve_suffix_from_stem(stem: str) -> str:
    prefix = "perturbation_timecurve_focus__"
    if not stem.startswith(prefix):
        return ""
    raw = stem[len(prefix) :].strip()
    if not raw:
        return ""
    raw = re.sub(r"[^a-zA-Z0-9._+-]+", "-", raw).strip("-._+")
    return raw.lower()


def _normalize_selected_figures(selected_figures: Optional[Sequence[str]]) -> List[str]:
    if not selected_figures:
        return list(ALL_FIGURE_IDS)
    dedup: List[str] = []
    seen = set()
    for item in selected_figures:
        key = str(item).strip()
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            dedup.append(key)
    unknown = [x for x in dedup if x not in ALL_FIGURE_IDS]
    if unknown:
        raise ValueError(f"Unknown figure ids: {unknown}. Available: {list(ALL_FIGURE_IDS)}")
    return dedup


def _require_file(path: Path, *, context: str) -> Path:
    if not path.exists():
        raise _FigureSkip(f"missing input for {context}: {path.name}")
    return path


def _maybe_ci_bounds(df: pd.DataFrame, low_col: str, high_col: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if low_col not in df.columns or high_col not in df.columns:
        return None
    low = df[low_col].to_numpy(dtype=float)
    high = df[high_col].to_numpy(dtype=float)
    if len(low) == 0 or len(high) == 0:
        return None
    return low, high


def plot_dense_time_trajectories_dual(
    traj_main_path: str,
    traj_sub_path: str,
    time_grid: Sequence[float],
    output_path: str,
    *,
    cfg: Dict[str, Any],
    n_show_traj: int = 120,
) -> None:
    main = np.load(traj_main_path)
    sub = np.load(traj_sub_path)
    if main.ndim != 3 or sub.ndim != 3:
        raise ValueError(f"traj arrays must be 3D: main={main.shape}, sub={sub.shape}")
    if main.shape != sub.shape:
        raise ValueError(f"traj shape mismatch: main={main.shape}, sub={sub.shape}")

    t_num, n_cells, dim = main.shape
    time_arr = np.asarray(time_grid, dtype=float)
    if time_arr.shape[0] != t_num:
        raise ValueError(f"time_grid length {time_arr.shape[0]} != traj time dim {t_num}")

    plot_max_cells = int(cfg.get("trajectory_plot_max_cells", 0) or 0)
    if plot_max_cells > 0 and n_cells > plot_max_cells:
        rng = np.random.default_rng(int(cfg.get("trajectory_umap_random_state", 42)))
        keep = np.sort(rng.choice(n_cells, size=plot_max_cells, replace=False))
        main = main[:, keep, :]
        sub = sub[:, keep, :]
        n_cells = int(plot_max_cells)

    proj_mode = str(cfg.get("trajectory_projection", "pca")).strip().lower()
    if proj_mode not in {"pca", "umap"}:
        raise ValueError(f"trajectory_projection must be 'pca' or 'umap', got: {proj_mode}")

    stacked = np.concatenate([main.reshape(-1, dim), sub.reshape(-1, dim)], axis=0)
    if proj_mode == "pca":
        reducer = PCA(n_components=2, random_state=42)
        reducer.fit(stacked)
        main_2d = reducer.transform(main.reshape(-1, dim)).reshape(t_num, n_cells, 2)
        sub_2d = reducer.transform(sub.reshape(-1, dim)).reshape(t_num, n_cells, 2)
        axis_x = "PC1"
        axis_y = "PC2"
    else:
        try:
            import umap  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError("trajectory_projection=umap requires umap-learn to be installed") from exc
        reducer = umap.UMAP(
            n_neighbors=int(cfg.get("trajectory_umap_n_neighbors", 30)),
            min_dist=float(cfg.get("trajectory_umap_min_dist", 0.2)),
            metric=str(cfg.get("trajectory_umap_metric", "euclidean")),
            n_components=2,
            random_state=int(cfg.get("trajectory_umap_random_state", 42)),
        )
        fit_max_points = int(cfg.get("trajectory_umap_fit_max_points", 0) or 0)
        if fit_max_points > 0 and stacked.shape[0] > fit_max_points:
            rng = np.random.default_rng(int(cfg.get("trajectory_umap_random_state", 42)))
            fit_idx = np.sort(rng.choice(stacked.shape[0], size=fit_max_points, replace=False))
            reducer.fit(stacked[fit_idx])
        else:
            reducer.fit(stacked)
        main_2d = reducer.transform(main.reshape(-1, dim)).reshape(t_num, n_cells, 2)
        sub_2d = reducer.transform(sub.reshape(-1, dim)).reshape(t_num, n_cells, 2)
        axis_x = "UMAP1"
        axis_y = "UMAP2"

    cmap = _build_traj_cmap(str(cfg.get("trajectory_traj_cmap", "sunset")))
    norm = Normalize(vmin=float(np.nanmin(time_arr)), vmax=float(np.nanmax(time_arr)))

    def _draw_panel(ax, arr_2d: np.ndarray, title: str) -> None:
        bg_points_list: List[np.ndarray] = []
        bg_times_list: List[np.ndarray] = []
        for i in range(t_num):
            data_t = arr_2d[i]
            valid_mask_t = ~np.isnan(data_t).any(axis=1)
            valid_t = data_t[valid_mask_t]
            if valid_t.size == 0:
                continue
            bg_points_list.append(valid_t[:, :2])
            bg_times_list.append(np.full(valid_t.shape[0], time_arr[i], dtype=float))
        bg_points = np.vstack(bg_points_list) if bg_points_list else np.empty((0, 2), dtype=float)
        bg_times = np.concatenate(bg_times_list, axis=0) if bg_times_list else None

        _draw_background(
            ax=ax,
            bg_points=bg_points,
            bg_labels=bg_times,
            bg_palette="Set2",
            background_mode="scatter",
            cmap=cmap,
            norm=norm,
            is_time_color=True,
        )

        show_n = min(int(n_show_traj), n_cells)
        arrow_budget = 12
        arrows_drawn = 0
        for j in range(show_n):
            traj = arr_2d[:, j, :]
            valid_mask = ~np.isnan(traj).any(axis=1)
            if valid_mask.sum() == 0:
                continue

            valid_idx = np.where(valid_mask)[0]
            split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
            idx_chunks = np.split(valid_idx, split_points)
            for chunk in idx_chunks:
                if chunk.size < 2:
                    continue
                pts = traj[chunk]
                t_chunk = time_arr[chunk]
                segments = np.stack([pts[:-1], pts[1:]], axis=1)
                seg_t = 0.5 * (t_chunk[:-1] + t_chunk[1:])
                seg_alpha = np.linspace(0.45, 1.0, len(seg_t))

                base_colors = cmap(norm(seg_t))
                glow_colors = base_colors.copy()
                glow_colors[:, 3] = 0.11 * seg_alpha
                main_colors = base_colors.copy()
                main_colors[:, 3] = 0.78 * seg_alpha

                lc_glow = LineCollection(segments, colors=glow_colors, linewidths=3.5, capstyle="round", zorder=2)
                lc_main = LineCollection(segments, colors=main_colors, linewidths=1.25, capstyle="round", zorder=3)
                ax.add_collection(lc_glow)
                ax.add_collection(lc_main)

                if arrows_drawn < arrow_budget and chunk.size >= 8:
                    mid = int(0.50 * (chunk.size - 1))
                    p0 = pts[max(mid - 1, 0)]
                    p1 = pts[min(mid + 1, chunk.size - 1)]
                    arrow_color = cmap(norm(float(t_chunk[mid])))
                    ax.annotate(
                        "",
                        xy=p1,
                        xytext=p0,
                        arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=0, mutation_scale=8, alpha=0.85),
                        zorder=3.4,
                    )
                    arrows_drawn += 1

            start_idx = int(valid_idx[0])
            end_idx = int(valid_idx[-1])
            _draw_endpoints(ax, traj, time_arr, start_idx, end_idx, norm, cmap)

        ax.set_title(title)
        ax.set_xlabel(axis_x)
        ax.set_ylabel(axis_y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    primary_label = str(cfg.get("primary_modality_label", "RNA")).strip() or "RNA"
    secondary_label = str(cfg.get("secondary_modality_label", "Protein")).strip() or "Protein"
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=True, sharey=True)
    _draw_panel(axes[0], main_2d, f"Main Space ({primary_label} latent)")
    _draw_panel(axes[1], sub_2d, f"Mapped Sub Space ({secondary_label} latent)")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Time Point")
    ticks = np.unique(np.asarray(time_grid, dtype=float))
    if ticks.size > 0:
        cbar.set_ticks(ticks)

    out = Path(output_path)
    _ensure_parent(out)
    plt.tight_layout()
    fig.savefig(out, dpi=int(cfg["dpi"]), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=int(cfg["dpi"]), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_intermediate_bridging(
    pred_profile_csv: str,
    real_profile_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
    markers: Optional[Iterable[str]] = None,
) -> None:
    pred = pd.read_csv(pred_profile_csv)
    real = pd.read_csv(real_profile_csv)
    if pred.empty:
        raise _FigureSkip("empty predicted marker profile")

    all_markers = sorted(pred["marker"].astype(str).unique().tolist())
    if markers:
        use_markers = [m for m in markers if str(m) in set(all_markers)]
    else:
        score = (
            pred.groupby("marker", as_index=False)["mean_expr"]
            .apply(lambda x: float(np.max(np.abs(x.to_numpy(dtype=float)))))
            .rename(columns={"mean_expr": "abs_max"})
            .sort_values("abs_max", ascending=False)
        )
        use_markers = score["marker"].astype(str).head(int(cfg["marker_top_n"])).tolist()
    if len(use_markers) == 0:
        raise _FigureSkip("no marker selected for bridging plot")

    cols = min(3, len(use_markers))
    rows = int(np.ceil(len(use_markers) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.5 * rows), squeeze=False)
    data_color = str(cfg.get("bridging_data_color", "#1f77b4"))
    anchor_color = str(cfg.get("bridging_anchor_color", "#d62728"))
    for i, marker in enumerate(use_markers):
        ax = axes[i // cols][i % cols]
        p = pred[pred["marker"].astype(str) == str(marker)].sort_values("time")
        r = real[real["marker"].astype(str) == str(marker)].sort_values("time")
        if p.empty:
            ax.axis("off")
            continue
        x = p["time"].to_numpy(dtype=float)
        y = p["mean_expr"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, label="pred", color=data_color)
        bounds = _maybe_ci_bounds(p, "expr_p25", "expr_p75")
        if bounds is not None:
            ax.fill_between(x, bounds[0], bounds[1], color=data_color, alpha=0.16, linewidth=0.0, label="pred P25-P75")
        if not r.empty:
            ax.scatter(r["time"], r["mean_expr"], marker="X", s=52, color=anchor_color, label="real anchor")
        ax.set_title(str(marker))
        ax.set_xlabel("time")
        ax.set_ylabel("mean expression")

    for j in range(len(use_markers), rows * cols):
        axes[j // cols][j % cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_cell_type_probability_curves(
    cell_type_pred_csv: str,
    cell_type_real_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
) -> None:
    pred = pd.read_csv(cell_type_pred_csv)
    real = pd.read_csv(cell_type_real_csv)
    if pred.empty:
        raise _FigureSkip("empty cell type prediction table")

    exclude_cell_types = {str(x).strip().lower() for x in cfg.get("cell_type_exclude", ["HSC"])}
    if exclude_cell_types:
        pred = pred[~pred["cell_type"].astype(str).str.lower().isin(exclude_cell_types)].copy()
        if not real.empty:
            real = real[~real["cell_type"].astype(str).str.lower().isin(exclude_cell_types)].copy()
    if pred.empty:
        raise _FigureSkip("empty cell type prediction table after exclusion filter")

    top_n = int(cfg["cell_type_top_n"])
    avg = pred.groupby("cell_type", as_index=False)["prob"].mean().sort_values("prob", ascending=False)
    cell_types = avg["cell_type"].astype(str).head(top_n).tolist()
    if len(cell_types) == 0:
        raise _FigureSkip("no cell types available")
    cell_types = _resolve_cell_type_order(cell_types, cfg)

    color_lut = _resolve_cell_type_color_lut(cell_types, cfg, cmap_name="tab10")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for cell_type in cell_types:
        p = pred[pred["cell_type"].astype(str) == cell_type].sort_values("time")
        if p.empty:
            continue
        x = p["time"].to_numpy(dtype=float)
        y = p["prob"].to_numpy(dtype=float)
        ax.plot(x, y, color=color_lut[cell_type], linewidth=2.0, label=f"{cell_type} pred")
        bounds = _maybe_ci_bounds(p, "prob_p25", "prob_p75")
        if bounds is not None:
            ax.fill_between(x, bounds[0], bounds[1], color=color_lut[cell_type], alpha=0.16, linewidth=0.0)
        r = real[real["cell_type"].astype(str) == cell_type].sort_values("time")
        if not r.empty:
            ax.scatter(r["time"], r["prob"], color=color_lut[cell_type], marker="X", s=42)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("probability")
    ax.set_title("Cell-Type Fate Dynamics")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_cell_type_fate_heatmap(
    cell_type_pred_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
) -> None:
    pred = pd.read_csv(cell_type_pred_csv)
    if pred.empty:
        raise _FigureSkip("empty cell type prediction table")

    exclude_cell_types = {str(x).strip().lower() for x in cfg.get("cell_type_exclude", ["HSC"])}
    if exclude_cell_types:
        pred = pred[~pred["cell_type"].astype(str).str.lower().isin(exclude_cell_types)].copy()
    if pred.empty:
        raise _FigureSkip("empty cell type prediction table after exclusion filter")

    top_n = int(cfg["cell_type_top_n"])
    keep = (
        pred.groupby("cell_type", as_index=False)["prob"]
        .mean()
        .sort_values("prob", ascending=False)["cell_type"]
        .astype(str)
        .head(top_n)
        .tolist()
    )
    keep = _resolve_cell_type_order(keep, cfg)
    show = pred[pred["cell_type"].astype(str).isin(keep)].copy()
    if show.empty:
        raise _FigureSkip("cell type heatmap has no rows")

    mat = (
        show.pivot_table(index="cell_type", columns="time", values="prob", aggfunc="mean")
        .reindex(index=keep)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(max(8.8, 0.75 * mat.shape[1]), max(4.0, 0.35 * mat.shape[0] + 2.0)))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", interpolation="nearest")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.astype(str).tolist())
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"{float(x):g}" for x in mat.columns], rotation=45, ha="right")
    ax.set_xlabel("time")
    ax.set_title("Cell-Type Fate Heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("pred probability")
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_program_dynamics_curves(
    program_pred_csv: str,
    program_real_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
) -> None:
    pred = pd.read_csv(program_pred_csv)
    real = pd.read_csv(program_real_csv)
    if pred.empty:
        raise _FigureSkip("empty program prediction table")

    rank = (
        pred.groupby("program", as_index=False)["score"]
        .apply(lambda x: float(np.max(np.abs(x.to_numpy(dtype=float)))))
        .rename(columns={"score": "abs_max"})
        .sort_values("abs_max", ascending=False)
    )
    top_n = int(cfg["program_top_n"])
    programs = rank["program"].astype(str).head(top_n).tolist()
    if len(programs) == 0:
        raise _FigureSkip("no programs to plot")

    colors = _pick_colors(len(programs), cmap_name="tab20")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for color, program in zip(colors, programs):
        p = pred[pred["program"].astype(str) == program].sort_values("time")
        if p.empty:
            continue
        x = p["time"].to_numpy(dtype=float)
        y = p["score"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linewidth=1.9, label=f"{program} pred")
        bounds = _maybe_ci_bounds(p, "score_p25", "score_p75")
        if bounds is not None:
            ax.fill_between(x, bounds[0], bounds[1], color=color, alpha=0.16, linewidth=0.0)
        r = real[real["program"].astype(str) == program].sort_values("time")
        if not r.empty:
            ax.scatter(r["time"], r["score"], color=color, marker="X", s=42)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("time")
    ax.set_ylabel("program score (z)")
    ax.set_title("Mechanism Program Dynamics")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def _merge_program_space_tables(main_df: pd.DataFrame, sec_df: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    def _normalize_program_quantiles(df: pd.DataFrame, *, space_name: str) -> Optional[pd.DataFrame]:
        if df.empty:
            return None
        tmp = df.copy()
        low_col = "score_p25" if "score_p25" in tmp.columns else ("score_p40" if "score_p40" in tmp.columns else None)
        mid_col = "score_p50" if "score_p50" in tmp.columns else None
        high_col = "score_p75" if "score_p75" in tmp.columns else ("score_p60" if "score_p60" in tmp.columns else None)

        if low_col is None:
            tmp["score_p25"] = np.nan
        elif low_col != "score_p25":
            tmp["score_p25"] = tmp[low_col]

        if mid_col is None:
            tmp["score_p50"] = np.nan

        if high_col is None:
            tmp["score_p75"] = np.nan
        elif high_col != "score_p75":
            tmp["score_p75"] = tmp[high_col]

        tmp["space"] = space_name
        return tmp

    main_norm = _normalize_program_quantiles(main_df, space_name="main")
    sec_norm = _normalize_program_quantiles(sec_df, space_name="sec")
    if main_norm is not None:
        frames.append(main_norm)
    if sec_norm is not None:
        frames.append(sec_norm)
    if not frames:
        return pd.DataFrame(columns=["time", "program", "score", "score_p25", "score_p50", "score_p75", "n_cells", "source"])

    merged = pd.concat(frames, axis=0, ignore_index=True)
    agg = (
        merged.groupby(["time", "program", "source"], as_index=False)
        .agg(
            {
                "score": "mean",
                "score_p25": "mean",
                "score_p50": "mean",
                "score_p75": "mean",
                "n_cells": "sum",
            }
        )
        .sort_values(["program", "time", "source"])
    )
    return agg


def plot_program_dynamics_both_curves(
    program_pred_csv: str,
    program_real_csv: str,
    program_sec_pred_csv: str,
    program_sec_real_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
) -> None:
    pred_main = pd.read_csv(program_pred_csv)
    real_main = pd.read_csv(program_real_csv)
    pred_sec = pd.read_csv(program_sec_pred_csv)
    real_sec = pd.read_csv(program_sec_real_csv)
    pred = _merge_program_space_tables(pred_main, pred_sec)
    real = _merge_program_space_tables(real_main, real_sec)
    if pred.empty:
        raise _FigureSkip("empty main and secondary program prediction tables")

    rank = (
        pred.groupby("program", as_index=False)["score"]
        .apply(lambda x: float(np.max(np.abs(x.to_numpy(dtype=float)))))
        .rename(columns={"score": "abs_max"})
        .groupby("program", as_index=False)["abs_max"]
        .max()
        .sort_values("abs_max", ascending=False)
    )
    top_n = int(cfg["program_top_n"])
    programs = rank["program"].astype(str).head(top_n).tolist()
    if len(programs) == 0:
        raise _FigureSkip("no programs to plot")

    colors = _pick_colors(len(programs), cmap_name="tab20")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for color, program in zip(colors, programs):
        p = pred[pred["program"].astype(str) == program].sort_values("time")
        if p.empty:
            continue
        x = p["time"].to_numpy(dtype=float)
        y = p["score"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linewidth=1.9, label=f"{program} pred")
        bounds = _maybe_ci_bounds(p, "score_p25", "score_p75")
        if bounds is not None:
            ax.fill_between(x, bounds[0], bounds[1], color=color, alpha=0.16, linewidth=0.0)
        r = real[real["program"].astype(str) == program].sort_values("time")
        if not r.empty:
            ax.scatter(r["time"], r["score"], color=color, marker="X", s=42)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("time")
    ax.set_ylabel("program score (z)")
    ax.set_title("Mechanism Program Dynamics (Merged Spaces)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_program_dynamics_heatmap(
    program_pred_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
) -> None:
    pred = pd.read_csv(program_pred_csv)
    if pred.empty:
        raise _FigureSkip("empty program prediction table")

    rank = (
        pred.groupby("program", as_index=False)["score"]
        .apply(lambda x: float(np.max(np.abs(x.to_numpy(dtype=float)))))
        .rename(columns={"score": "abs_max"})
        .sort_values("abs_max", ascending=False)
    )
    top_n = int(cfg["program_top_n"])
    programs = rank["program"].astype(str).head(top_n).tolist()
    show = pred[pred["program"].astype(str).isin(programs)].copy()
    if show.empty:
        raise _FigureSkip("program heatmap has no rows")

    mat = (
        show.pivot_table(index="program", columns="time", values="score", aggfunc="mean")
        .reindex(index=programs)
        .sort_index(axis=1)
    )
    vmax = float(np.nanmax(np.abs(mat.to_numpy(dtype=float)))) if mat.size else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(max(8.8, 0.75 * mat.shape[1]), max(4.2, 0.35 * mat.shape[0] + 2.2)))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.astype(str).tolist())
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"{float(x):g}" for x in mat.columns], rotation=45, ha="right")
    ax.set_xlabel("time")
    ax.set_title("Program Dynamics Heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("program score")
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_cycle_consistency_over_time(qc_metrics_json: str, output_path: str, *, cfg: Dict[str, Any]) -> None:
    with open(qc_metrics_json, "r", encoding="utf-8") as f:
        qc = json.load(f)
    times = np.asarray(qc.get("time_grid", []), dtype=float)
    rmse = np.asarray(qc.get("cycle_rmse_by_time", []), dtype=float)
    if len(times) == 0 or len(rmse) == 0:
        raise _FigureSkip("missing cycle consistency arrays")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(times, rmse, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xlabel("time")
    ax.set_ylabel("cycle RMSE")
    ax.set_title("Cross-Space Cycle Consistency")
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_jacobian_grn_overview(edges_csv: str, output_path: str, *, cfg: Dict[str, Any], top_n: int = 20) -> None:
    df = pd.read_csv(edges_csv)
    if df.empty:
        raise _FigureSkip("empty jacobian edge table")

    if {"rna_feature", "protein_feature"}.issubset(df.columns):
        src_col, dst_col = "rna_feature", "protein_feature"
    elif {"main_var_feature", "sec_var_feature"}.issubset(df.columns):
        src_col, dst_col = "main_var_feature", "sec_var_feature"
    else:
        raise _FigureSkip("jacobian table does not contain expected feature columns")

    if "time" not in df.columns:
        agg = (
            df.groupby([src_col, dst_col], as_index=False)["weight"]
            .mean()
            .assign(abs_weight=lambda d: d["weight"].abs())
            .sort_values("abs_weight", ascending=False)
            .head(top_n)
        )
        labels = agg[src_col].astype(str) + " -> " + agg[dst_col].astype(str)
        fig, ax = plt.subplots(figsize=(9.2, max(3.0, 0.35 * len(agg))))
        ax.barh(labels.iloc[::-1], agg["weight"].iloc[::-1], color="#f58518")
        ax.set_xlabel("mean Jacobian coupling weight")
        ax.set_title("Jacobian-based GRN (Top Edges)")
        _save_fig(output_path, dpi=int(cfg["dpi"]))
        return

    has_source = "source" in df.columns
    group_cols = ["source", "time"] if has_source else ["time"]
    group_items = list(df.groupby(group_cols, dropna=False, sort=True))
    if len(group_items) == 0:
        raise _FigureSkip("no grouped jacobian rows")

    n_panels = len(group_items)
    n_cols = 1 if n_panels == 1 else 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.1 * n_cols, max(3.5, 2.9 * n_rows)),
        squeeze=False,
    )

    for idx, (grp_key, sub) in enumerate(group_items):
        ax = axes[idx // n_cols][idx % n_cols]
        agg = (
            sub.groupby([src_col, dst_col], as_index=False)["weight"]
            .mean()
            .assign(abs_weight=lambda d: d["weight"].abs())
            .sort_values("abs_weight", ascending=False)
            .head(top_n)
        )
        if agg.empty:
            ax.axis("off")
            continue

        labels = agg[src_col].astype(str) + " -> " + agg[dst_col].astype(str)
        vals = agg["weight"].to_numpy(dtype=float)
        colors = ["#2ca02c" if x >= 0 else "#d62728" for x in vals]
        ax.barh(labels.iloc[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Jacobian coupling weight")

        if has_source:
            if isinstance(grp_key, tuple):
                source, t = grp_key
            else:
                source, t = "unknown", grp_key
            title = f"{str(source)} | t={float(t):g}"
        else:
            t = grp_key[-1] if isinstance(grp_key, tuple) else grp_key
            title = f"t={float(t):g}"
        ax.set_title(title)

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Jacobian-based GRN by Time Point", fontsize=12)
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_perturbation_fate_shift(
    fate_main_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
    top_n: int = 15,
) -> None:
    df = pd.read_csv(fate_main_csv)
    if df.empty:
        raise _FigureSkip("empty perturbation fate table")

    agg = (
        df.groupby(["tag", "cell_type"], as_index=False)["delta_prob"]
        .mean()
        .assign(abs_shift=lambda d: d["delta_prob"].abs())
        .sort_values("abs_shift", ascending=False)
        .head(top_n)
    )
    labels = agg["tag"].astype(str) + " | " + agg["cell_type"].astype(str)
    fig, ax = plt.subplots(figsize=(9.6, max(3.0, 0.33 * len(agg))))
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in agg["delta_prob"]]
    ax.barh(labels.iloc[::-1], agg["delta_prob"].iloc[::-1], color=colors[::-1])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("delta probability")
    ax.set_title("Perturbation Fate Shift")
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def _load_sweep_df(sweep_csv: str) -> pd.DataFrame:
    df = pd.read_csv(sweep_csv)
    if df.empty:
        raise _FigureSkip(f"empty sweep table: {Path(sweep_csv).name}")
    if "cell_type" not in df.columns or "delta_rate" not in df.columns:
        raise _FigureSkip(f"missing sweep columns in {Path(sweep_csv).name}")

    selected_mask = df.get("is_selected_cell_type")
    if selected_mask is not None:
        mask = selected_mask.astype(bool)
        if mask.any():
            show = df[mask].copy()
        else:
            show = df.copy()
    else:
        show = df.copy()
    if show.empty:
        raise _FigureSkip(f"no rows after selected-cell-type filtering in {Path(sweep_csv).name}")
    return show


def _load_focus_curve_df(focus_curve_csv: str) -> pd.DataFrame:
    df = pd.read_csv(focus_curve_csv)
    if df.empty:
        raise _FigureSkip(f"empty focus-curve table: {Path(focus_curve_csv).name}")
    required_cols = {"target_group", "cell_type", "z_score", "time", "delta_rate", "perturbed_prob"}
    if not required_cols.issubset(set(df.columns)):
        raise _FigureSkip(
            f"missing required focus-curve columns in {Path(focus_curve_csv).name}: "
            f"need={sorted(required_cols)}"
        )

    selected_mask = df.get("is_selected_cell_type")
    if selected_mask is not None:
        mask = selected_mask.astype(bool)
        if mask.any():
            show = df[mask].copy()
        else:
            show = df.copy()
    else:
        show = df.copy()
    if show.empty:
        raise _FigureSkip(f"no rows after selected-cell-type filtering in {Path(focus_curve_csv).name}")
    return show


def plot_perturbation_zscore_sweep(
    sweep_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
) -> None:
    show = _load_sweep_df(sweep_csv)

    max_groups = int(cfg["perturb_max_groups"])
    max_cell_types = int(cfg["perturb_max_cell_types"])

    group_order = (
        show.groupby("target_group", as_index=False)["delta_rate"]
        .apply(lambda x: float(np.max(np.abs(x))))
        .rename(columns={"delta_rate": "max_abs_rate"})
        .sort_values("max_abs_rate", ascending=False)
    )
    groups = group_order["target_group"].astype(str).head(max_groups).tolist()
    show = show[show["target_group"].astype(str).isin(groups)].copy()
    if show.empty:
        raise _FigureSkip("no rows after perturb target-group selection")

    cell_type_order = (
        show.groupby("cell_type", as_index=False)["delta_rate"]
        .apply(lambda x: float(np.max(np.abs(x))))
        .rename(columns={"delta_rate": "max_abs_rate"})
        .sort_values("max_abs_rate", ascending=False)
    )
    ranked_cell_types = cell_type_order["cell_type"].astype(str).tolist()
    requested_cell_types = [
        str(x).strip()
        for x in cfg.get("perturb_heatmap_cell_type_order", [])
        if str(x).strip()
    ]
    if requested_cell_types:
        available_lut = {name.lower(): name for name in ranked_cell_types}
        cell_types = []
        for requested in requested_cell_types:
            resolved = available_lut.get(requested.lower())
            if resolved is not None and resolved not in cell_types:
                cell_types.append(resolved)
        cell_types.extend(name for name in ranked_cell_types if name not in cell_types)
        cell_types = cell_types[:max_cell_types]
    else:
        cell_types = ranked_cell_types[:max_cell_types]
    cell_types = _resolve_cell_type_order(cell_types, cfg)
    show = show[show["cell_type"].astype(str).isin(cell_types)].copy()
    if show.empty:
        raise _FigureSkip("no rows after perturb cell-type selection")

    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10.2, max(3.6, 2.9 * n_rows)), sharex=True, squeeze=False)
    color_lut = _resolve_cell_type_color_lut(cell_types, cfg, cmap_name="tab20")

    for i, group in enumerate(groups):
        ax = axes[i][0]
        sub = show[show["target_group"].astype(str) == group]
        for cell_type in cell_types:
            s = sub[sub["cell_type"].astype(str) == cell_type].sort_values("z_score")
            if s.empty:
                continue
            ax.plot(
                s["z_score"].to_numpy(dtype=float),
                100.0 * s["delta_rate"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                label=cell_type,
                color=color_lut[cell_type],
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.65)
        ax.set_ylabel("delta rate (%)")
        ax.set_title(f"Perturbation Sweep: {group}")
    axes[-1][0].set_xlabel("z score")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2, fontsize=8)
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_perturbation_sweep_heatmap(
    sweep_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
    value_col: str = "delta_rate",
) -> None:
    show = _load_sweep_df(sweep_csv)
    if value_col not in show.columns:
        raise _FigureSkip(f"missing value_col={value_col} in {Path(sweep_csv).name}")

    max_groups = int(cfg["perturb_max_groups"])
    max_cell_types = int(cfg["perturb_max_cell_types"])

    group_order = (
        show.groupby("target_group", as_index=False)["delta_rate"]
        .apply(lambda x: float(np.max(np.abs(x))))
        .rename(columns={"delta_rate": "max_abs_rate"})
        .sort_values("max_abs_rate", ascending=False)
    )
    groups = group_order["target_group"].astype(str).head(max_groups).tolist()
    show = show[show["target_group"].astype(str).isin(groups)].copy()
    if show.empty:
        raise _FigureSkip("no rows after perturb target-group selection")

    cell_type_order = (
        show.groupby("cell_type", as_index=False)["delta_rate"]
        .apply(lambda x: float(np.max(np.abs(x))))
        .rename(columns={"delta_rate": "max_abs_rate"})
        .sort_values("max_abs_rate", ascending=False)
    )
    ranked_cell_types = cell_type_order["cell_type"].astype(str).tolist()
    requested_cell_types = [
        str(x).strip()
        for x in cfg.get("perturb_heatmap_cell_type_order", [])
        if str(x).strip()
    ]
    if requested_cell_types:
        available_lut = {name.lower(): name for name in ranked_cell_types}
        cell_types = []
        for requested in requested_cell_types:
            resolved = available_lut.get(requested.lower())
            if resolved is not None and resolved not in cell_types:
                cell_types.append(resolved)
        cell_types.extend(name for name in ranked_cell_types if name not in cell_types)
        cell_types = cell_types[:max_cell_types]
    else:
        cell_types = ranked_cell_types[:max_cell_types]
    show = show[show["cell_type"].astype(str).isin(cell_types)].copy()
    if show.empty:
        raise _FigureSkip("no rows after perturb cell-type selection")

    z_present = sorted({float(x) for x in show["z_score"].to_numpy(dtype=float)})
    z_order_req = _resolve_optional_z_order(cfg.get("perturb_heatmap_z_order", []))
    if z_order_req:
        z_grid = [float(x) for x in z_order_req]
        tail = [float(x) for x in z_present if not any(np.isclose(float(x), float(y)) for y in z_grid)]
        z_grid.extend(tail)
    else:
        z_grid = list(z_present)
    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10.4, max(3.8, 2.9 * n_rows)), squeeze=False)

    if value_col == "delta_rate":
        vmax_percent_cfg = cfg.get("perturb_heatmap_vmax_percent", None)
        if vmax_percent_cfg is None:
            vmax_percent = 100.0 * float(np.max(np.abs(show["delta_rate"].to_numpy(dtype=float))))
        else:
            vmax_percent = abs(float(vmax_percent_cfg))
        vmax_percent = max(vmax_percent, 1e-6)

    orientation = str(cfg.get("perturb_heatmap_orientation", "cell_type_by_z")).strip().lower()
    if orientation not in {"cell_type_by_z", "z_by_cell_type"}:
        raise ValueError(f"Unsupported perturb_heatmap_orientation={orientation!r}")

    for i, group in enumerate(groups):
        ax = axes[i][0]
        sub = show[show["target_group"].astype(str) == group]
        mat = (
            sub.pivot_table(index="cell_type", columns="z_score", values=value_col, aggfunc="mean")
            .reindex(index=cell_types, columns=z_grid)
            .fillna(0.0)
        )
        if orientation == "z_by_cell_type":
            mat = mat.T
        if value_col == "delta_rate":
            im = ax.imshow(
                100.0 * mat.to_numpy(dtype=float),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax_percent,
                vmax=vmax_percent,
            )
            cbar_label = "delta rate (%)"
            title = f"Perturbation Sweep Heatmap: {group}"
        else:
            im = ax.imshow(
                mat.to_numpy(dtype=float),
                aspect="auto",
                cmap="YlGnBu",
                vmin=0.0,
                vmax=1.0,
            )
            cbar_label = "perturbed probability"
            title = f"Perturbation Sweep Heatmap (Prob): {group}"
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_xticks(np.arange(mat.shape[1]))
        if orientation == "z_by_cell_type":
            ax.set_xticklabels(mat.columns.astype(str).tolist(), rotation=0)
            yticklabels = [f"{float(x):g}" for x in mat.index]
            xlabel, ylabel = "cell type", "z score"
        else:
            ax.set_xticklabels([f"{float(x):g}" for x in mat.columns], rotation=0)
            yticklabels = mat.index.astype(str).tolist()
            xlabel, ylabel = "z score", "cell type"
        ax.set_yticklabels(yticklabels)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label)

    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_perturbation_sweep_heatmap_prob(
    sweep_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
) -> None:
    plot_perturbation_sweep_heatmap(
        sweep_csv,
        output_path,
        cfg=cfg,
        value_col="perturbed_prob",
    )


def plot_perturbation_timecurve_focus(
    focus_curve_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
    target_group: str,
    z_score: float,
) -> None:
    show = _load_focus_curve_df(focus_curve_csv)
    sub = show[
        (show["target_group"].astype(str) == str(target_group))
        & (np.isclose(show["z_score"].to_numpy(dtype=float), float(z_score)))
    ].copy()
    if sub.empty:
        raise _FigureSkip(f"no rows for target_group={target_group}, z_score={z_score:g}")

    focus_types_cfg = cfg.get("perturb_curve_cell_types", ["EryP", "MasP", "MkP", "MoP", "NeuP"])
    focus_types = [str(x) for x in focus_types_cfg if str(x).strip()]
    if focus_types:
        focus_lut = {x.lower() for x in focus_types}
        sub = sub[sub["cell_type"].astype(str).str.lower().isin(focus_lut)].copy()
    if sub.empty:
        raise _FigureSkip(f"no focus cell types for target_group={target_group}, z_score={z_score:g}")

    t_grid = sorted({float(x) for x in sub["time"].to_numpy(dtype=float)})
    if len(t_grid) == 0:
        raise _FigureSkip(f"empty time grid for target_group={target_group}, z_score={z_score:g}")

    cell_types = _resolve_cell_type_order(sorted(sub["cell_type"].astype(str).unique().tolist()), cfg)
    color_lut = _resolve_cell_type_color_lut(cell_types, cfg, cmap_name="tab10")
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for cell_type in cell_types:
        s = sub[sub["cell_type"].astype(str) == cell_type].sort_values("time")
        if s.empty:
            continue
        ax.plot(
            s["time"].to_numpy(dtype=float),
            s["perturbed_prob"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.9,
            label=cell_type,
            color=color_lut[cell_type],
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("cell-type probability")
    ax.set_title(f"Perturbation Timecurve: {target_group} @ z={float(z_score):g}")
    ax.legend(loc="best", frameon=False, ncol=2)
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def plot_key_molecule_dotplot(
    union_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
    top_n: int = 25,
) -> None:
    df = pd.read_csv(union_csv)
    if df.empty:
        raise _FigureSkip("empty key_molecules_union table")

    show = df.head(top_n).copy()
    show["label"] = show["feature"].astype(str) + " (" + show["modality"].astype(str) + ")"
    y = np.arange(len(show))
    size = 25 + 40 * show["time_hits"].to_numpy(dtype=float)
    color = show["best_p_adj"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.1, max(3.0, 0.33 * len(show))))
    sc = ax.scatter(show["mean_log2fc"], y, s=size, c=color, cmap="viridis_r")
    ax.set_yticks(y)
    ax.set_yticklabels(show["label"])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("mean log2FC")
    ax.set_title("Key Path-Specific Molecules")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("best adjusted p")
    _save_fig(output_path, dpi=int(cfg["dpi"]))


def _normalize_gene_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = [x.strip() for x in values.split(",")]
    elif isinstance(values, (list, tuple, np.ndarray, pd.Index)):
        raw = [str(x).strip() for x in values]
    else:
        raw = [str(values).strip()]
    return [x for x in raw if x]


def _to_upper_gene_set(values: Any) -> set:
    return {x.upper() for x in _normalize_gene_list(values)}


def _resolve_volcano_direction_sets(volcano_cfg: Dict[str, Any], space_key: str) -> tuple[set, set]:
    direction_sets = volcano_cfg.get("direction_sets", {})
    space_cfg: Dict[str, Any] = {}
    if isinstance(direction_sets, dict):
        candidate = direction_sets.get(space_key, {})
        if isinstance(candidate, dict):
            space_cfg = candidate

    early_vals = space_cfg.get(
        "early",
        volcano_cfg.get(f"early_set_{space_key}", volcano_cfg.get("early_set", [])),
    )
    late_vals = space_cfg.get(
        "late",
        volcano_cfg.get(f"late_set_{space_key}", volcano_cfg.get("late_set", [])),
    )
    return _to_upper_gene_set(early_vals), _to_upper_gene_set(late_vals)


def _resolve_volcano_prior_set(volcano_cfg: Dict[str, Any], space_key: str) -> set:
    prior_sets = volcano_cfg.get("prior_sets", {})
    if isinstance(prior_sets, dict):
        per_space = prior_sets.get(space_key, [])
        if per_space:
            return _to_upper_gene_set(per_space)
    return _to_upper_gene_set(volcano_cfg.get("prior_genes", []))


def _noise_patterns_from_cfg(volcano_cfg: Dict[str, Any]) -> tuple[List[re.Pattern], set]:
    raw_patterns = volcano_cfg.get("noise_prefix_regex", [])
    patterns: List[re.Pattern] = []
    for raw in _normalize_gene_list(raw_patterns):
        try:
            patterns.append(re.compile(raw, flags=re.IGNORECASE))
        except re.error:
            continue
    noise_exact = _to_upper_gene_set(volcano_cfg.get("noise_exact_genes", []))
    return patterns, noise_exact


def _select_directional_labels(
    show: pd.DataFrame,
    *,
    side: str,
    top_n_per_side: int,
    require_significant: bool,
    exclude_genes_upper: set,
    filter_noise_labels: bool,
    noise_patterns: List[re.Pattern],
    noise_exact_upper: set,
    rank_mode: str,
    prior_genes_upper: set,
) -> pd.DataFrame:
    sub = show.copy()
    sub = sub[np.isfinite(sub["log2fc"]) & np.isfinite(sub["p_adj"]) & np.isfinite(sub["neglog10p"])].copy()
    if require_significant and "is_significant" in sub.columns:
        sub = sub[sub["is_significant"].astype(bool)].copy()
    if side == "left":
        sub = sub[sub["log2fc"] < 0].copy()
    else:
        sub = sub[sub["log2fc"] > 0].copy()
    if exclude_genes_upper:
        sub = sub[~sub["feature"].astype(str).str.upper().isin(exclude_genes_upper)].copy()
    if filter_noise_labels and not sub.empty:
        feature_upper = sub["feature"].astype(str).str.upper()
        drop = feature_upper.isin(noise_exact_upper)
        if noise_patterns:
            feature_raw = sub["feature"].astype(str)
            for pat in noise_patterns:
                drop = drop | feature_raw.str.match(pat)
        sub = sub[~drop].copy()
    if sub.empty:
        return sub
    sub["abs_log2fc"] = sub["log2fc"].abs()
    if rank_mode == "bio_soft" and prior_genes_upper:
        sub["prior_hit"] = sub["feature"].astype(str).str.upper().isin(prior_genes_upper)
        sub = sub.sort_values(["prior_hit", "p_adj", "abs_log2fc"], ascending=[False, True, False])
    else:
        sub = sub.sort_values(["p_adj", "abs_log2fc"], ascending=[True, False])
    sub = sub.drop_duplicates(subset=["feature"], keep="first")
    return sub.head(max(1, int(top_n_per_side))).copy()


def _spread_tied_y_values(
    y_vals: np.ndarray,
    x_vals: np.ndarray,
    *,
    jitter: float,
) -> np.ndarray:
    y = np.asarray(y_vals, dtype=float).copy()
    if y.size <= 1 or jitter <= 0.0:
        return y
    # Deterministically spread exact ties to avoid visual top-line piling.
    uniq, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    for gid, cnt in enumerate(counts):
        if cnt <= 1:
            continue
        idx = np.where(inv == gid)[0]
        if idx.size <= 1:
            continue
        order = idx[np.argsort(np.asarray(x_vals, dtype=float)[idx])]
        offsets = np.linspace(-float(jitter), float(jitter), num=order.size)
        y[order] = y[order] + offsets
    return y


def _format_volcano_tick_label(value: float) -> str:
    val = float(value)
    if abs(val - round(val)) <= 1e-8:
        return str(int(round(val)))
    if abs(val) >= 10.0:
        return f"{val:.1f}"
    return f"{val:.2f}".rstrip("0").rstrip(".")


def _build_piecewise_volcano_y_axis(
    y_values: Sequence[np.ndarray],
    *,
    y_cut: float,
    lower_ratio: float = 2.0,
    upper_ratio: float = 5.0,
    y_max_override: Optional[float] = None,
    y_max_padding: float = 0.0,
) -> Dict[str, Any]:
    cut = max(float(y_cut), 1e-12)
    lower_frac = float(lower_ratio) / float(lower_ratio + upper_ratio)
    upper_frac = 1.0 - lower_frac

    if y_max_override is not None:
        y_max = max(float(y_max_override), cut)
    else:
        y_max = cut
        for arr in y_values:
            vals = np.asarray(arr, dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size > 0:
                y_max = max(y_max, float(np.max(finite)))
        y_max = max(y_max, cut) + max(float(y_max_padding), 0.0)

    has_upper = bool(y_max > cut + 1e-12)

    def _map(vals: np.ndarray) -> np.ndarray:
        arr = np.asarray(vals, dtype=float)
        arr = np.clip(arr, 0.0, y_max)
        out = np.zeros_like(arr, dtype=float)
        if not has_upper:
            denom = max(cut, 1e-12)
            out = np.clip(arr, 0.0, None) / denom
            return np.clip(out, 0.0, 1.0)
        lower_mask = arr <= cut
        out[lower_mask] = lower_frac * np.clip(arr[lower_mask], 0.0, None) / cut
        upper_denom = max(y_max - cut, 1e-12)
        out[~lower_mask] = lower_frac + upper_frac * (arr[~lower_mask] - cut) / upper_denom
        return np.clip(out, 0.0, 1.0)

    if not has_upper:
        raw_ticks = np.linspace(0.0, cut, num=5)
    else:
        lower_ticks = np.asarray([0.0, 0.5 * cut, cut], dtype=float)
        upper_ticks = np.asarray([0.5 * (cut + y_max), y_max], dtype=float)
        raw_ticks = np.concatenate([lower_ticks, upper_ticks], axis=0)
    raw_ticks = np.asarray(sorted({float(x) for x in raw_ticks if np.isfinite(x)}), dtype=float)
    tick_pos = _map(raw_ticks)
    tick_labels = [_format_volcano_tick_label(x) for x in raw_ticks]

    return {
        "map": _map,
        "tick_positions": tick_pos,
        "tick_labels": tick_labels,
        "cut_fraction": (lower_frac if has_upper else float(_map(np.asarray([cut]))[0])),
        "y_max": float(y_max),
        "has_upper": bool(has_upper),
    }


def _draw_volcano_axis_break(ax: plt.Axes, *, y_frac: float) -> None:
    y = float(np.clip(y_frac, 0.0, 1.0))
    dx = 0.018
    dy = 0.018
    kwargs = {
        "transform": ax.transAxes,
        "color": "black",
        "clip_on": False,
        "linewidth": 0.9,
    }
    ax.plot((-dx, +dx), (y - dy, y + dy), **kwargs)
    ax.plot((1.0 - dx, 1.0 + dx), (y - dy, y + dy), **kwargs)


def plot_key_molecule_volcano(
    de_gene_csv: str,
    de_protein_csv: str,
    output_path: str,
    *,
    cfg: Dict[str, Any],
    top_n: int = 16,
) -> None:
    gene = pd.read_csv(de_gene_csv)
    prot = pd.read_csv(de_protein_csv)
    if gene.empty and prot.empty:
        raise _FigureSkip("both DE tables are empty")

    fdr_alpha = float(cfg["fdr_alpha"])
    min_abs_log2fc = float(cfg["min_abs_log2fc"])
    volcano_cfg = cfg.get("volcano_labeling", {})
    if not isinstance(volcano_cfg, dict):
        volcano_cfg = {}
    p_col_for_y = str(volcano_cfg.get("p_col_for_y", "p_adj")).strip()
    if p_col_for_y not in {"p_adj", "p_value"}:
        p_col_for_y = "p_adj"
    y_jitter_ties = float(volcano_cfg.get("y_jitter_ties", 0.0))
    directional_exclusion = bool(volcano_cfg.get("exclude_opposite_direction_only", True))
    require_significant = bool(volcano_cfg.get("require_significant", True))
    top_n_per_side = int(volcano_cfg.get("top_n_per_side", max(1, int(top_n) // 2)))
    filter_noise_labels = bool(volcano_cfg.get("filter_noise_labels", False))
    rank_mode = str(volcano_cfg.get("label_rank_mode", "default")).strip().lower()
    if rank_mode not in {"default", "bio_soft"}:
        rank_mode = "default"
    noise_patterns, noise_exact_upper = _noise_patterns_from_cfg(volcano_cfg)
    fallback_policy = str(volcano_cfg.get("fallback_policy", "keep_short")).strip().lower()
    if fallback_policy not in {"keep_short"}:
        fallback_policy = "keep_short"
    y_cut_override = float(volcano_cfg.get("axis_break_cut", 16.0))
    y_max_override_raw = volcano_cfg.get("axis_break_ymax", None)
    if y_max_override_raw is None or str(y_max_override_raw).strip().lower() in {"", "none", "auto", "data"}:
        y_max_override = None
    else:
        y_max_override = float(y_max_override_raw)

    dfs: List[pd.DataFrame] = []
    for src in [gene, prot]:
        if src.empty:
            dfs.append(src.copy())
            continue
        show = src.copy()
        p_src = show[p_col_for_y] if p_col_for_y in show.columns else show["p_adj"]
        p_raw = pd.to_numeric(p_src, errors="coerce").to_numpy(dtype=float)
        finite_pos = p_raw[np.isfinite(p_raw) & (p_raw > 0.0)]
        if finite_pos.size > 0:
            floor = max(float(np.min(finite_pos)) * 0.5, float(np.finfo(np.float64).tiny))
        else:
            floor = float(np.finfo(np.float64).tiny)
        p_plot = np.where(np.isfinite(p_raw) & (p_raw > 0.0), p_raw, floor)
        show["neglog10p"] = -np.log10(p_plot)
        if y_jitter_ties > 0.0:
            show["neglog10p"] = _spread_tied_y_values(
                show["neglog10p"].to_numpy(dtype=float),
                show["log2fc"].to_numpy(dtype=float),
                jitter=y_jitter_ties,
            )
        dfs.append(show)

    y_cut = float(y_cut_override)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=False)
    for ax, show, title, space_key in [
        (axes[0], dfs[0], "Main-Var DE", "main"),
        (axes[1], dfs[1], "Sec-Var DE", "sec"),
    ]:
        if show.empty:
            ax.axis("off")
            continue
        axis_spec = _build_piecewise_volcano_y_axis(
            [show["neglog10p"].to_numpy(dtype=float)],
            y_cut=float(y_cut),
            lower_ratio=2.0,
            upper_ratio=5.0,
            y_max_override=y_max_override,
            y_max_padding=0.2,
        )
        sig = show["is_significant"].astype(bool)
        y_plot = axis_spec["map"](show["neglog10p"].to_numpy(dtype=float))
        show["neglog10p_plot"] = y_plot

        ax.scatter(show.loc[~sig, "log2fc"], show.loc[~sig, "neglog10p_plot"], s=9, alpha=0.30, color="#9ecae1")
        ax.scatter(show.loc[sig, "log2fc"], show.loc[sig, "neglog10p_plot"], s=12, alpha=0.72, color="#e6550d")

        if directional_exclusion:
            early_set, late_set = _resolve_volcano_direction_sets(volcano_cfg, space_key)
            prior_set = _resolve_volcano_prior_set(volcano_cfg, space_key)
            left = _select_directional_labels(
                show,
                side="left",
                top_n_per_side=top_n_per_side,
                require_significant=require_significant,
                exclude_genes_upper=late_set,
                filter_noise_labels=filter_noise_labels,
                noise_patterns=noise_patterns,
                noise_exact_upper=noise_exact_upper,
                rank_mode=rank_mode,
                prior_genes_upper=prior_set,
            )
            right = _select_directional_labels(
                show,
                side="right",
                top_n_per_side=top_n_per_side,
                require_significant=require_significant,
                exclude_genes_upper=early_set,
                filter_noise_labels=filter_noise_labels,
                noise_patterns=noise_patterns,
                noise_exact_upper=noise_exact_upper,
                rank_mode=rank_mode,
                prior_genes_upper=prior_set,
            )
            if fallback_policy == "keep_short":
                top = pd.concat([left, right], axis=0, ignore_index=True)
            else:
                top = pd.concat([left, right], axis=0, ignore_index=True)
        else:
            top = show.sort_values("p_adj", ascending=True).head(top_n)

        for _, row in top.iterrows():
            ax.text(float(row["log2fc"]), float(row["neglog10p_plot"]), str(row["feature"]), fontsize=7)

        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.axvline(min_abs_log2fc, color="#555555", linewidth=0.7, linestyle="--", alpha=0.8)
        ax.axvline(-min_abs_log2fc, color="#555555", linewidth=0.7, linestyle="--", alpha=0.8)
        ax.axhline(float(axis_spec["cut_fraction"]), color="#555555", linewidth=0.7, linestyle="--", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("log2FC")
        ax.set_ylabel("-log10(adj p)")
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(axis_spec["tick_positions"])
        ax.set_yticklabels(axis_spec["tick_labels"])
        if axis_spec["has_upper"]:
            _draw_volcano_axis_break(ax, y_frac=float(axis_spec["cut_fraction"]))

    _save_fig(output_path, dpi=int(cfg["dpi"]))


def _valid_sweep_candidates(tables: Path) -> List[Path]:
    candidates = sorted(
        tables.glob("perturbation_zscore_sweep__*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    valid: List[Path] = []
    for cand in candidates:
        try:
            cols = pd.read_csv(cand, nrows=0).columns
        except Exception:
            continue
        if "cell_type" in cols and "delta_rate" in cols:
            valid.append(cand)
    return valid


def _valid_focus_curve_candidates(tables: Path) -> List[Path]:
    candidates = sorted(
        tables.glob("perturbation_timecurve_focus__*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    valid: List[Path] = []
    required_cols = {"target_group", "cell_type", "z_score", "time", "delta_rate", "perturbed_prob"}
    for cand in candidates:
        try:
            cols = set(pd.read_csv(cand, nrows=0).columns)
        except Exception:
            continue
        if required_cols.issubset(cols):
            valid.append(cand)
    return valid


def generate_dense_time_report_figures(
    run_dir: str,
    selected_figures: Optional[Sequence[str]] = None,
    figure_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _resolve_figure_cfg(figure_cfg)
    if str(cfg.get("theme", "journal")).strip().lower() == "journal":
        _apply_journal_theme()

    root = Path(run_dir)
    assets = root / "assets"
    tables = root / "tables"
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    requested = _normalize_selected_figures(selected_figures)
    manifest: Dict[str, Any] = {
        "requested_figures": requested,
        "generated_figures": {},
        "skipped_figures": {},
        "failed_figures": {},
    }

    qc_path = assets / "qc_metrics.json"
    if qc_path.exists():
        with qc_path.open("r", encoding="utf-8") as f:
            qc = json.load(f)
        time_grid = qc.get("time_grid", [])
    else:
        time_grid = []

    def _run(fig_id: str, fn) -> None:
        try:
            result = fn()
            manifest["generated_figures"][fig_id] = result
        except _FigureSkip as exc:
            manifest["skipped_figures"][fig_id] = str(exc)
        except Exception as exc:  # pragma: no cover - defensive logging path
            manifest["failed_figures"][fig_id] = f"{type(exc).__name__}: {exc}"

    for fig_id in requested:
        if fig_id == "trajectory":
            def _build() -> str:
                _require_file(assets / "traj_main.npy", context=fig_id)
                _require_file(assets / "traj_sub.npy", context=fig_id)
                if not time_grid:
                    raise _FigureSkip("missing time_grid in qc_metrics.json")
                p = figs / "dense_time_dual_trajectory.pdf"
                plot_dense_time_trajectories_dual(
                    str(assets / "traj_main.npy"),
                    str(assets / "traj_sub.npy"),
                    time_grid,
                    str(p),
                    cfg=cfg,
                )
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "rna_bridging":
            def _build() -> str:
                _require_file(tables / "main_var_marker_pred.csv", context=fig_id)
                _require_file(tables / "main_var_marker_real.csv", context=fig_id)
                p = figs / "rna_intermediate_bridging.pdf"
                local_cfg = dict(cfg)
                local_cfg["bridging_data_color"] = "#5892C0"
                local_cfg["bridging_anchor_color"] = "#000000"
                plot_intermediate_bridging(
                    str(tables / "main_var_marker_pred.csv"),
                    str(tables / "main_var_marker_real.csv"),
                    str(p),
                    cfg=local_cfg,
                )
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "protein_bridging":
            def _build() -> str:
                _require_file(tables / "sec_var_marker_pred.csv", context=fig_id)
                _require_file(tables / "sec_var_marker_real.csv", context=fig_id)
                p = figs / "protein_intermediate_bridging.pdf"
                local_cfg = dict(cfg)
                local_cfg["bridging_data_color"] = "#D65350"
                local_cfg["bridging_anchor_color"] = "#000000"
                plot_intermediate_bridging(
                    str(tables / "sec_var_marker_pred.csv"),
                    str(tables / "sec_var_marker_real.csv"),
                    str(p),
                    cfg=local_cfg,
                )
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "cell_type_main":
            def _build() -> str:
                _require_file(tables / "cell_type_pred_main.csv", context=fig_id)
                _require_file(tables / "cell_type_real_main.csv", context=fig_id)
                p = figs / "cell_type_fate_dynamics.pdf"
                plot_cell_type_probability_curves(
                    str(tables / "cell_type_pred_main.csv"),
                    str(tables / "cell_type_real_main.csv"),
                    str(p),
                    cfg=cfg,
                )
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "cell_type_heatmap":
            def _build() -> str:
                _require_file(tables / "cell_type_pred_main.csv", context=fig_id)
                p = figs / "cell_type_fate_heatmap.pdf"
                plot_cell_type_fate_heatmap(str(tables / "cell_type_pred_main.csv"), str(p), cfg=cfg)
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "program":
            def _build() -> str:
                _require_file(tables / "program_pred.csv", context=fig_id)
                _require_file(tables / "program_real.csv", context=fig_id)
                p = figs / "program_dynamics.pdf"
                plot_program_dynamics_curves(
                    str(tables / "program_pred.csv"),
                    str(tables / "program_real.csv"),
                    str(p),
                    cfg=cfg,
                )
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "program_both":
            def _build() -> str:
                _require_file(tables / "program_pred.csv", context=fig_id)
                _require_file(tables / "program_real.csv", context=fig_id)
                _require_file(tables / "program_sec_pred.csv", context=fig_id)
                _require_file(tables / "program_sec_real.csv", context=fig_id)
                p = figs / "program_dynamics_both.pdf"
                plot_program_dynamics_both_curves(
                    str(tables / "program_pred.csv"),
                    str(tables / "program_real.csv"),
                    str(tables / "program_sec_pred.csv"),
                    str(tables / "program_sec_real.csv"),
                    str(p),
                    cfg=cfg,
                )
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "program_heatmap":
            def _build() -> str:
                _require_file(tables / "program_pred.csv", context=fig_id)
                p = figs / "program_dynamics_heatmap.pdf"
                plot_program_dynamics_heatmap(str(tables / "program_pred.csv"), str(p), cfg=cfg)
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "cycle_consistency":
            def _build() -> str:
                _require_file(assets / "qc_metrics.json", context=fig_id)
                p = figs / "cycle_consistency.pdf"
                plot_cycle_consistency_over_time(str(assets / "qc_metrics.json"), str(p), cfg=cfg)
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "jacobian_grn":
            def _build() -> str:
                _require_file(tables / "jacobian_grn_edges.csv", context=fig_id)
                p = figs / "jacobian_grn_overview.pdf"
                plot_jacobian_grn_overview(str(tables / "jacobian_grn_edges.csv"), str(p), cfg=cfg)
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "perturbation_fate_shift":
            def _build() -> str:
                _require_file(tables / "perturbation_fate_shift_main.csv", context=fig_id)
                p = figs / "perturbation_fate_shift.pdf"
                plot_perturbation_fate_shift(str(tables / "perturbation_fate_shift_main.csv"), str(p), cfg=cfg)
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "perturbation_zscore_sweep":
            def _build() -> List[str]:
                sweep_candidates = _valid_sweep_candidates(tables)
                if not sweep_candidates:
                    raise _FigureSkip("no valid perturbation_zscore_sweep__*.csv files")
                outs: List[str] = []
                for sweep_csv in sweep_candidates:
                    suffix_from_stem = _extract_sweep_suffix_from_stem(sweep_csv.stem)
                    suffix = _compact_suffix_with_hash(suffix_from_stem or "all")
                    p = figs / f"perturbation_zscore_sweep__{suffix}.pdf"
                    plot_perturbation_zscore_sweep(str(sweep_csv), str(p), cfg=cfg)
                    outs.append(str(p))
                return outs
            _run(fig_id, _build)
            continue

        if fig_id == "perturbation_sweep_heatmap":
            def _build() -> List[str]:
                sweep_candidates = _valid_sweep_candidates(tables)
                if not sweep_candidates:
                    raise _FigureSkip("no valid perturbation_zscore_sweep__*.csv files")
                outs: List[str] = []
                for sweep_csv in sweep_candidates:
                    suffix_from_stem = _extract_sweep_suffix_from_stem(sweep_csv.stem)
                    suffix = _compact_suffix_with_hash(suffix_from_stem or "all")
                    p = figs / f"perturbation_sweep_heatmap__{suffix}.pdf"
                    plot_perturbation_sweep_heatmap(str(sweep_csv), str(p), cfg=cfg)
                    outs.append(str(p))
                return outs
            _run(fig_id, _build)
            continue

        if fig_id == "perturbation_sweep_heatmap_prob":
            def _build() -> List[str]:
                sweep_candidates = _valid_sweep_candidates(tables)
                if not sweep_candidates:
                    raise _FigureSkip("no valid perturbation_zscore_sweep__*.csv files")
                outs: List[str] = []
                for sweep_csv in sweep_candidates:
                    suffix_from_stem = _extract_sweep_suffix_from_stem(sweep_csv.stem)
                    suffix = _compact_suffix_with_hash(suffix_from_stem or "all")
                    p = figs / f"perturbation_sweep_heatmap_prob__{suffix}.pdf"
                    plot_perturbation_sweep_heatmap_prob(str(sweep_csv), str(p), cfg=cfg)
                    outs.append(str(p))
                return outs
            _run(fig_id, _build)
            continue

        if fig_id == "perturbation_timecurve_focus":
            def _build() -> List[str]:
                focus_curve_candidates = _valid_focus_curve_candidates(tables)
                if not focus_curve_candidates:
                    raise _FigureSkip("no valid perturbation_timecurve_focus__*.csv files")
                outs: List[str] = []
                max_groups = int(cfg["perturb_max_groups"])
                focus_scores_cfg = cfg.get("perturb_curve_scores", [-8.0, 0.0, 8.0])
                focus_scores = sorted({float(x) for x in focus_scores_cfg})
                for focus_curve_csv in focus_curve_candidates:
                    show = _load_focus_curve_df(str(focus_curve_csv))
                    group_order = (
                        show.groupby("target_group", as_index=False)["delta_rate"]
                        .apply(lambda x: float(np.max(np.abs(x))))
                        .rename(columns={"delta_rate": "max_abs_rate"})
                        .sort_values("max_abs_rate", ascending=False)
                    )
                    groups = group_order["target_group"].astype(str).head(max_groups).tolist()
                    suffix_from_stem = _extract_focus_curve_suffix_from_stem(focus_curve_csv.stem)
                    suffix = _compact_suffix_with_hash(suffix_from_stem or "all")
                    for group in groups:
                        sub = show[show["target_group"].astype(str) == str(group)]
                        if sub.empty:
                            continue
                        available_scores = sorted({float(x) for x in sub["z_score"].to_numpy(dtype=float)})
                        score_list = [
                            float(z)
                            for z in available_scores
                            if any(abs(float(z) - float(sel)) <= 1e-8 for sel in focus_scores)
                        ]
                        if not score_list:
                            score_list = available_scores
                        group_slug = _slugify_token(group, fallback="group")
                        for z in score_list:
                            z_slug = _signed_float_token(float(z))
                            p = figs / f"perturbation_timecurve_focus__{suffix}__{group_slug}__z-{z_slug}.pdf"
                            plot_perturbation_timecurve_focus(
                                str(focus_curve_csv),
                                str(p),
                                cfg=cfg,
                                target_group=str(group),
                                z_score=float(z),
                            )
                            outs.append(str(p))
                if not outs:
                    raise _FigureSkip("no perturbation_timecurve_focus outputs generated")
                return outs
            _run(fig_id, _build)
            continue

        if fig_id == "key_molecule_dotplot":
            def _build() -> str:
                _require_file(tables / "key_molecules_union.csv", context=fig_id)
                p = figs / "key_molecule_dotplot.pdf"
                plot_key_molecule_dotplot(str(tables / "key_molecules_union.csv"), str(p), cfg=cfg)
                return str(p)
            _run(fig_id, _build)
            continue

        if fig_id == "key_molecule_volcano":
            def _build() -> str:
                _require_file(tables / "de_main_var_path_wilcoxon.csv", context=fig_id)
                _require_file(tables / "de_sec_var_path_wilcoxon.csv", context=fig_id)
                p = figs / "key_molecule_volcano.pdf"
                plot_key_molecule_volcano(
                    str(tables / "de_main_var_path_wilcoxon.csv"),
                    str(tables / "de_sec_var_path_wilcoxon.csv"),
                    str(p),
                    cfg=cfg,
                )
                return str(p)
            _run(fig_id, _build)
            continue

        manifest["failed_figures"][fig_id] = "unhandled figure id"

    with (figs / "figure_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest
