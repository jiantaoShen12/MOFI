import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
from pathlib import Path

def _save_dual_format(fig: plt.Figure, output_path: Path):
    """辅助函数：同时保存 PNG 和 PDF"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

def _to_label(value):
    if value is None:
        return "NA"
    try:
        if np.isnan(value):
            return "NA"
    except Exception:
        pass
    return str(value)

def _soften_color(rgb, saturation_keep=0.78, white_mix=0.25):
    desat_rgb = sns.desaturate(rgb, saturation_keep)
    return tuple((1.0 - white_mix) * float(c) + white_mix for c in desat_rgb[:3])

def _build_traj_cmap(traj_cmap):
    name = str(traj_cmap).lower()
    if name in {"sunset1"}:
        base = LinearSegmentedColormap.from_list(
            "nature_v3",
            ["#2F2F85", "#CC4A77", "#FFC829", "#57E020"],
        )
    elif name in {"sunset_31800"}:
        base = LinearSegmentedColormap.from_list(
            "nature_v3",
            ["#2F2F85", "#CC4A77", "#57E020"],
        )
    elif name in {"sunset", "nature_v4"}:
        base = LinearSegmentedColormap.from_list(
            "nature_v3",
            ["#57E020", "#FFC829", "#CC4A77", "#2F2F85"],
        )
    elif name in {"nature_v3"}:
        base = LinearSegmentedColormap.from_list(
            "nature_v4",
            ["#FFC829", "#CC4A77", "#2F2F85"],
        )
    elif name in {"flare", "mako", "rocket", "crest"}:
        base = sns.color_palette(name, as_cmap=True)
    else:
        base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_trimmed_v3",
        base(np.linspace(0.08, 0.95, 256)),
    )

def _build_bg_palette(labels_plot, bg_palette, max_legend_items=10):
    unique_labels, counts = np.unique(labels_plot, return_counts=True)
    order = unique_labels[np.argsort(counts)[::-1]]
    if len(order) > max_legend_items:
        keep_n = max(max_legend_items - 1, 1)
        kept = set(order[:keep_n])
        labels_plot = np.array(
            [lab if lab in kept else "Other" for lab in labels_plot],
            dtype=object,
        )
        legend_labels = list(order[:keep_n]) + ["Other"]
    else:
        legend_labels = list(order)

    palette = sns.color_palette(bg_palette, n_colors=len(legend_labels))
    color_map = {
        lab: _soften_color(palette[i], saturation_keep=0.80, white_mix=0.24)
        for i, lab in enumerate(legend_labels)
    }
    colors = np.array([color_map[lab] for lab in labels_plot], dtype=float)
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor=color_map[lab], markeredgecolor=color_map[lab],
            markeredgewidth=0.0, markersize=5.2, label=lab
        ) for lab in legend_labels
    ]
    return colors, handles

def _draw_background(
    ax, bg_points, bg_labels, bg_palette, background_mode,
    cmap=None, norm=None, is_time_color=False
):
    if bg_points.size == 0:
        return None

    # ====================== 【核心修改】时间上色模式 ======================
    if is_time_color and cmap is not None and norm is not None and bg_labels is not None:
        time_values = np.asarray(bg_labels, dtype=float)
        colors = cmap(norm(time_values))
        ax.scatter(
            bg_points[:, 0], bg_points[:, 1],
            s=6.5, c=colors, marker="o", alpha=0.22,
            linewidths=0, rasterized=True, zorder=1
        )
        return None
    # ====================================================================

    if bg_labels is None:
        mode = str(background_mode).lower()
        if mode == "auto":
            mode = "hexbin" if bg_points.shape[0] > 35000 else "scatter"

        if mode == "hexbin":
            ax.hexbin(
                bg_points[:, 0], bg_points[:, 1], gridsize=115,
                cmap="Greys", mincnt=1, linewidths=0, alpha=0.18, zorder=1
            )
        elif mode == "contour":
            ax.tricontourf(
                bg_points[:, 0], bg_points[:, 1],
                np.ones(bg_points.shape[0], dtype=float),
                levels=6, cmap="Greys", alpha=0.12, zorder=1
            )
        else:
            ax.scatter(
                bg_points[:, 0], bg_points[:, 1],
                s=4.0, c="#9DA3AD", marker="o", alpha=0.10,
                linewidths=0, rasterized=True, zorder=1
            )
        return None

    labels_plot = np.array([_to_label(v) for v in bg_labels], dtype=object)
    bg_colors, legend_handles = _build_bg_palette(labels_plot, bg_palette, max_legend_items=10)
    ax.scatter(
        bg_points[:, 0], bg_points[:, 1],
        s=6.5, c=bg_colors, marker="o", alpha=0.18,
        linewidths=0, rasterized=True, zorder=1
    )
    return legend_handles
    
def _draw_endpoints(ax, traj, traj_times, start_idx, end_idx, norm, cmap):
    start_pt = traj[start_idx]
    end_pt = traj[end_idx]
    start_color = cmap(norm(float(traj_times[start_idx])))
    end_color = cmap(norm(float(traj_times[end_idx])))

    ax.scatter(
        start_pt[0], start_pt[1], s=28,
        facecolors=[start_color], edgecolors=[start_color],
        linewidths=1.00, zorder=1.8
    )
    ax.scatter(
        end_pt[0], end_pt[1], s=58,
        c=[end_color], edgecolors="black", linewidth=0.25, zorder=4.4
    )
    ax.scatter(
        end_pt[0], end_pt[1], s=92,
        facecolors="none", edgecolors=[(*end_color[:3], 0.30)],
        linewidths=0.80, zorder=4.1
    )

def plot_ode_v3(
    X,
    traj_array,
    save_path,
    traj_times,
    time_ticks=None,
    bg_obs_values=None,
    bg_obs_name=None,
    bg_palette="Set2",
    traj_cmap="sunset",
    background_mode="auto",
    show_axes=True,
    dpi=300,
):
    """
    Nature-style trajectory rendering with smooth gradient flow and layered background.
    增强：bg_obs_values = "time_point_processed" 时，背景点使用轨迹同款时间渐变色
    """
    if traj_array.ndim != 3 or traj_array.shape[-1] != 2:
        raise ValueError(f"traj_array must have shape (T, M, 2), got {traj_array.shape}")

    traj_times = np.asarray(traj_times, dtype=float)
    if traj_times.shape[0] != traj_array.shape[0]:
        raise ValueError("traj_times length must match traj_array first dimension")

    sns.set_style("white")
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.grid": False, "axes.linewidth": 0.72,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # ====================== 【关键标记】是否启用时间上色 ======================
    use_time_background = (bg_obs_values == "time_point_processed")
    has_bg_obs = not use_time_background and (bg_obs_values is not None)
    # ========================================================================

    if has_bg_obs:
        if len(bg_obs_values) != len(X):
            raise ValueError("bg_obs_values must have same length as X")
        fig = plt.figure(figsize=(7.8, 6.5), dpi=dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.34], wspace=0.03)
    else:
        fig = plt.figure(figsize=(6.9, 6.5), dpi=dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.13], wspace=0.03)

    ax = fig.add_subplot(gs[0, 0])
    side_ax = fig.add_subplot(gs[0, 1])
    side_ax.set_axis_off()

    bg_points_list = []
    bg_labels_list = []
    for idx, data_t in enumerate(X):
        if data_t is None or len(data_t) == 0: continue
        valid_mask_t = ~np.isnan(data_t).any(axis=1)
        valid_t = data_t[valid_mask_t]
        if len(valid_t) == 0: continue
        bg_points_list.append(valid_t[:, :2])

        # ====================== 时间上色：存入对应时间值 ======================
        if use_time_background:
            t_val = traj_times[idx]
            bg_labels_list.append(np.full(len(valid_t), t_val, dtype=float))
        elif has_bg_obs:
            labels_t = np.asarray(bg_obs_values[idx])
            bg_labels_list.append(labels_t[valid_mask_t])

    bg_points = np.vstack(bg_points_list) if bg_points_list else np.empty((0, 2), dtype=float)
    bg_labels = None
    if use_time_background or has_bg_obs:
        if bg_labels_list:
            bg_labels = np.concatenate(bg_labels_list, axis=0)

    cmap = _build_traj_cmap(traj_cmap)
    norm = Normalize(vmin=float(np.nanmin(traj_times)), vmax=float(np.nanmax(traj_times)))

    # ====================== 调用背景绘制 ======================
    legend_handles = _draw_background(
        ax=ax, bg_points=bg_points, bg_labels=bg_labels,
        bg_palette=bg_palette, background_mode=background_mode,
        cmap=cmap, norm=norm, is_time_color=use_time_background
    )

    n_trajectories = traj_array.shape[1]
    arrow_budget = 18
    arrows_drawn = 0
    for j in range(n_trajectories):
        traj = traj_array[:, j, :]
        valid_mask = ~np.isnan(traj).any(axis=1)
        if valid_mask.sum() == 0: continue

        valid_idx = np.where(valid_mask)[0]
        split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
        idx_chunks = np.split(valid_idx, split_points)

        for chunk in idx_chunks:
            if chunk.size < 2: continue
            pts = traj[chunk]
            t_chunk = traj_times[chunk]
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            seg_t = 0.5 * (t_chunk[:-1] + t_chunk[1:])
            seg_alpha = np.linspace(0.45, 1.0, len(seg_t))

            base_colors = cmap(norm(seg_t))
            glow_colors = base_colors.copy()
            glow_colors[:, 3] = 0.11 * seg_alpha
            main_colors = base_colors.copy()
            main_colors[:, 3] = 0.78 * seg_alpha

            lc_glow = LineCollection(segments, colors=glow_colors, linewidths=3.9, capstyle="round", zorder=2)
            lc_main = LineCollection(segments, colors=main_colors, linewidths=1.4, capstyle="round", zorder=3)
            ax.add_collection(lc_glow)
            ax.add_collection(lc_main)

            if arrows_drawn < arrow_budget and chunk.size >= 8:
                mid = int(0.50 * (chunk.size - 1))
                p0 = pts[max(mid - 1, 0)]
                p1 = pts[min(mid + 1, chunk.size - 1)]
                arrow_color = cmap(norm(float(t_chunk[mid])))
                ax.annotate(
                    "", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=0, mutation_scale=8, alpha=0.85),
                    zorder=3.4
                )
                arrows_drawn += 1

        start_idx = int(valid_idx[0])
        end_idx = int(valid_idx[-1])
        _draw_endpoints(ax, traj, traj_times, start_idx, end_idx, norm, cmap)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if has_bg_obs:
        cax = side_ax.inset_axes([0.43, 0.54, 0.22, 0.40])
    else:
        cax = side_ax.inset_axes([0.42, 0.12, 0.22, 0.78])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Time Point")
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(length=2.2, width=0.6, pad=1.5)
    if time_ticks is not None:
        ticks = np.unique(np.asarray(time_ticks, dtype=float))
        if ticks.size > 0:
            cbar.set_ticks(ticks)

    if legend_handles:
        side_ax.legend(
            handles=legend_handles, title=(bg_obs_name or "Background"),
            title_fontsize=8, fontsize=7.5, loc="lower left",
            bbox_to_anchor=(0,0.02), frameon=False
        )

    ax.set_aspect("equal", adjustable="box")
    if show_axes:
        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")
        ax.tick_params(length=2.8, width=0.7)
        sns.despine(ax=ax, top=True, right=True)
    else:
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel(""), ax.set_ylabel("")
        for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    save_parent = os.path.dirname(save_path)
    if save_parent: os.makedirs(save_parent, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    _save_dual_format(fig=fig, output_path=Path(save_path))
    plt.close(fig)
