#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.Map.tl.transport_factory import build_transport_map
from CytoBridge.tl.analysis_dense_time import _build_ode_trajectory, _map_to_secondary, _resolve_runtime_device
from CytoBridge.utils import load_model_from_adata
from downstream.plotting.hc_author5_palette import AUTHOR5_COLORS, AUTHOR5_ORDER, author5_color

DEFAULT_CONFIG = REPO_ROOT / "downstream" / "pipline_configs" / "hc_author5_12_2.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return obj


def _extract_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths = cfg.get("paths", {})
    required = ["dynamic_adata", "main_processed", "sub_processed", "map_model_path"]
    out: Dict[str, Path] = {}
    for key in required:
        raw = str(paths.get(key, "")).strip()
        if not raw:
            raise ValueError(f"Missing config.paths.{key}")
        out[key] = Path(raw).expanduser().resolve()
    return out


def _output_dir(cfg: Dict[str, Any], explicit: str) -> Path:
    if explicit.strip():
        out = Path(explicit).expanduser().resolve()
    else:
        out = Path(str(cfg["output"]["root_dir"])).expanduser().resolve() / "author5_time_slice_panels"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _author_labels(obs: pd.DataFrame, cell_type_key: str) -> np.ndarray:
    if "author_cell_type" in obs.columns:
        return obs["author_cell_type"].astype(str).to_numpy()
    return obs[cell_type_key].astype(str).to_numpy()


def _build_mapper(dynamic_adata: ad.AnnData, map_model_path: Path, main_dim: int, sub_dim: int, device: str) -> Any:
    multi_cfg = dynamic_adata.uns.get("all_model", {}).get("model_config", {}).get("multi", {})
    hidden_dim = int(multi_cfg.get("hidden_dim", 64))
    mapper_type = str(multi_cfg.get("mapper_type", "ae"))
    mapper_kwargs = dict(multi_cfg.get("mapper_kwargs", {}))
    return build_transport_map(
        map_model_path=str(map_model_path),
        input_dim1=int(main_dim),
        input_dim2=int(sub_dim),
        mode=(1, 2),
        hidden_dim=hidden_dim,
        device=device,
        mapper_type=mapper_type,
        mapper_kwargs=mapper_kwargs,
    )


def _fit_knn(latent: np.ndarray, labels: np.ndarray, k: int = 25) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=min(int(k), max(1, latent.shape[0] - 1)), n_jobs=-1)
    model.fit(np.asarray(latent, dtype=np.float32), np.asarray(labels).astype(str))
    return model


def _simulate_midpoint(
    *,
    model: Any,
    dynamic_adata: ad.AnnData,
    mapper: Any,
    anchor_main: np.ndarray,
    anchor_time: float,
    midpoint: float,
    dt: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    sim = _build_ode_trajectory(
        model,
        np.asarray(anchor_main, dtype=np.float32),
        [float(anchor_time), float(midpoint)],
        adata=dynamic_adata,
        dt=float(dt),
        device=str(device),
    )
    main_mid = np.asarray(sim["main_latent"][-1], dtype=np.float32)
    sub_mid = np.asarray(_map_to_secondary(mapper, sim["main_latent"])["sub_latent"][-1], dtype=np.float32)
    return main_mid, sub_mid


def _labels_to_colors(labels: np.ndarray) -> np.ndarray:
    return np.asarray([author5_color(str(x)) for x in labels], dtype=object)

# ====================== 新增：边框设置函数（核心修改1）======================
def _set_panel_frame(ax: plt.Axes, *, is_real: bool) -> None:
    """为面板设置边框：真实数据 → 实线框；模拟插值数据 → 虚线框"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.6)
        spine.set_color("black")
        # 真实时间点用实线，模拟时间点用虚线
        spine.set_linestyle("-") if is_real else (0, (3, 2))
# ========================================================================

# ====================== 修改：绘图函数（核心修改2）======================
def _draw_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    *,
    is_real: bool,
) -> None:
    colors = _labels_to_colors(labels)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8 if is_real else 9, alpha=0.70 if is_real else 0.82, linewidths=0)
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([])
    ax.set_yticks([])
    # 调用边框函数，删除原隐藏边框代码
    _set_panel_frame(ax, is_real=is_real)
# ====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 9 HC author5 real/interpolated time-slice panels.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(Path(args.config).expanduser().resolve())
    paths = _extract_paths(cfg)
    out_dir = _output_dir(cfg, args.output_dir)

    analysis_cfg = cfg.get("analysis", {})
    inference_cfg = cfg.get("inference", {})
    time_key = str(analysis_cfg.get("time_key", "time_point_processed"))
    cell_type_key = str(analysis_cfg.get("cell_type_key", "author_cell_type"))
    dt = float(inference_cfg.get("dt", 0.05))
    device = _resolve_runtime_device(str(inference_cfg.get("device", "cpu")))

    main_proc = ad.read_h5ad(str(paths["main_processed"]))
    sub_proc = ad.read_h5ad(str(paths["sub_processed"]))
    dynamic_adata = ad.read_h5ad(str(paths["dynamic_adata"]))

    real_times = sorted({float(x) for x in main_proc.obs[time_key].tolist()})
    if len(real_times) != 5:
        raise ValueError(f"Expected exactly 5 real times for HC author5, got {real_times}")
    display_times = [real_times[0], 0.5, real_times[1], 1.5, real_times[2], 2.5, real_times[3], 3.5, real_times[4]]

    main_latent = np.asarray(main_proc.obsm["X_latent"], dtype=np.float32)
    sub_latent = np.asarray(sub_proc.obsm["X_latent"], dtype=np.float32)
    main_labels = _author_labels(main_proc.obs, cell_type_key)
    sub_labels = _author_labels(sub_proc.obs, cell_type_key)
    main_times = np.asarray(main_proc.obs[time_key], dtype=float)
    sub_times = np.asarray(sub_proc.obs[time_key], dtype=float)

    main_knn = _fit_knn(main_latent, main_labels)
    sub_knn = _fit_knn(sub_latent, sub_labels)
    pca_main = PCA(n_components=2, random_state=args.seed).fit(main_latent)
    pca_sub = PCA(n_components=2, random_state=args.seed).fit(sub_latent)

    model = load_model_from_adata(dynamic_adata)
    mapper = _build_mapper(dynamic_adata, paths["map_model_path"], main_latent.shape[1], sub_latent.shape[1], device)

    slices: Dict[float, Dict[str, Any]] = {}
    for t in real_times:
        main_mask = np.isclose(main_times, float(t))
        sub_mask = np.isclose(sub_times, float(t))
        slices[float(t)] = {
            "main_latent": main_latent[main_mask],
            "sub_latent": sub_latent[sub_mask],
            "main_labels": main_labels[main_mask],
            "sub_labels": sub_labels[sub_mask],
            "is_real": True,
        }

    for start, end in zip(real_times[:-1], real_times[1:]):
        mid = 0.5 * (float(start) + float(end))
        anchor = np.asarray(main_latent[np.isclose(main_times, float(start))], dtype=np.float32)
        mid_main, mid_sub = _simulate_midpoint(
            model=model,
            dynamic_adata=dynamic_adata,
            mapper=mapper,
            anchor_main=anchor,
            anchor_time=float(start),
            midpoint=float(mid),
            dt=dt,
            device=device,
        )
        slices[mid] = {
            "main_latent": mid_main,
            "sub_latent": mid_sub,
            "main_labels": main_knn.predict(mid_main).astype(str),
            "sub_labels": sub_knn.predict(mid_sub).astype(str),
            "is_real": False,
        }

    fig, axes = plt.subplots(2, len(display_times), figsize=(2.55 * len(display_times), 5.6), squeeze=False)
    coord_rows: List[Dict[str, object]] = []
    for col, t in enumerate(display_times):
        item = slices[float(t)]
        main_coords = pca_main.transform(np.asarray(item["main_latent"], dtype=np.float32))
        sub_coords = pca_sub.transform(np.asarray(item["sub_latent"], dtype=np.float32))
        _draw_panel(axes[0][col], main_coords, np.asarray(item["main_labels"]).astype(str), f"RNA t={float(t):g}", is_real=bool(item["is_real"]))
        _draw_panel(axes[1][col], sub_coords, np.asarray(item["sub_labels"]).astype(str), f"ATAC t={float(t):g}", is_real=bool(item["is_real"]))
        if col == 0:
            axes[0][col].set_ylabel("RNA", fontsize=12)
            axes[1][col].set_ylabel("ATAC", fontsize=12)
        for modality, coords_key, labels_key in [("rna", main_coords, item["main_labels"]), ("atac", sub_coords, item["sub_labels"])]:
            labels = np.asarray(labels_key).astype(str)
            for idx in range(coords_key.shape[0]):
                coord_rows.append(
                    {
                        "time": float(t),
                        "modality": modality,
                        "is_real": bool(item["is_real"]),
                        "author_cell_type": str(labels[idx]),
                        "dim1": float(coords_key[idx, 0]),
                        "dim2": float(coords_key[idx, 1]),
                    }
                )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=6.2, color=AUTHOR5_COLORS[label], label=label)
        for label in AUTHOR5_ORDER
    ]
    legend_handles.append(plt.Line2D([0], [0], color="black", linewidth=1.2, label="real anchor"))
    legend_handles.append(plt.Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="interpolated"))
    fig.legend(handles=legend_handles, loc="center right", bbox_to_anchor=(0.995, 0.52), frameon=False, fontsize=9.3)
    fig.subplots_adjust(right=0.92, wspace=0.03, hspace=0.12)

    pdf_path = out_dir / "author5_time_slice_panels.pdf"
    png_path = out_dir / "author5_time_slice_panels.png"
    fig.savefig(pdf_path, dpi=320, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)

    coords_csv = out_dir / "author5_time_slice_coordinates.csv"
    pd.DataFrame(coord_rows).to_csv(coords_csv, index=False)
    manifest = {
        "config": str(Path(args.config).expanduser().resolve()),
        "output_dir": str(out_dir),
        "time_key": time_key,
        "cell_type_key": cell_type_key,
        "real_times": [float(x) for x in real_times],
        "display_times": [float(x) for x in display_times],
        "palette": AUTHOR5_COLORS,
        "outputs": {
            "pdf": str(pdf_path),
            "png": str(png_path),
            "coordinates_csv": str(coords_csv),
        },
    }
    manifest_path = out_dir / "author5_time_slice_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
