"""Release helpers for the five MOFI paper-reproduction notebooks.

The helpers in this module deliberately *orchestrate* existing MOFI APIs.  They
do not contain alternative dynamics, mapping, perturbation, or plotting
implementations.  Keeping path resolution, smoke-test sampling, and manifest
validation here makes the notebooks short and prevents notebook-local analysis
code from drifting away from the library.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

# Numba and matplotlib otherwise try to cache inside a read-only conda
# installation on managed workstations.  Keep all runtime caches local and
# ignored by Git.
os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PaperDataset:
    dataset_id: str
    primary_name: str
    secondary_name: str
    primary_path: str
    secondary_path: str
    dynamics_path: str
    training_map_path: str
    analysis_map_path: str
    training_config: str
    analysis_config: str
    time_key: str
    cell_type_key: str
    expected_shape: tuple[int, int]


PAPER_DATASETS: Dict[str, PaperDataset] = {
    "gse213152": PaperDataset(
        dataset_id="gse213152",
        primary_name="RNA",
        secondary_name="ATAC",
        primary_path="paper_data/processed/gse213152/rna.h5ad",
        secondary_path="paper_data/processed/gse213152/atac.h5ad",
        dynamics_path="paper_data/dynamics/gse213152/adata.h5ad",
        training_map_path="paper_data/maps/gse213152_training/best_model.pt",
        analysis_map_path="paper_data/maps/gse213152_analysis/best_model.pt",
        training_config="configs/training/gse213152.yaml",
        analysis_config="configs/analysis/gse213152.yaml",
        time_key="time_point_processed",
        cell_type_key="cell_type_fine",
        expected_shape=(55177, 50),
    ),
    "hspc_31800": PaperDataset(
        dataset_id="hspc_31800",
        primary_name="RNA",
        secondary_name="protein",
        primary_path="paper_data/processed/hspc_31800/rna.h5ad",
        secondary_path="paper_data/processed/hspc_31800/protein.h5ad",
        dynamics_path="paper_data/dynamics/hspc_31800/adata.h5ad",
        training_map_path="paper_data/maps/hspc_31800/best_model.pt",
        analysis_map_path="paper_data/maps/hspc_31800/best_model.pt",
        training_config="configs/training/hspc_31800.yaml",
        analysis_config="configs/analysis/hspc_31800.yaml",
        time_key="time_point_processed",
        cell_type_key="cell_type",
        expected_shape=(28087, 50),
    ),
    "ho": PaperDataset(
        dataset_id="ho",
        primary_name="RNA",
        secondary_name="ATAC",
        primary_path="paper_data/processed/ho/rna.h5ad",
        secondary_path="paper_data/processed/ho/atac.h5ad",
        dynamics_path="paper_data/dynamics/ho/adata.h5ad",
        training_map_path="paper_data/maps/ho/best_model.pt",
        analysis_map_path="paper_data/maps/ho/best_model.pt",
        training_config="configs/training/ho.yaml",
        analysis_config="configs/analysis/ho.yaml",
        time_key="time_point_processed",
        cell_type_key="cell_type",
        expected_shape=(9894, 50),
    ),
    "hc_author5": PaperDataset(
        dataset_id="hc_author5",
        primary_name="RNA",
        secondary_name="ATAC",
        primary_path="paper_data/processed/hc_author5/rna.h5ad",
        secondary_path="paper_data/processed/hc_author5/atac.h5ad",
        dynamics_path="paper_data/dynamics/hc_author5/adata.h5ad",
        training_map_path="paper_data/maps/hc_author5/best_model.pt",
        analysis_map_path="paper_data/maps/hc_author5/best_model.pt",
        training_config="configs/training/hc_author5.yaml",
        analysis_config="configs/analysis/hc_author5.yaml",
        time_key="time_point_processed",
        cell_type_key="author_cell_type",
        expected_shape=(6163, 50),
    ),
}


DEFAULT_FIGURES: Dict[str, list[str]] = {
    "gse213152": [
        "trajectory",
        "cell_type_main",
        "rna_bridging",
        "protein_bridging",
    ],
    "hspc_31800": [
        "trajectory",
        "perturbation_zscore_sweep",
        "perturbation_sweep_heatmap_prob",
        "program_both",
        "key_molecule_volcano",
    ],
    "ho": [
        "trajectory",
        "program_both",
        "rna_bridging",
        "protein_bridging",
        "key_molecule_volcano",
    ],
    "hc_author5": [
        "trajectory",
        "perturbation_zscore_sweep",
        "perturbation_sweep_heatmap_prob",
        "perturbation_timecurve_focus",
    ],
}


def _abs(relative_path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def dataset_spec(dataset_id: str) -> PaperDataset:
    try:
        return PAPER_DATASETS[dataset_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown paper dataset {dataset_id!r}; choose {sorted(PAPER_DATASETS)}"
        ) from exc


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with _abs(path).open("r", encoding="utf-8") as stream:
        obj = yaml.safe_load(stream)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return obj


def require_paper_data(dataset_id: str, repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    spec = dataset_spec(dataset_id)
    paths = {
        "primary": _abs(spec.primary_path, repo_root),
        "secondary": _abs(spec.secondary_path, repo_root),
        "dynamics": _abs(spec.dynamics_path, repo_root),
        "training_map": _abs(spec.training_map_path, repo_root),
        "analysis_map": _abs(spec.analysis_map_path, repo_root),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Paper data are incomplete. Run `python scripts/download_data.py --all` "
            f"or restore the local release bundle. Missing: {missing}"
        )
    return {key: str(path) for key, path in paths.items()}


def correct_ho_stage_labels(
    adata: ad.AnnData,
    *,
    label_key: str = "cell_type",
) -> tuple[ad.AnnData, Dict[str, Any]]:
    """Correct the swapped HO stage abbreviations in a release copy.

    The early stage originally labelled ``nepi`` is neuroectoderm (``nect``),
    whereas the later stage originally labelled ``nect`` is neuroepithelium
    (``nepi``).  A temporary token prevents an in-place two-way replacement
    from collapsing both categories.
    """

    if label_key not in adata.obs:
        raise KeyError(f"HO label column not found: {label_key}")
    before = adata.obs[label_key].astype(str)
    after = before.replace({"nepi": "__mofi_early_nect__", "nect": "nepi"})
    after = after.replace({"__mofi_early_nect__": "nect"})
    changed = int(np.sum(before.to_numpy() != after.to_numpy()))
    if changed == 0 and {"nect", "nepi"}.issubset(set(before.unique())):
        correction = adata.uns.get("mofi_ho_label_correction", {})
        if correction.get("status") == "applied":
            return adata, dict(correction)
    adata.obs[label_key] = pd.Categorical(after)
    audit = {
        "status": "applied",
        "label_key": label_key,
        "mapping": {"early_original_nepi": "nect", "late_original_nect": "nepi"},
        "changed_cells": changed,
        "before_counts": before.value_counts(dropna=False).to_dict(),
        "after_counts": after.value_counts(dropna=False).to_dict(),
    }
    adata.uns["mofi_ho_label_correction"] = audit
    return adata, audit


def apply_ho_release_label_fix(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Apply the HO label correction only to the local release data copies."""

    spec = dataset_spec("ho")
    targets = [
        _abs(spec.primary_path, repo_root),
        _abs(spec.secondary_path, repo_root),
        _abs(spec.dynamics_path, repo_root),
    ]
    report: Dict[str, Any] = {"files": {}}
    for path in targets:
        obj = ad.read_h5ad(path)
        if "cell_type" not in obj.obs:
            report["files"][str(path)] = {"status": "no_cell_type_column"}
            continue
        obj, audit = correct_ho_stage_labels(obj)
        obj.write_h5ad(path)
        report["files"][str(path)] = audit
    report_path = _abs("validation/ho_label_correction.json", repo_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _paired_time_sample(
    primary: ad.AnnData,
    secondary: ad.AnnData,
    *,
    time_key: str,
    cells_per_time: int,
    seed: int,
) -> tuple[ad.AnnData, ad.AnnData, Dict[str, int]]:
    """Return a deterministic paired subset for a training smoke test."""

    if time_key not in primary.obs or time_key not in secondary.obs:
        raise KeyError(f"Both modalities must contain obs[{time_key!r}]")
    common = primary.obs_names.intersection(secondary.obs_names, sort=False)
    if len(common) == 0:
        if primary.n_obs != secondary.n_obs:
            raise ValueError("Modalities have no shared obs names and different cell counts")
        primary = primary.copy()
        secondary = secondary.copy()
        paired_names = pd.Index([f"paired_{i}" for i in range(primary.n_obs)])
        primary.obs_names = paired_names
        secondary.obs_names = paired_names
        common = paired_names
    primary = primary[common].copy()
    secondary = secondary[common].copy()
    p_time = primary.obs[time_key].astype(float).to_numpy()
    s_time = secondary.obs[time_key].astype(float).to_numpy()
    if not np.allclose(p_time, s_time):
        raise ValueError("Paired modalities disagree on time labels")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    counts: Dict[str, int] = {}
    for value in sorted(np.unique(p_time)):
        candidates = np.flatnonzero(np.isclose(p_time, value))
        take = min(int(cells_per_time), len(candidates))
        selected = rng.choice(candidates, size=take, replace=False)
        chosen.extend(int(x) for x in selected)
        counts[str(float(value))] = int(take)
    chosen_arr = np.asarray(chosen, dtype=int)
    return primary[chosen_arr].copy(), secondary[chosen_arr].copy(), counts


def resolved_training_config(
    dataset_id: str,
    *,
    output_dir: str | Path,
    epochs_per_stage: Optional[int] = None,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    spec = dataset_spec(dataset_id)
    paths = require_paper_data(dataset_id, repo_root)
    cfg = copy.deepcopy(load_yaml(_abs(spec.training_config, repo_root)))
    cfg["ckpt_dir"] = str(_abs(output_dir, repo_root))
    cfg.setdefault("model", {}).setdefault("multi", {})["Map_model_path"] = paths[
        "training_map"
    ]
    if epochs_per_stage is not None:
        if epochs_per_stage <= 0:
            raise ValueError("epochs_per_stage must be positive")
        for stage in cfg["training"]["plan"]:
            stage["epochs"] = int(epochs_per_stage)
    return cfg


def run_real_training_smoke(
    dataset_id: str,
    *,
    output_dir: str | Path,
    device: str = "cuda",
    cells_per_time: int = 24,
    epochs_per_stage: int = 1,
    seed: int = 42,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Run the original :func:`CytoBridge.tl.fit` on a small paired subset."""

    from CytoBridge.tl.fit import fit
    from CytoBridge.utils.utils import set_seed

    spec = dataset_spec(dataset_id)
    paths = require_paper_data(dataset_id, repo_root)
    primary = ad.read_h5ad(paths["primary"])
    secondary = ad.read_h5ad(paths["secondary"])
    if dataset_id == "ho":
        primary, _ = correct_ho_stage_labels(primary)
        secondary, _ = correct_ho_stage_labels(secondary)
    primary, secondary, sampled = _paired_time_sample(
        primary,
        secondary,
        time_key=spec.time_key,
        cells_per_time=cells_per_time,
        seed=seed,
    )
    cfg = resolved_training_config(
        dataset_id,
        output_dir=output_dir,
        epochs_per_stage=epochs_per_stage,
        repo_root=repo_root,
    )
    set_seed(seed)
    started = time.perf_counter()
    trained = fit(
        primary,
        config=cfg,
        adata_sec=secondary,
        batch_size=min(400, max(sampled.values())),
        device=device,
    )
    elapsed = float(time.perf_counter() - started)
    velocity = np.asarray(trained.obsm["velocity_latent"])
    growth = np.asarray(trained.obsm["growth_rate"])
    if not np.isfinite(velocity).all() or not np.isfinite(growth).all():
        raise RuntimeError("Smoke training produced non-finite velocity or growth")
    result = {
        "dataset_id": dataset_id,
        "status": "passed",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "elapsed_seconds": elapsed,
        "epochs_per_stage": int(epochs_per_stage),
        "sampled_cells_by_time": sampled,
        "n_obs": int(trained.n_obs),
        "latent_dim": int(velocity.shape[1]),
        "velocity_finite": True,
        "growth_finite": True,
        "output_adata": str(_abs(output_dir, repo_root) / "adata.h5ad"),
    }
    out = _abs(output_dir, repo_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_test.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def resolved_analysis_config(
    dataset_id: str,
    *,
    output_dir: str | Path,
    device: str = "cuda",
    sample_max_cells: Optional[int] = None,
    reuse_cache: bool = True,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    spec = dataset_spec(dataset_id)
    paths = require_paper_data(dataset_id, repo_root)
    cfg = copy.deepcopy(load_yaml(_abs(spec.analysis_config, repo_root)))
    cfg["paths"] = {
        "dynamic_adata": paths["dynamics"],
        "main_processed": paths["primary"],
        "sub_processed": paths["secondary"],
        "map_model_path": paths["analysis_map"],
    }
    cfg.setdefault("output", {})["root_dir"] = str(_abs(output_dir, repo_root))
    inference = cfg.setdefault("inference", {})
    inference["device"] = str(device)
    inference["reuse_cache"] = bool(reuse_cache)
    if sample_max_cells is not None:
        inference["sample_max_cells"] = int(sample_max_cells)
    return cfg


def write_resolved_analysis_config(
    dataset_id: str,
    *,
    output_dir: str | Path,
    config_path: str | Path,
    device: str = "cuda",
    sample_max_cells: Optional[int] = None,
    reuse_cache: bool = True,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Write a portable, machine-local copy of a paper analysis config."""

    cfg = resolved_analysis_config(
        dataset_id,
        output_dir=output_dir,
        device=device,
        sample_max_cells=sample_max_cells,
        reuse_cache=reuse_cache,
        repo_root=repo_root,
    )
    destination = _abs(config_path, repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return destination


def run_real_downstream(
    dataset_id: str,
    *,
    output_dir: str | Path,
    selected_figures: Optional[Sequence[str]] = None,
    device: str = "cuda",
    sample_max_cells: Optional[int] = None,
    reuse_cache: bool = True,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Run granular MOFI inference, analysis, and plotting calls.

    This intentionally does not call ``run_dense_time_workflow``.  The three
    public calls below are the same original stages used by that pipeline.
    """

    from CytoBridge.pl.plot_dense_time import generate_dense_time_report_figures
    from CytoBridge.tl.analysis_dense_time import (
        analyze_dense_time_predictions,
        infer_dense_time_dual,
        resolve_time_grid,
    )

    spec = dataset_spec(dataset_id)
    cfg = resolved_analysis_config(
        dataset_id,
        output_dir=output_dir,
        device=device,
        sample_max_cells=sample_max_cells,
        reuse_cache=reuse_cache,
        repo_root=repo_root,
    )
    paths = cfg["paths"]
    inference = cfg["inference"]
    analysis_cfg = cfg["analysis"]
    output_root = Path(cfg["output"]["root_dir"])
    mode = str(inference.get("run_grids", [{"mode": "coarse_dense"}])[0]["mode"])
    run_dir = output_root / mode / "sample"
    observed = ad.read_h5ad(paths["main_processed"], backed="r")
    observed_times = np.asarray(observed.obs[spec.time_key], dtype=float)
    observed.file.close()
    grid = resolve_time_grid(
        observed_times,
        mode=mode,
        values=None,
        time_grid_presets=inference.get("time_grid_presets", {}),
    )
    started = time.perf_counter()
    inference_result = infer_dense_time_dual(
        dynamic_adata_path=paths["dynamic_adata"],
        main_processed_path=paths["main_processed"],
        sub_processed_path=paths["sub_processed"],
        map_model_path=paths["map_model_path"],
        output_dir=str(run_dir / "assets"),
        time_grid=grid,
        time_key=spec.time_key,
        cell_type_key=spec.cell_type_key,
        sampling_mode=str(inference.get("sampling_mode", "random")),
        sampling_cell_type=inference.get("sampling_cell_type"),
        sampling_indices=inference.get("sampling_indices"),
        strict_index_init_time=bool(inference.get("strict_index_init_time", True)),
        max_cells=inference.get("sample_max_cells"),
        init_time=inference.get("init_time"),
        dt=float(inference.get("dt", 0.05)),
        seed=int(inference.get("seed", 42)),
        device=str(inference.get("device", device)),
        mapper_type=str(inference.get("mapper_type", "ae")),
        mapper_kwargs=inference.get("mapper_kwargs", {}),
        reuse_cache=bool(inference.get("reuse_cache", True)),
    )
    analysis_result = analyze_dense_time_predictions(
        inference_output=inference_result,
        output_dir=str(run_dir),
        analysis_cfg=analysis_cfg,
    )
    figure_manifest = generate_dense_time_report_figures(
        str(run_dir),
        selected_figures=list(selected_figures or DEFAULT_FIGURES[dataset_id]),
        figure_cfg=cfg.get("figure", {}),
    )
    elapsed = float(time.perf_counter() - started)
    result = {
        "dataset_id": dataset_id,
        "run_dir": str(run_dir),
        "elapsed_seconds": elapsed,
        "inference_output": inference_result,
        "analysis_output": analysis_result,
        "figure_manifest": figure_manifest,
    }
    (run_dir / "notebook_stage_manifest.json").write_text(
        json.dumps(_json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def run_paper_ode_v3(
    dataset_id: str,
    *,
    output_dir: str | Path,
    device: str = "cuda",
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Render manuscript ODE trajectories through the original ``pl.plot`` API.

    The paper-selected UMAP estimators are copied into the exact cache names
    consumed by :func:`plot_ode_trajectories_v2` and
    :func:`plot_ode_trajectories_map_v2`.  Those functions route drawing to
    ``pl.plot_ode_v3``; this helper only resolves release-local paths.
    """

    if dataset_id not in {"gse213152", "ho"}:
        raise ValueError("Paper ODE-v3 release helper currently covers GSE213152 and HO")

    from CytoBridge.Map.tl.transport_factory import build_transport_map
    from CytoBridge.pl.plot import (
        plot_growth_map,
        plot_growthv1,
        plot_ode_trajectories_map_v2,
        plot_ode_trajectories_v2,
    )
    from CytoBridge.pl.plot_ode_v3 import plot_ode_v3 as plot_ode_v3_core
    from CytoBridge.utils import load_model_from_adata

    spec = dataset_spec(dataset_id)
    paths = require_paper_data(dataset_id, repo_root)
    umap_dir = _abs(Path("paper_data") / "umap" / dataset_id, repo_root)
    primary_umap = umap_dir / "primary.pkl"
    secondary_umap = umap_dir / "secondary.pkl"
    use_archived_umap = dataset_id in {"gse213152", "ho"}
    missing = [str(p) for p in (primary_umap, secondary_umap) if not p.exists()]
    if use_archived_umap and missing:
        raise FileNotFoundError(f"Paper-selected UMAP assets are missing: {missing}")

    out = _abs(output_dir, repo_root)
    ode_dir = out / "ode_results"
    ode_dir.mkdir(parents=True, exist_ok=True)
    primary_cache = ode_dir / "dim_reducer_umap.pkl"
    secondary_cache = ode_dir / "dim_reducer_map_v2_umap.pkl"

    if dataset_id == "ho":
        projected_dir = _abs(Path("paper_data") / "projected" / "ho_figure5", repo_root)
        archived_dir = _abs(Path("paper_data") / "archived" / "ho_figure5", repo_root)
        original_primary_png = archived_dir / "ode_trajectories_v2_original.png"
        original_primary_pdf = archived_dir / "ode_trajectories_v2_original.pdf"
        expected_primary_sha256 = "21b7e8a5991bd28f6a3ce798d2c72e9f7372baa8d747b9b97ac3e0518a1b16c3"
        if not original_primary_png.exists() or not original_primary_pdf.exists():
            raise FileNotFoundError("The archived Figure 5 RNA trajectory assets are missing")
        if hashlib.sha256(original_primary_png.read_bytes()).hexdigest() != expected_primary_sha256:
            raise RuntimeError("Archived Figure 5 RNA trajectory PNG failed its checksum")
        projected = {
            name: np.load(projected_dir / filename)
            for name, filename in {
                "primary_background": "primary_background.npy",
                "primary_trajectories": "primary_trajectories.npy",
                "primary_points": "primary_points.npy",
                "secondary_background": "secondary_background.npy",
                "secondary_trajectories": "secondary_trajectories.npy",
                "secondary_points": "secondary_points.npy",
            }.items()
        }
        dynamics = ad.read_h5ad(paths["dynamics"])
        dynamics, _ = correct_ho_stage_labels(dynamics)
        obs_time = np.asarray(dynamics.obs[spec.time_key], dtype=float)
        unique_times = np.sort(np.unique(obs_time))
        if any(projected[key].shape[0] != dynamics.n_obs for key in ("primary_background", "secondary_background")):
            raise ValueError("Archived Figure 5 UMAP backgrounds do not match HO cell count")

        # The original HO analysis kept its RNA Figure 5 UMAP trajectory as a
        # rendered PDF and its trajectory/point arrays, but not the fitted
        # background coordinates.  Re-fitting UMAP today changes the coordinate
        # system.  Preserve the paper-used RNA panel verbatim; ATAC remains
        # rendered live through the original plot_ode_v3 implementation below.
        primary_output = ode_dir / "ode_trajectories_primary.png"
        shutil.copy2(original_primary_png, primary_output)

        for prefix in ("secondary",):
            background = projected[f"{prefix}_background"]
            trajectories = projected[f"{prefix}_trajectories"]
            X = [background[np.isclose(obs_time, time)] for time in unique_times]
            labels = [
                dynamics.obs.loc[np.isclose(obs_time, time), spec.cell_type_key].astype(str).to_numpy()
                for time in unique_times
            ]
            save_path = ode_dir / f"ode_trajectories_{prefix}.png"
            plot_ode_v3_core(
                X=X,
                traj_array=trajectories,
                save_path=str(save_path),
                traj_times=np.linspace(float(unique_times[0]), float(unique_times[-1]), trajectories.shape[0]),
                time_ticks=unique_times,
                bg_obs_values=labels,
                bg_obs_name=spec.cell_type_key,
                bg_palette="tab10",
                show_axes=True,
            )

        dynamics.obsm["X_umap"] = projected["primary_background"]
        plot_growthv1(dynamics, dim_reduction="umap", output_path=str(out / "growth_primary.png"))
        dynamics.obsm["X_umap"] = projected["secondary_background"]
        plot_growthv1(dynamics, dim_reduction="umap", output_path=str(out / "growth_secondary.png"))

        archived_sampling = json.loads((projected_dir / "sampling_manifest.json").read_text(encoding="utf-8"))
        payload = {
            "dataset_id": dataset_id,
            "backend": "CytoBridge.pl.plot_ode_v3",
            "umap_mode": "archived_original_rna_umap_and_live_atac_plot_ode_v3",
            "projected_assets": str(projected_dir),
            "primary_trajectory_source_pdf": str(original_primary_pdf),
            "primary_trajectory_source_png_sha256": expected_primary_sha256,
            "sampling": archived_sampling["sampling"],
            "init_indices": archived_sampling["init_indices"],
            "n_trajectories": int(projected["primary_trajectories"].shape[1]),
            "n_bins": int(projected["primary_trajectories"].shape[0]),
            "outputs": {
                "primary_trajectory_png": str(primary_output),
                "secondary_trajectory_png": str(ode_dir / "ode_trajectories_secondary.png"),
                "primary_growth_png": str(out / "growth_primary.png"),
                "secondary_growth_png": str(out / "growth_secondary.png"),
            },
        }
        (out / "paper_ode_v3_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload

    shutil.copy2(primary_umap, primary_cache)
    shutil.copy2(secondary_umap, secondary_cache)

    dynamics = ad.read_h5ad(paths["dynamics"])
    if dataset_id == "ho":
        dynamics, _ = correct_ho_stage_labels(dynamics)
    model = load_model_from_adata(dynamics)
    mapper = build_transport_map(
        map_model_path=paths["analysis_map"],
        input_dim1=int(dynamics.obsm["X_latent"].shape[1]),
        input_dim2=50,
        mode=(1, 2),
        hidden_dim=32,
        device=device,
        mapper_type="ae",
    )

    obs_time = np.asarray(dynamics.obs[spec.time_key], dtype=float)
    t0 = float(np.min(obs_time))
    if dataset_id == "gse213152":
        sampling_key = "cell_type"
        init_cell_type = "Nephron_progenitor"
        background_key = spec.cell_type_key
        sampling_mode = "index"
        init_indices = [1003, 8808, 1041, 7463, 5016, 4951, 7954, 1099, 2333, 9768]
        n_trajectories = 10
        n_bins = 301
    else:
        # Match the manuscript pipeline_auto/ho12.yaml visualization block:
        # random t0 sampling, 20 trajectories, 10 simulation bins, seed 42.
        sampling_key = spec.cell_type_key
        init_cell_type = None
        background_key = spec.cell_type_key
        sampling_mode = "random"
        init_indices = None
        n_trajectories = 20
        n_bins = 10

    common = dict(
        adata=dynamics,
        model=model,
        output_path=str(out),
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        dim_reduction="umap",
        bg_obs_key=background_key,
        bg_palette="tab10",
        device=device,
        sampling_mode=sampling_mode,
        init_time=t0,
        init_cell_type=init_cell_type,
        init_indices=init_indices,
        cell_type_key=sampling_key,
        sampling_seed=42,
        strict_index_init_time=(dataset_id == "ho"),
    )
    plot_ode_trajectories_v2(
        **common,
        save_base_name="ode_trajectories_primary.png",
    )
    if dataset_id == "gse213152":
        # Figure 3E was drawn in the full-data ATAC UMAP fitted on all 60,845
        # observed ATAC cells.  Re-transforming mapped RNA states with the
        # earlier 2,500-cell search reducer collapses this manifold into a
        # compact blob.  Use the archived full-data embedding and the original
        # plot_ode_v3-projected trajectory arrays instead.
        projected_dir = _abs(
            Path("paper_data") / "projected" / "gse213152_figure3", repo_root
        )
        secondary_background = np.load(projected_dir / "secondary_background.npy")
        secondary_trajectories = np.load(projected_dir / "secondary_trajectories.npy")
        secondary_data = ad.read_h5ad(paths["secondary"], backed="r")
        if secondary_background.shape != (secondary_data.n_obs, 2):
            raise ValueError(
                "Archived Figure 3 ATAC UMAP background does not match processed ATAC cells"
            )
        secondary_times = np.asarray(secondary_data.obs[spec.time_key], dtype=float)
        secondary_unique_times = np.sort(np.unique(secondary_times))
        secondary_X = [
            secondary_background[np.isclose(secondary_times, time)]
            for time in secondary_unique_times
        ]
        secondary_labels = [
            secondary_data.obs.loc[
                np.isclose(secondary_times, time), spec.cell_type_key
            ].astype(str).to_numpy()
            for time in secondary_unique_times
        ]
        plot_ode_v3_core(
            X=secondary_X,
            traj_array=secondary_trajectories,
            save_path=str(ode_dir / "ode_trajectories_secondary.png"),
            traj_times=np.linspace(
                float(secondary_unique_times[0]),
                float(secondary_unique_times[-1]),
                secondary_trajectories.shape[0],
            ),
            time_ticks=secondary_unique_times,
            bg_obs_values=secondary_labels,
            bg_obs_name=spec.cell_type_key,
            bg_palette="tab10",
            show_axes=True,
        )
    else:
        plot_ode_trajectories_map_v2(
            **common,
            tranmap=mapper,
            save_base_name="ode_trajectories_secondary.png",
        )

    with primary_cache.open("rb") as stream:
        primary_reducer = __import__("pickle").load(stream)
    dynamics.obsm["X_umap"] = primary_reducer.transform(
        np.asarray(dynamics.obsm["X_latent"], dtype=np.float32)
    )
    plot_growthv1(
        dynamics,
        dim_reduction="umap",
        output_path=str(out / "growth_primary.png"),
    )
    plot_growth_map(
        dynamics,
        mapper,
        dim_reduction="umap",
        output_path=str(out / "growth_secondary.png"),
        device=device,
        umap_model_path=str(secondary_cache),
    )
    payload = {
        "dataset_id": dataset_id,
        "backend": "CytoBridge.pl.plot -> CytoBridge.pl.plot_ode_v3",
        "umap_mode": "legacy_umap_adata_coordinates" if dataset_id == "ho" else ("archived_paper_selected" if use_archived_umap else "original_library_refit"),
        "primary_umap": str(primary_cache),
        "secondary_umap": str(secondary_cache),
        "init_cell_type": init_cell_type,
        "sampling_mode": sampling_mode,
        "init_indices": init_indices,
        "n_trajectories": n_trajectories,
        "n_bins": n_bins,
        "outputs": {
            "primary_trajectory_png": str(ode_dir / "ode_trajectories_primary.png"),
            "secondary_trajectory_png": str(ode_dir / "ode_trajectories_secondary.png"),
            "primary_growth_png": str(out / "growth_primary.png"),
            "secondary_growth_png": str(out / "growth_secondary.png"),
        },
    }
    if dataset_id == "gse213152":
        payload["secondary_projection_mode"] = "archived_full_atac_umap_coordinates"
        payload["secondary_projected_assets"] = str(
            _abs(Path("paper_data") / "projected" / "gse213152_figure3", repo_root)
        )
    (out / "paper_ode_v3_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def assert_figure_manifest(manifest: Mapping[str, Any], expected: Iterable[str]) -> None:
    expected_set = set(expected)
    generated = set(manifest.get("generated_figures", {}))
    failed = manifest.get("failed_figures", {})
    skipped = manifest.get("skipped_figures", {})
    missing = sorted(expected_set - generated)
    if missing:
        raise AssertionError(
            f"Required figures were not generated: {missing}; failed={failed}; skipped={skipped}"
        )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
