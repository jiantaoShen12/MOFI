"""Data loading and analysis helpers for the MOFI notebooks."""

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
    analysis_config: str
    time_key: str
    cell_type_key: str
    expected_shape: tuple[int, int]


PAPER_DATASETS: Dict[str, PaperDataset] = {
    "gse213152": PaperDataset(
        dataset_id="gse213152",
        primary_name="RNA",
        secondary_name="ATAC",
        primary_path="paper_data/inference/gse213152/rna.h5ad",
        secondary_path="paper_data/inference/gse213152/atac.h5ad",
        dynamics_path="results/reviewer_inputs/gse213152/dynamics.h5ad",
        training_map_path="results/reviewer_inputs/gse213152/map_gse213152_training.pt",
        analysis_map_path="results/reviewer_inputs/gse213152/map_gse213152_analysis.pt",
        analysis_config="configs/analysis/gse213152.yaml",
        time_key="time_point_processed",
        cell_type_key="cell_type_fine",
        expected_shape=(55177, 50),
    ),
    "hspc_31800": PaperDataset(
        dataset_id="hspc_31800",
        primary_name="RNA",
        secondary_name="protein",
        primary_path="paper_data/inference/hspc_31800/rna.h5ad",
        secondary_path="paper_data/inference/hspc_31800/protein.h5ad",
        dynamics_path="results/reviewer_inputs/hspc_31800/dynamics.h5ad",
        training_map_path="results/reviewer_inputs/hspc_31800/map_hspc_31800.pt",
        analysis_map_path="results/reviewer_inputs/hspc_31800/map_hspc_31800.pt",
        analysis_config="configs/analysis/hspc_31800.yaml",
        time_key="time_point_processed",
        cell_type_key="cell_type",
        expected_shape=(28087, 50),
    ),
    "ho": PaperDataset(
        dataset_id="ho",
        primary_name="RNA",
        secondary_name="ATAC",
        primary_path="paper_data/inference/ho/rna.h5ad",
        secondary_path="paper_data/inference/ho/atac.h5ad",
        dynamics_path="results/reviewer_inputs/ho/dynamics.h5ad",
        training_map_path="results/reviewer_inputs/ho/map_ho.pt",
        analysis_map_path="results/reviewer_inputs/ho/map_ho.pt",
        analysis_config="configs/analysis/ho.yaml",
        time_key="time_point_processed",
        cell_type_key="cell_type",
        expected_shape=(9894, 50),
    ),
    "hc_author5": PaperDataset(
        dataset_id="hc_author5",
        primary_name="RNA",
        secondary_name="ATAC",
        primary_path="paper_data/inference/hc_author5/rna.h5ad",
        secondary_path="paper_data/inference/hc_author5/atac.h5ad",
        dynamics_path="results/reviewer_inputs/hc_author5/dynamics.h5ad",
        training_map_path="results/reviewer_inputs/hc_author5/map_hc_author5.pt",
        analysis_map_path="results/reviewer_inputs/hc_author5/map_hc_author5.pt",
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
    bundle = _abs(f"paper_data/inference/{dataset_id}", repo_root)
    source_paths = [bundle / "rna.h5ad", bundle / f"{spec.secondary_name.lower()}.h5ad",
                    bundle / "model.json", bundle / "weights.npz"]
    map_paths = sorted(bundle.glob("map_*.npz"))
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources or not map_paths:
        raise FileNotFoundError(f"Required inference bundle is incomplete: {missing_sources}; map files: {len(map_paths)}")
    output = _abs(f"results/reviewer_inputs/{dataset_id}", repo_root)
    expected = [output / "dynamics.h5ad"] + [output / (path.stem + ".pt") for path in map_paths]
    if any(not path.is_file() for path in expected) or min(path.stat().st_mtime_ns for path in expected) < max(path.stat().st_mtime_ns for path in source_paths + map_paths):
        import torch
        from CytoBridge.inference import materialize_legacy_inputs
        materialize_legacy_inputs(bundle, output, device="cuda" if torch.cuda.is_available() else "cpu")
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
            "Required data files are missing. See paper_data/README.md. "
            f"Missing: {missing}"
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
    (``nepi``), matching the manuscript HO Results section. This helper is
    only for release copies known to have the original swapped abbreviations;
    it cannot infer biological identity from labels alone. An applied audit is
    checked before replacement and survives AnnData copying and H5AD I/O.
    """

    if label_key not in adata.obs:
        raise KeyError(f"HO label column not found: {label_key}")
    correction = adata.uns.get("mofi_ho_label_correction", {})
    if correction.get("status") == "applied":
        # Older release audits already include label_key. Do not let a marker
        # for another column silently suppress this column's correction.
        if correction.get("label_key") != label_key:
            raise ValueError("HO correction audit refers to a different label column")
        return adata, copy.deepcopy(dict(correction))
    before = adata.obs[label_key].astype(str)
    after = before.replace({"nepi": "__mofi_early_nect__", "nect": "nepi"})
    after = after.replace({"__mofi_early_nect__": "nect"})
    changed = int(np.sum(before.to_numpy() != after.to_numpy()))
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


def ho_label_signature(
    adata: ad.AnnData,
    *,
    label_key: str = "cell_type",
    time_key: str = "time_point_processed",
) -> str:
    """Fingerprint actual annotations so pre-fix analysis caches are not reused."""
    values = adata.obs[[label_key, time_key]].copy()
    values[label_key] = values[label_key].astype(str)
    hashed = pd.util.hash_pandas_object(values, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def validate_ho_label_consistency(
    objects: Mapping[str, ad.AnnData],
    *,
    label_key: str = "cell_type",
    time_key: str = "time_point_processed",
) -> Dict[str, Any]:
    """Read-only checks of release provenance and matched-cell annotations.

    Compare by shared, unique obs_names, never by row position or population
    counts. Disjoint objects cannot be checked cell by cell and are reported.
    An applied marker is provenance, not proof of biological correctness: old
    releases affected by repeated swaps still need manuscript/data review.
    """
    report: Dict[str, Any] = {"objects": {}, "pairs": {}}
    for name, obj in objects.items():
        audit = obj.uns.get("mofi_ho_label_correction", {})
        if audit.get("status") != "applied" or audit.get("label_key") != label_key:
            raise ValueError(f"HO {name}: missing applied correction audit for {label_key}")
        if label_key not in obj.obs or time_key not in obj.obs:
            raise ValueError(f"HO {name}: missing label or time column")
        if not obj.obs_names.is_unique:
            raise ValueError(f"HO {name}: obs_names must be unique for label comparison")
        report["objects"][name] = {
            "n_obs": int(obj.n_obs),
            "labels_by_time": {
                str(t): group[label_key].astype(str).value_counts().to_dict()
                for t, group in obj.obs.groupby(time_key, observed=True)
            },
        }
    names = list(objects)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = objects[left], objects[right]
            common = a.obs_names.intersection(b.obs_names, sort=False)
            report["pairs"][f"{left}/{right}"] = {
                "shared_cells": int(len(common)),
                "status": "consistent" if len(common) else "no_shared_ids_unverified",
            }
            if not len(common):
                continue
            a_obs, b_obs = a.obs.loc[common], b.obs.loc[common]
            if not np.array_equal(a_obs[label_key].astype(str), b_obs[label_key].astype(str)):
                raise ValueError(f"HO {left}/{right}: labels disagree for shared cells")
            if not np.array_equal(a_obs[time_key].to_numpy(), b_obs[time_key].to_numpy()):
                raise ValueError(f"HO {left}/{right}: times disagree for shared cells")
    return report


def validate_ho_release_labels(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Inspect the three released HO objects without changing files."""
    spec = dataset_spec("ho")
    objects = {}
    try:
        for name, path in (("RNA", spec.primary_path), ("ATAC", spec.secondary_path),
                           ("dynamics", spec.dynamics_path)):
            objects[name] = ad.read_h5ad(_abs(path, repo_root), backed="r")
        return validate_ho_label_consistency(objects)
    finally:
        for obj in objects.values():
            obj.file.close()


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
    if dataset_id == "ho":
        validate_ho_release_labels(repo_root)
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
    if dataset_id == "ho":
        validate_ho_release_labels(repo_root)
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


    if dataset_id != "ho":
        shutil.copy2(primary_umap, primary_cache)
    shutil.copy2(secondary_umap, secondary_cache)

    dynamics = ad.read_h5ad(paths["dynamics"])
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
        sampling_key = spec.cell_type_key
        init_cell_type = None
        background_key = spec.cell_type_key
        sampling_mode = "index"
        init_indices = [9836, 8189, 3146, 6074, 410, 1431]
        n_trajectories = 6
        n_bins = 400

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
    primary_options = dict(common)
    if dataset_id == "ho":
        primary_options["dim_reduction"] = "umap_adata"
    primary_options["traj_cmap"] = "sunset1"
    plot_ode_trajectories_v2(
        **primary_options,
        save_base_name="ode_trajectories_primary.png",
    )
    if dataset_id == "gse213152":
        from CytoBridge.pl.plot import _apply_transport_map_array, apply_umap_transform
        from CytoBridge.inference import load_dynamics
        from CytoBridge.tl.analysis import generate_ode_trajectories
        with secondary_cache.open("rb") as stream:
            secondary_reducer = __import__("pickle").load(stream)
        secondary_background = np.asarray(secondary_reducer.embedding_)
        # The historical ATAC panel used a full-data dynamics model; the RNA
        # panel used a balanced model. Keep the two model inputs separate.
        legacy_bundle = _abs("paper_data/inference/gse213152/figure3_secondary_legacy", repo_root)
        legacy_data = ad.read_h5ad(legacy_bundle / "rna.h5ad")
        legacy_model = load_dynamics(legacy_bundle, device=device)
        # The bundle contains only the ten historical starting cells, in order.
        # Original full-input indices: 996, 8817, 1031, 7453, 5002,
        # 4937, 7958, 1089, 2330, 9778.
        legacy_indices = list(range(10))
        _, raw_trajectory, _ = generate_ode_trajectories(
            legacy_model, legacy_data, n_trajectories=10, n_bins=301,
            device=device, exp_dir=str(ode_dir / "secondary_model"),
            sampling_mode="index", init_time=0.0, init_indices=legacy_indices,
            cell_type_key="cell_type", strict_index_init_time=True,
            evaluation_times=[0, 1, 2, 3, 4], return_metadata=True,
        )
        secondary_mapper = build_transport_map(
            map_model_path=paths["training_map"], input_dim1=50, input_dim2=50,
            mode=(1, 2), hidden_dim=32, device=device, mapper_type="ae",
        )
        mapped_trajectory = _apply_transport_map_array(secondary_mapper, raw_trajectory, device=device)
        secondary_trajectories = apply_umap_transform(secondary_reducer, mapped_trajectory)
        np.save(ode_dir / "secondary_trajectory_computed.npy", secondary_trajectories)
        secondary_obs = pd.read_csv(_abs("paper_data/inference/gse213152/atac_background_obs.csv", repo_root))
        if secondary_background.shape != (len(secondary_obs), 2):
            raise ValueError("ATAC UMAP and background annotations have inconsistent lengths")
        secondary_times = np.asarray(secondary_obs[spec.time_key], dtype=float)
        secondary_unique_times = np.sort(np.unique(secondary_times))
        secondary_X = [
            secondary_background[np.isclose(secondary_times, time)]
            for time in secondary_unique_times
        ]
        secondary_labels = [
            secondary_obs.loc[
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
            traj_cmap="sunset1",
            show_axes=True,
        )
    else:
        plot_ode_trajectories_map_v2(
            **common,
            tranmap=mapper,
            save_base_name="ode_trajectories_secondary.png",
        )

    if dataset_id == "ho":
        dynamics.obsm["X_umap"] = np.load(ode_dir / "background_2d.npy")
    else:
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
        "umap_mode": "refit_umap_with_trajectories" if dataset_id == "ho" else "paper_selected",
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
        payload["secondary_projection_mode"] = "computed_trajectory_full_atac_umap"
        payload["secondary_dynamics_model"] = "full-data historical model"
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
