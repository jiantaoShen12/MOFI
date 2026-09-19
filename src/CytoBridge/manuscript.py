"""High-level Python entry points used by the manuscript notebooks.

The notebooks are intentionally thin clients of this module.  The numerical
methods remain in :mod:`CytoBridge`; this module only resolves the release
assets, connects existing plotting APIs, and records the small validation
contracts needed to reproduce the paper panels.  The command-line scripts
remain available for users who prefer a shell workflow, but notebook code does
not execute them through ``sys.argv`` or duplicate their implementations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .reproducibility import (
    REPO_ROOT,
    _abs,
    dataset_spec,
    load_yaml,
    require_paper_data,
    resolved_analysis_config,
    sha256_file,
)

__all__ = [
    "build_figure2_inputs",
    "train_figure2",
    "evaluate_figure2",
    "render_figure2",
    "render_synchronized_corridor",
    "render_hspc_time_slices",
    "render_hc_time_slices",
    "render_hc_lineage",
    "plot_marker_bridging",
    "validate_archived_volcano_sources",
    "build_transformation_summary",
]


def _root_path(path: str | Path, repo_root: Path) -> Path:
    return _abs(path, repo_root).resolve()


def build_figure2_inputs(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Build and validate the public Figure 2 AnnData inputs.

    This is the library form of the deterministic CSV-to-AnnData preparation;
    it does not regenerate or perturb the retained simulation values.
    """

    simulation_dir = _root_path("datasets/simulation", repo_root)
    csv_path = simulation_dir / "simulation_gene.csv"
    reference_path = simulation_dir / "reference" / "simulation_gene.csv"
    df = pd.read_csv(csv_path)
    expected_columns = ["samples", "x1", "x2", "x3", "x4"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")

    obs = pd.DataFrame(index=pd.RangeIndex(len(df)).astype(str))
    obs["samples"] = df["samples"].to_numpy(dtype=np.float64)

    def make_adata(columns: Sequence[str]) -> ad.AnnData:
        return ad.AnnData(X=df[list(columns)].to_numpy(dtype=np.float64), obs=obs.copy())

    adata_2d = make_adata(["x1", "x2"])
    adata_3d = make_adata(["x1", "x2", "x4"])
    adata_2d_path = simulation_dir / "simulation4_2d.h5ad"
    adata_3d_path = simulation_dir / "simulation4_3d.h5ad"
    merged_path = simulation_dir / "simulation4_merged.h5ad"
    adata_2d.write_h5ad(adata_2d_path)
    adata_3d.write_h5ad(adata_3d_path)

    merged = ad.AnnData(
        X=np.hstack([np.asarray(adata_2d.X), np.asarray(adata_3d.X)]),
        obs=adata_2d.obs.copy(),
    )
    merged.var_names = ["mod1_0", "mod1_1", "mod2_0", "mod2_1", "mod2_2"]
    merged.var["feature_modality"] = ["mod1", "mod1", "mod2", "mod2", "mod2"]
    merged.write_h5ad(merged_path)

    expected_counts = {0.0: 400, 0.75: 426, 1.5: 464, 2.25: 539, 3.0: 623}
    counts = df["samples"].value_counts().sort_index().astype(int).to_dict()
    if counts != expected_counts:
        raise RuntimeError(f"Figure 2 cell counts changed: {counts}")
    ref_df = pd.read_csv(reference_path)
    if list(ref_df.columns) != list(df.columns) or ref_df.shape != df.shape:
        raise AssertionError("Figure 2 reference and generated CSV shapes differ")
    max_abs = float(np.max(np.abs(ref_df.to_numpy(dtype=float) - df.to_numpy(dtype=float))))
    if max_abs > 1e-12:
        raise AssertionError(f"Figure 2 generated data differ from reference: {max_abs}")

    return {
        "csv": csv_path,
        "reference_csv": reference_path,
        "adata_2d": adata_2d_path,
        "adata_3d": adata_3d_path,
        "merged": merged_path,
        "rows": int(len(df)),
        "counts": counts,
        "max_abs_difference_to_reference": max_abs,
    }


def train_figure2(
    *,
    output_dir: str | Path,
    device: str = "cuda",
    repo_root: Path = REPO_ROOT,
    config_path: str | Path = "configs/training/simulation_figure2.yaml",
    epochs_override: Optional[int] = None,
) -> dict[str, Any]:
    """Train the Figure 2 model with the same public MOFI fit API as the paper."""

    from CytoBridge.pp import preprocess
    from CytoBridge.tl.fit import fit
    from CytoBridge.utils.utils import set_seed

    root = Path(repo_root).resolve()
    simulation_dir = root / "datasets" / "simulation"
    data_path = simulation_dir / "simulation4_2d.h5ad"
    simulation_csv = simulation_dir / "simulation_gene.csv"
    cfg_path = _root_path(config_path, root)
    out = _root_path(output_dir, root)
    out.mkdir(parents=True, exist_ok=True)

    time_mapping = {0.0: 0.0, 0.75: 1.0, 1.5: 2.0, 2.25: 3.0, 3.0: 4.0}
    set_seed(22)
    trained_input = preprocess(
        ad.read_h5ad(data_path),
        time_key="samples",
        time_mapping=time_mapping,
        dim_reduction="none",
        normalization=False,
        log1p=False,
        select_hvg=False,
    )
    simulation_df = pd.read_csv(simulation_csv)
    auxiliary = ad.AnnData(simulation_df[["x1", "x2", "x4"]].to_numpy(dtype="float32"))
    auxiliary.obs["samples"] = simulation_df["samples"].to_numpy()
    _ = preprocess(
        auxiliary,
        time_key="samples",
        time_mapping=time_mapping,
        dim_reduction="none",
        normalization=False,
        log1p=False,
        select_hvg=False,
    )

    config = load_yaml(cfg_path)
    config["ckpt_dir"] = str(out)
    if epochs_override is not None:
        if epochs_override <= 0:
            raise ValueError("epochs_override must be positive")
        for stage in config["training"]["plan"]:
            stage["epochs"] = int(epochs_override)

    paper_parameters = epochs_override is None
    started = time.perf_counter()
    fitted = fit(trained_input, config=config, batch_size=400, device=device)
    elapsed = float(time.perf_counter() - started)
    output_adata = out / "adata.h5ad"
    fitted.write_h5ad(output_adata)
    manifest = {
        "paper_parameters": paper_parameters,
        "seed": 22,
        "data_path": str(data_path),
        "simulation_csv": str(simulation_csv),
        "config_path": str(cfg_path),
        "output_adata": str(output_adata),
        "device": str(device),
        "time_mapping": time_mapping,
        "resolved_config": config,
        "training_seconds": elapsed,
    }
    (out / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"output_adata": output_adata, "manifest": out / "reproduction_manifest.json", **manifest}


def evaluate_figure2(
    adata_path: str | Path,
    *,
    train_dir: str | Path,
    config_path: str | Path,
    device: str = "cuda",
    training_seconds: Optional[float] = None,
    val: int = 200,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Evaluate Figure 2 with :class:`CytoBridge.tl.trainer.TrainingPipeline`."""

    import torch
    from CytoBridge.tl.trainer import TrainingPipeline
    from CytoBridge.utils import load_model_from_adata

    trained = ad.read_h5ad(_root_path(adata_path, repo_root))
    cfg = load_yaml(_root_path(config_path, repo_root))
    output_dir = _root_path(train_dir, repo_root)
    cfg["ckpt_dir"] = str(output_dir)
    times = sorted(float(x) for x in trained.obs["time_point_processed"].unique())
    tensors = [
        torch.tensor(
            np.asarray(trained[trained.obs["time_point_processed"] == value].X),
            dtype=torch.float32,
        )
        for value in times
    ]
    evaluator = TrainingPipeline(
        load_model_from_adata(trained), cfg, 400, device, tensors, progress_callback=False
    )
    metrics = evaluator.evaluate(trained, tensors, times, val=val)
    if training_seconds is not None:
        metrics["training_seconds"] = float(training_seconds)
    evaluation_path = output_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def render_figure2(
    *,
    adata_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    sampling_seed: int | None = 22,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Render Figure 2 through ``CytoBridge.pl`` with reproducible trajectory sampling."""

    from CytoBridge.pl import plot as cb_pl
    from CytoBridge.utils import load_model_from_adata
    from CytoBridge.utils.utils import set_seed

    root = Path(repo_root).resolve()
    out = _root_path(output_dir, root)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(22)
    adata = ad.read_h5ad(_root_path(adata_path, root))
    model = load_model_from_adata(adata)
    model.to(device)
    for name in ("predicted_growth.png", "predicted_growth.pdf"):
        cb_pl.plot_growth(adata, dim_reduction="none", output_path=str(out / name))
    cb_pl.plot_ode_trajectories_v2(
        adata=adata,
        model=model,
        output_path=str(out),
        n_trajectories=25,
        n_bins=40,
        dim_reduction="none",
        device=device,
        sampling_seed=sampling_seed,
    )
    cb_pl.plot_landscape(adata, model, output_path=str(out), dim_reduction="none", device=device)
    if "velocity_latent" in adata.obsm:
        cb_pl.plot_velocity_stream(adata, model, str(out), dim_reduction="none", device=device)

    source = pd.read_csv(root / "datasets" / "simulation" / "simulation_gene.csv")
    x_csv = source[["x1", "x2"]].to_numpy(dtype=np.float64)
    x_adata = np.asarray(adata.X, dtype=np.float64)
    if x_csv.shape != x_adata.shape or np.max(np.abs(x_csv - x_adata)) > 1e-8:
        raise ValueError("CSV and Figure 2 AnnData rows are not aligned")
    true_adata = adata.copy()
    b_values = source["x2"].to_numpy(dtype=np.float64)
    true_adata.obsm["growth_rate"] = (0.05 * (b_values**2 / (1.0 + b_values**2)))[:, None].astype(np.float32)
    for name in ("true_growth_from_generator.png", "true_growth_from_generator.pdf"):
        cb_pl.plot_growth(true_adata, dim_reduction="none", output_path=str(out / name))
    return out


def render_synchronized_corridor(
    dataset_id: str,
    *,
    output_dir: str | Path,
    device: str = "cuda",
    direction: str = "1to2",
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Render a synchronized multimodal corridor for a released dataset."""

    from downstream.plotting.plot_sync_multimodal_time_corridor_generic import render

    spec = dataset_spec(dataset_id)
    paths = require_paper_data(dataset_id, repo_root)
    kwargs: dict[str, Any] = {
        "dynamic_adata": paths["dynamics"],
        "space1_processed": paths["primary"],
        "space2_processed": paths["secondary"],
        "map_model": paths["analysis_map"],
        "direction": direction,
        "time_key": spec.time_key,
        "device": device,
        "output_dir": str(_root_path(output_dir, repo_root)),
    }
    if dataset_id in {"gse213152", "ho"}:
        umap_dir = _root_path(f"paper_data/umap/{dataset_id}", repo_root)
        kwargs.update(
            space1_umap_model=str(umap_dir / "primary.pkl"),
            space2_umap_model=str(umap_dir / "secondary.pkl"),
        )
    return Path(render(**kwargs))


def render_hspc_time_slices(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Render the HSPC Figure 4 time-slice panel via its Python renderer."""

    from downstream.plotting.plot_sync_multimodal_time_slices_31800 import render

    # Preserve the original notebook contract: Figure 4 uses the archived
    # paper UMAP estimators and must not refit them on the current machine.
    umap_dir = _root_path("paper_data/umap/hspc_31800", repo_root)
    return Path(
        render(
            config=str(_root_path(config_path, repo_root)),
            output_dir=str(_root_path(output_dir, repo_root)),
            rna_umap_model=umap_dir / "time_primary.pkl",
            sub_umap_model=umap_dir / "time_secondary.pkl",
        )
    )


def render_hc_time_slices(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Render the HC author-5 Figure 6 time-slice panel."""

    from downstream.plotting.plot_sync_multimodal_time_slices_hc_author5_12 import render

    return Path(
        render(
            config=str(_root_path(config_path, repo_root)),
            output_dir=str(_root_path(output_dir, repo_root)),
            seed=int(seed),
        )
    )


def render_hc_lineage(
    *,
    run_dir: str | Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Render the HC author-5 lineage Sankey from an analysis run."""

    from downstream.plotting.run_hc_author5_lineage_sankey import render

    return Path(render(run_dir=_root_path(run_dir, repo_root)))


def plot_marker_bridging(
    dataset_id: str,
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    markers_by_modality: Mapping[str, Sequence[str]],
    output_names: Optional[Mapping[str, str]] = None,
    config_output_dir: Optional[str | Path] = None,
    device: str = "cuda",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Path]:
    """Plot marker bridging from the released dense-time tables."""

    from CytoBridge.pl.plot_dense_time import plot_intermediate_bridging

    run = _root_path(run_dir, repo_root)
    out = _root_path(output_dir, repo_root)
    out.mkdir(parents=True, exist_ok=True)
    cfg = resolved_analysis_config(
        dataset_id,
        output_dir=config_output_dir or run.parent,
        device=device,
        repo_root=repo_root,
    )["figure"]
    outputs: dict[str, Path] = {}
    for modality, markers in markers_by_modality.items():
        normalized = str(modality).lower().replace("primary", "main_var").replace("secondary", "sec_var")
        if normalized not in {"main_var", "sec_var"}:
            raise ValueError("markers_by_modality keys must be main_var/sec_var")
        marker_token = "_".join(str(marker) for marker in markers)
        stem = (output_names or {}).get(normalized, marker_token)
        output_path = out / f"{stem}.pdf"
        plot_intermediate_bridging(
            str(run / "tables" / f"{normalized}_marker_pred.csv"),
            str(run / "tables" / f"{normalized}_marker_real.csv"),
            str(output_path),
            cfg=cfg,
            markers=list(markers),
        )
        outputs[normalized] = output_path
    return outputs


def validate_archived_volcano_sources(
    *,
    repo_root: Path = REPO_ROOT,
    dataset_id: str = "ho",
    expected_sha256: Mapping[str, str],
    expected_rows: int = 2000,
) -> dict[str, Any]:
    """Validate the archived CSV inputs used by a manuscript volcano panel."""

    if dataset_id != "ho":
        raise ValueError("Only the archived HO volcano contract is defined in this release")
    root = _root_path("paper_data/archived/ho_figure5", repo_root)
    tables: dict[str, pd.DataFrame] = {}
    for filename, digest in expected_sha256.items():
        path = root / filename
        if sha256_file(path) != digest:
            raise AssertionError(f"Archive checksum mismatch: {path}")
        table = pd.read_csv(path)
        if table.shape[0] != expected_rows:
            raise AssertionError(f"Unexpected volcano table shape for {path}: {table.shape}")
        tables[filename] = table
    return {"directory": root, "tables": tables, "sha256": dict(expected_sha256)}


def build_transformation_summary(
    transformation: Mapping[str, Any],
    *,
    selected_csv: str | Path,
    output_csv: str | Path,
    repo_root: Path = REPO_ROOT,
) -> pd.DataFrame:
    """Create the compact Figure 4 transformation table from library outputs."""

    selected = pd.read_csv(_root_path(selected_csv, repo_root))
    main_matrix = pd.read_csv(transformation["outputs"]["main_variable_matrix_csv"]).set_index("main_variable")
    sub_matrix = pd.read_csv(transformation["outputs"]["sub_variable_matrix_csv"]).set_index("sub_variable")
    aliases = {"integrin beta7": "integrinB7", "CD49d (alpha4)": "CD49d", "CD29 (beta1)": "CD29"}
    rows: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        feature = str(row["feature"])
        if row["feature_type"] == "gene":
            matrix, key, formula = main_matrix, feature, "main_forward @ J_main_to_sub"
        else:
            matrix, key, formula = sub_matrix, aliases.get(feature, feature), "sub_forward @ J_sub_to_main"
        if key not in matrix.index:
            rows.append({**row, "matrix_feature": key, "status": "missing", "transformation_l2": float("nan"), "formula": formula})
            continue
        vector = matrix.loc[key].to_numpy(dtype=float)
        top = sorted(range(len(vector)), key=lambda index: abs(vector[index]), reverse=True)[:5]
        rows.append({
            **row,
            "matrix_feature": key,
            "status": "ok",
            "transformation_l2": float(np.linalg.norm(vector)),
            "top_latent_components": "|".join(f"{index}:{vector[index]:.6g}" for index in top),
            "formula": formula,
        })
    result = pd.DataFrame(rows)
    output = _root_path(output_csv, repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
