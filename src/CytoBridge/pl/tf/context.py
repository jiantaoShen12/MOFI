from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np

from CytoBridge.Map.tl.transport_factory import build_transport_map
from CytoBridge.tl.analysis_dense_time import _load_pca_bundle
from CytoBridge.utils import load_model_from_adata


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _safe_float_list(values: Sequence[float]) -> List[float]:
    out = sorted({float(v) for v in values})
    if len(out) == 0:
        raise ValueError("empty time list")
    return out


def _pick_existing_times(full_grid: Sequence[float], requested: Optional[Sequence[float]]) -> List[float]:
    full = np.asarray(full_grid, dtype=float).reshape(-1)
    if full.size == 0:
        raise ValueError("full time grid is empty")
    if not requested:
        return [float(x) for x in full.tolist()]
    out: List[float] = []
    for t in requested:
        idx = np.where(np.isclose(full, float(t)))[0]
        if idx.size == 0:
            continue
        out.append(float(full[int(idx[0])]))
    if len(out) == 0:
        return [float(x) for x in full.tolist()]
    return _safe_float_list(out)


@dataclass
class TFRunContext:
    run_dir: Path
    assets_dir: Path
    tables_dir: Path
    reports_dir: Path
    manifest: Dict[str, Any]
    qc_metrics: Dict[str, Any]

    time_grid: List[float]
    time_key: str
    cell_type_key: str

    primary_domain: int
    secondary_domain: int

    domain1_processed_path: Path
    domain2_processed_path: Path
    primary_processed_path: Path
    secondary_processed_path: Path

    dynamic_adata_path: Path
    map_model_path: Path

    output_table_dir: Path
    output_figure_dir: Path

    _cache: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        *,
        output_table_dir: str = "",
        output_figure_dir: str = "",
    ) -> "TFRunContext":
        run = Path(run_dir).resolve()
        assets = run / "assets"
        tables = run / "tables"
        reports = run / "reports"
        manifest = _load_json(assets / "manifest.json")
        qc_path = assets / "qc_metrics.json"
        qc = _load_json(qc_path) if qc_path.exists() else {}

        tg = qc.get("time_grid", manifest.get("time_grid", []))
        time_grid = _safe_float_list([float(x) for x in list(tg)])

        time_key = str(manifest.get("time_key", "time_point_processed"))
        cell_type_key = str(manifest.get("cell_type_key", "cell_type"))

        primary_domain = int(manifest.get("primary_domain", 1))
        secondary_domain = int(manifest.get("secondary_domain", 2))

        d1 = Path(str(manifest.get("domain1_processed_path", manifest.get("main_processed_path", "")))).resolve()
        d2 = Path(str(manifest.get("domain2_processed_path", manifest.get("sub_processed_path", "")))).resolve()
        p1 = Path(str(manifest.get("primary_processed_path", ""))).resolve() if str(manifest.get("primary_processed_path", "")).strip() else d1
        p2 = Path(str(manifest.get("secondary_processed_path", ""))).resolve() if str(manifest.get("secondary_processed_path", "")).strip() else d2

        dyn = Path(str(manifest.get("dynamic_adata_path", ""))).resolve()
        map_model = Path(str(manifest.get("map_model_path", ""))).resolve()

        out_tables = Path(output_table_dir).resolve() if str(output_table_dir).strip() else (run / "tables" / "tf")
        out_figs = Path(output_figure_dir).resolve() if str(output_figure_dir).strip() else (run / "figures" / "tf")
        out_tables.mkdir(parents=True, exist_ok=True)
        out_figs.mkdir(parents=True, exist_ok=True)

        return cls(
            run_dir=run,
            assets_dir=assets,
            tables_dir=tables,
            reports_dir=reports,
            manifest=manifest,
            qc_metrics=qc,
            time_grid=time_grid,
            time_key=time_key,
            cell_type_key=cell_type_key,
            primary_domain=primary_domain,
            secondary_domain=secondary_domain,
            domain1_processed_path=d1,
            domain2_processed_path=d2,
            primary_processed_path=p1,
            secondary_processed_path=p2,
            dynamic_adata_path=dyn,
            map_model_path=map_model,
            output_table_dir=out_tables,
            output_figure_dir=out_figs,
        )

    def resolve_time_points(self, requested: Optional[Sequence[float]] = None) -> List[float]:
        return _pick_existing_times(self.time_grid, requested)

    def latest_perturbation_sweep_csv(self) -> Path:
        candidates = sorted(
            self.tables_dir.glob("perturbation_zscore_sweep__*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"no perturbation_zscore_sweep__*.csv under {self.tables_dir}")
        return candidates[0]

    def load_primary_secondary_processed(self, *, backed: bool = False) -> Tuple[ad.AnnData, ad.AnnData]:
        key = f"proc::{int(backed)}"
        if key in self._cache:
            return self._cache[key]
        mode = "r" if backed else None
        primary = ad.read_h5ad(str(self.primary_processed_path), backed=mode)
        secondary = ad.read_h5ad(str(self.secondary_processed_path), backed=mode)
        self._cache[key] = (primary, secondary)
        return primary, secondary

    def load_domain12_processed(self, *, backed: bool = False) -> Tuple[ad.AnnData, ad.AnnData]:
        key = f"domain12::{int(backed)}"
        if key in self._cache:
            return self._cache[key]
        mode = "r" if backed else None
        d1 = ad.read_h5ad(str(self.domain1_processed_path), backed=mode)
        d2 = ad.read_h5ad(str(self.domain2_processed_path), backed=mode)
        self._cache[key] = (d1, d2)
        return d1, d2

    def load_dynamic_adata(self) -> ad.AnnData:
        key = "dyn"
        if key in self._cache:
            return self._cache[key]
        dyn = ad.read_h5ad(str(self.dynamic_adata_path))
        self._cache[key] = dyn
        return dyn

    def load_model(self) -> Any:
        key = "model"
        if key in self._cache:
            return self._cache[key]
        model = load_model_from_adata(self.load_dynamic_adata())
        self._cache[key] = model
        return model

    def _domain_latent_dims(self) -> Tuple[int, int]:
        key = "domain_dims"
        if key in self._cache:
            return self._cache[key]
        d1, d2 = self.load_domain12_processed(backed=True)
        dim1 = int(d1.obsm["X_latent"].shape[1])
        dim2 = int(d2.obsm["X_latent"].shape[1])
        if getattr(d1, "isbacked", False):
            d1.file.close()
        if getattr(d2, "isbacked", False):
            d2.file.close()
        self._cache[key] = (dim1, dim2)
        return dim1, dim2

    def build_mapper(self, *, device: str = "cpu") -> Any:
        key = f"mapper::{str(device)}"
        if key in self._cache:
            return self._cache[key]

        dyn = self.load_dynamic_adata()
        multi_cfg = dyn.uns.get("all_model", {}).get("model_config", {}).get("multi", {})
        hidden_dim = int(multi_cfg.get("hidden_dim", 64))
        input_dim1, input_dim2 = self._domain_latent_dims()

        mapper = build_transport_map(
            map_model_path=str(self.map_model_path),
            input_dim1=input_dim1,
            input_dim2=input_dim2,
            mode=(int(self.primary_domain), int(self.secondary_domain)),
            hidden_dim=hidden_dim,
            device=str(device),
            mapper_type=str(self.manifest.get("mapper_type", "ae")),
            mapper_kwargs=dict(self.manifest.get("mapper_kwargs", {})),
        )
        self._cache[key] = mapper
        return mapper

    def load_classifier_bundle(self) -> Dict[str, Any]:
        key = "classifier_bundle"
        if key in self._cache:
            return self._cache[key]

        manifest_path = self.reports_dir / "classifiers" / "classifier_manifest.json"
        cls_manifest = _load_json(manifest_path)
        cls_path = Path(str(cls_manifest["classifier_path"]))
        with cls_path.open("rb") as f:
            payload = pickle.load(f)

        classes = np.asarray(
            cls_manifest.get("cell_type_output_classes", payload.get("classes", [])),
            dtype=object,
        )
        if classes.size == 0:
            classes = np.asarray(payload.get("classes", []), dtype=object)

        out = {
            "classifier": payload["classifier"],
            "classes": classes,
            "train_space": str(cls_manifest.get("train_space", "joint")).strip().lower(),
            "map_batch": int(cls_manifest.get("joint_map_batch", 2048)),
            "manifest": cls_manifest,
        }
        self._cache[key] = out
        return out

    def load_primary_secondary_bundles(self) -> Tuple[Any, Any]:
        key = "pca_bundles"
        if key in self._cache:
            return self._cache[key]
        primary, secondary = self.load_primary_secondary_processed(backed=False)
        b1 = _load_pca_bundle(primary, space_name="primary_processed")
        b2 = _load_pca_bundle(secondary, space_name="secondary_processed")
        self._cache[key] = (b1, b2)
        return b1, b2

    def load_traj_arrays(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        key = "traj"
        if key in self._cache:
            return self._cache[key]
        traj_primary = np.asarray(np.load(self.assets_dir / "traj_main.npy", allow_pickle=False), dtype=np.float32)
        traj_secondary = np.asarray(np.load(self.assets_dir / "traj_sub.npy", allow_pickle=False), dtype=np.float32)
        weights = None
        w_path = self.assets_dir / "weights.npy"
        if w_path.exists():
            arr = np.asarray(np.load(w_path, allow_pickle=False), dtype=np.float32)
            if arr.shape == traj_primary.shape[:2]:
                weights = np.asarray(arr, dtype=np.float32)
        self._cache[key] = (traj_primary, traj_secondary, weights)
        return traj_primary, traj_secondary, weights

    def load_init_indices_from_pred_main(self) -> np.ndarray:
        key = "init_indices"
        if key in self._cache:
            return self._cache[key]

        pred_main_path = Path(str(self.manifest["output_files"]["pred_main_h5ad"]))
        pred_main = ad.read_h5ad(str(pred_main_path))
        obs = pred_main.obs.copy()
        if "pred_time" not in obs.columns or "source_cell_index" not in obs.columns:
            raise ValueError(f"pred_main missing required obs columns in {pred_main_path}")
        t0 = float(np.min(np.asarray(obs["pred_time"], dtype=float)))
        first = obs[np.isclose(np.asarray(obs["pred_time"], dtype=float), t0)].copy()
        if "trajectory_index" in first.columns:
            first = first.sort_values("trajectory_index")
        init_idx = np.asarray(first["source_cell_index"], dtype=int)
        self._cache[key] = init_idx
        return init_idx

    def close_backed(self) -> None:
        for key in ["proc::1", "domain12::1"]:
            if key in self._cache:
                items = self._cache[key]
                if isinstance(items, tuple):
                    for x in items:
                        if getattr(x, "isbacked", False):
                            x.file.close()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "primary_domain": int(self.primary_domain),
            "secondary_domain": int(self.secondary_domain),
            "primary_processed_path": str(self.primary_processed_path),
            "secondary_processed_path": str(self.secondary_processed_path),
            "domain1_processed_path": str(self.domain1_processed_path),
            "domain2_processed_path": str(self.domain2_processed_path),
            "time_grid": [float(x) for x in self.time_grid],
            "time_key": self.time_key,
            "cell_type_key": self.cell_type_key,
            "output_table_dir": str(self.output_table_dir),
            "output_figure_dir": str(self.output_figure_dir),
        }
