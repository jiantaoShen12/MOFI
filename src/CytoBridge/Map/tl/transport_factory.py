from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from CytoBridge.Map.tl.Mongemap import TransportMap
from CytoBridge.Map.tl.moscot_map import MoscotTransportMap


def _normalize_mapper_type(mapper_type: Optional[str]) -> str:
    if mapper_type is None:
        return "ae"
    return str(mapper_type).strip().lower()


def build_transport_map(
    *,
    map_model_path: Optional[str],
    input_dim1: int,
    input_dim2: int,
    mode: Tuple[int, int],
    hidden_dim: int = 16,
    device: str = "cpu",
    mapper_type: Optional[str] = None,
    mapper_kwargs: Optional[Dict[str, Any]] = None,
):
    mapper_type = _normalize_mapper_type(mapper_type)
    mapper_kwargs = {} if mapper_kwargs is None else dict(mapper_kwargs)

    if mapper_type in {"ae", "autoencoder", "transportmap", "monge"}:
        if map_model_path is None:
            raise ValueError("AE mapper requires `Map_model_path`.")
        return TransportMap(
            model_path=str(map_model_path),
            input_dim1=input_dim1,
            input_dim2=input_dim2,
            mode=mode,
            hidden_dim=hidden_dim,
            device=device,
        )

    if mapper_type in {"moscot", "ot"}:
        model_path = mapper_kwargs.pop("model_path", map_model_path)
        return MoscotTransportMap(
            model_path=None if model_path is None else str(model_path),
            input_dim1=input_dim1,
            input_dim2=input_dim2,
            mode=mode,
            hidden_dim=hidden_dim,
            device=device,
            **mapper_kwargs,
        )

    raise ValueError(f"Unsupported mapper_type: {mapper_type}. Use 'ae' or 'moscot'.")


def build_transport_map_from_multi_config(multi_cfg: Dict[str, Any], *, device: str):
    return build_transport_map(
        map_model_path=multi_cfg.get("Map_model_path"),
        input_dim1=int(multi_cfg.get("input_dim1", 50)),
        input_dim2=int(multi_cfg.get("input_dim2", 50)),
        mode=multi_cfg.get("mode", (1, 2)),
        hidden_dim=int(multi_cfg.get("hidden_dim", 32)),
        device=device,
        mapper_type=multi_cfg.get("mapper_type", "ae"),
        mapper_kwargs=multi_cfg.get("mapper_kwargs", {}),
    )

