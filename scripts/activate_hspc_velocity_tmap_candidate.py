#!/usr/bin/env python3
"""Activate an already audited HSPC velocity+tmap candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    dyn_dir = ROOT / "paper_data" / "dynamics" / "hspc_31800"
    map_dir = ROOT / "paper_data" / "maps" / "hspc_31800"
    candidate_dyn = dyn_dir / "adata_velocity_tmap_finetuned.h5ad"
    active_dyn = dyn_dir / "adata.h5ad"
    candidate_map = map_dir / "best_model_velocity_tmap_finetuned.pt"
    active_map = map_dir / "best_model.pt"
    shutil.copy2(candidate_dyn, active_dyn)
    shutil.copy2(candidate_map, active_map)
    audit_path = ROOT / "validation" / "hspc_velocity_tmap_finetune_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(
        {
            "activated": True,
            "active_dynamics_path": str(active_dyn.relative_to(ROOT)),
            "active_map_path": str(active_map.relative_to(ROOT)),
            "active_dynamics_sha256": sha256(active_dyn),
            "active_map_sha256": sha256(active_map),
        }
    )
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"activated": True, "dynamics": str(active_dyn), "map": str(active_map)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
