#!/usr/bin/env python3
"""Run one-epoch training smoke tests for all four real paper datasets."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The original library reports progress with Unicode symbols.  Force a UTF-8
# stream so this standalone validator also works in a default Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from CytoBridge.reproducibility import PAPER_DATASETS, run_real_training_smoke


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for dataset_id in PAPER_DATASETS:
        print(f"\n=== {dataset_id} ===", flush=True)
        results[dataset_id] = run_real_training_smoke(
            dataset_id,
            output_dir=ROOT / "validation" / "smoke" / dataset_id,
            device=device,
            cells_per_time=8,
            epochs_per_stage=1,
            seed=42,
            repo_root=ROOT,
        )
        print(json.dumps(results[dataset_id], indent=2), flush=True)
    report = ROOT / "validation" / "real_training_smoke_summary.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nPASS: {report}")


if __name__ == "__main__":
    main()
