#!/usr/bin/env python3
"""Run the HSPC downstream pipeline for the currently active local models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / ".vendor", ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from CytoBridge.reproducibility import run_real_downstream  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_real_downstream(
        "hspc_31800",
        output_dir=args.output_dir,
        selected_figures=[
            "perturbation_zscore_sweep",
            "perturbation_sweep_heatmap_prob",
            "program_both",
            "key_molecule_volcano",
        ],
        device="cuda",
        sample_max_cells=None,
        reuse_cache=False,
        repo_root=ROOT,
    )
    print(result["run_dir"])


if __name__ == "__main__":
    main()
