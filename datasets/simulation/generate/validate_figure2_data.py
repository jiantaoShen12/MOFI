#!/usr/bin/env python3
"""Validate regenerated Figure 2 data against a retained reference CSV."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SIMULATION_DIR = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=SIMULATION_DIR / "reference" / "simulation_gene.csv",
    )
    args = parser.parse_args()

    generated = SIMULATION_DIR / "simulation_gene.csv"
    ref_df = pd.read_csv(args.reference)
    gen_df = pd.read_csv(generated)
    if list(ref_df.columns) != list(gen_df.columns):
        raise AssertionError("CSV column mismatch")
    if ref_df.shape != gen_df.shape:
        raise AssertionError(f"CSV shape mismatch: {ref_df.shape} != {gen_df.shape}")
    max_abs = float(
        np.max(
            np.abs(
                ref_df.to_numpy(dtype=np.float64)
                - gen_df.to_numpy(dtype=np.float64)
            )
        )
    )
    if max_abs > 1e-12:
        raise AssertionError(f"Generated data mismatch, max_abs={max_abs}")

    print(f"Reference SHA256: {sha256(args.reference)}")
    print(f"Generated SHA256: {sha256(generated)}")
    print(f"Rows: {len(gen_df)}")
    print(f"Max absolute difference: {max_abs:.3e}")
    print("PASS: regenerated Figure 2 CSV matches the retained reference.")


if __name__ == "__main__":
    main()
