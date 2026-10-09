#!/usr/bin/env python3
"""Run a notebook with an installed Jupyter kernel and save its outputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("JUPYTER_PATH", str(ROOT / ".jupyter"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / ".cache" / "jupyter" / "config"))
os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".cache" / "jupyter" / "data"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".cache" / "jupyter" / "runtime"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache" / "ipython"))

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--kernel", default="python3", help="Installed Jupyter kernel name (default: python3).")
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    nb = nbformat.read(notebook_path, as_version=4)
    print(f"EXECUTING {notebook_path.name}", flush=True)
    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
        record_timing=True,
    )
    client.execute()
    nbformat.write(nb, notebook_path)
    print(f"PASS {notebook_path.name}", flush=True)


if __name__ == "__main__":
    main()
