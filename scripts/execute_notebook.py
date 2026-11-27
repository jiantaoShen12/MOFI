#!/usr/bin/env python3
"""Run a notebook with an installed Jupyter kernel and save its outputs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["JUPYTER_PATH"] = str(ROOT / ".jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / ".cache" / "jupyter" / "config"))
os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".cache" / "jupyter" / "data"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".cache" / "jupyter" / "runtime"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache" / "ipython"))
for key in ("JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "JUPYTER_RUNTIME_DIR", "IPYTHONDIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--kernel", default="mofi-current", help="Kernel name; default uses this Python interpreter.")
    parser.add_argument("--output", type=Path, default=None, help="Executed notebook path (defaults to results/executed/).")
    args = parser.parse_args()
    if args.kernel == "mofi-current":
        kernel_dir = ROOT / ".jupyter" / "kernels" / "mofi-current"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "MOFI current environment", "language": "python"
        }), encoding="utf-8")
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
    output = args.output or ROOT / "results" / "executed" / notebook_path.name
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output)
    print(f"PASS {notebook_path.name}", flush=True)


if __name__ == "__main__":
    main()
