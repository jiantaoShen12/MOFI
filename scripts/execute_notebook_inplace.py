#!/usr/bin/env python3
"""Execute a notebook with the available Jupyter kernel client and embed outputs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from jupyter_client import KernelManager


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: execute_notebook_inplace.py NOTEBOOK")
    notebook = Path(sys.argv[1]).resolve()
    root = notebook.parent.parent
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    env = dict(os.environ)
    env.setdefault("NUMBA_CACHE_DIR", str(root / ".numba_cache"))
    env.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    env["JUPYTER_RUNTIME_DIR"] = str(root / ".jupyter_runtime")
    env.setdefault("PYTHONUTF8", "1")
    Path(env["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)

    km = KernelManager(kernel_name="python")
    km.start_kernel(cwd=str(root), env=env)
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=120)
    execution_count = 0
    try:
        for cell in payload.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            outputs = []
            execution_count += 1
            msg_id = kc.execute("".join(cell.get("source", [])), store_history=True)
            while True:
                msg = kc.get_iopub_msg(timeout=600)
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                msg_type = msg.get("msg_type")
                content = msg.get("content", {})
                if msg_type == "stream":
                    outputs.append({"output_type": "stream", "name": content.get("name", "stdout"), "text": content.get("text", "")})
                elif msg_type in {"execute_result", "display_data"}:
                    outputs.append({
                        "output_type": msg_type,
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                        **({"execution_count": execution_count} if msg_type == "execute_result" else {}),
                    })
                elif msg_type == "error":
                    outputs.append({
                        "output_type": "error",
                        "ename": content.get("ename", ""),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    })
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    break
            cell["outputs"] = outputs
            cell["execution_count"] = execution_count
            if outputs and outputs[-1].get("output_type") == "error":
                raise RuntimeError(outputs[-1].get("evalue", "notebook cell failed"))
            notebook.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
