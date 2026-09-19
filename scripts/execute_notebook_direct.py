#!/usr/bin/env python3
"""Execute notebook cells directly when Windows Jupyter ACLs block kernels."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path


def _decode_png_repr(value):
    """Convert one or more base64 text layers to raw PNG bytes."""
    if not isinstance(value, str):
        return value
    candidate = value.encode()
    for _ in range(3):
        try:
            decoded = base64.b64decode(candidate)
        except Exception:
            break
        if decoded[:8] == b"\x89PNG\r\n\x1a\n":
            return decoded
        candidate = decoded
    return candidate


def _capture_mime(obj):
    # MOFI's original plotting functions create a Matplotlib Figure and close
    # it after saving.  The tutorial helper displays that same live Figure;
    # serialise it here so direct validation records the current run, too.
    if hasattr(obj, "savefig") and obj.__class__.__module__.startswith("matplotlib"):
        buffer = io.BytesIO()
        obj.savefig(buffer, format="png", bbox_inches="tight")
        return {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}
    if hasattr(obj, "_repr_png_"):
        value = obj._repr_png_()
        if value:
            if isinstance(value, str):
                # IPython.display.Image returns base64 text from _repr_png_;
                # decode it before the executor applies the notebook's one
                # required base64 encoding.  Otherwise the saved output is a
                # base64-of-base64 payload.
                value = _decode_png_repr(value)
            return {"image/png": base64.b64encode(value).decode("ascii")}
    if hasattr(obj, "_repr_jpeg_"):
        value = obj._repr_jpeg_()
        if value:
            if isinstance(value, str):
                try:
                    value = base64.b64decode(value, validate=True)
                except Exception:
                    value = value.encode()
            return {"image/jpeg": base64.b64encode(value).decode("ascii")}
    if hasattr(obj, "_repr_html_"):
        value = obj._repr_html_()
        if value:
            return {"text/html": value}
    if hasattr(obj, "_repr_markdown_"):
        value = obj._repr_markdown_()
        if value:
            return {"text/markdown": value}
    return {"text/plain": repr(obj)}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: execute_notebook_direct.py NOTEBOOK")
    notebook = Path(sys.argv[1]).resolve()
    root = notebook.parent.parent
    # Match the normal tutorial workflow: notebooks can import the small
    # release helper without carrying path boilerplate in every first cell.
    for import_path in (root, root / "src"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    os.environ["NUMBA_CACHE_DIR"] = str(root / ".numba_cache")
    os.environ["MPLCONFIGDIR"] = str(root / ".mplconfig")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONUTF8"] = "1"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(notebook), "__builtins__": __builtins__}
    import IPython.display as ipd
    import matplotlib.pyplot as plt

    captured = []
    original_display = ipd.display
    original_show = plt.show

    def capture_display(*objects, **kwargs):
        for obj in objects:
            captured.append({"output_type": "display_data", "data": _capture_mime(obj), "metadata": {}})

    def capture_show(*args, **kwargs):
        # In a real Jupyter kernel this is already an inline, freshly drawn
        # figure.  The direct executor records the same current figure.
        capture_display(plt.gcf())

    ipd.display = capture_display
    plt.show = capture_show
    try:
        count = 0
        for cell in payload.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            count += 1
            captured.clear()
            stdout = io.StringIO()
            stderr = io.StringIO()
            source = "".join(cell.get("source", []))
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(source, f"{notebook.name}:cell-{count}", "exec"), namespace)
            except Exception as exc:
                captured.append({"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc), "traceback": []})
                cell["outputs"] = ([{"output_type": "stream", "name": "stdout", "text": stdout.getvalue()}] if stdout.getvalue() else []) + captured
                cell["execution_count"] = count
                notebook.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                raise
            outputs = []
            if stdout.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
            if stderr.getvalue():
                outputs.append({"output_type": "stream", "name": "stderr", "text": stderr.getvalue()})
            outputs.extend(captured)
            cell["outputs"] = outputs
            cell["execution_count"] = count
            notebook.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    finally:
        ipd.display = original_display
        plt.show = original_show


if __name__ == "__main__":
    main()
