"""Notebook setup and figure display helpers."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


def display_png_file(path: str | Path) -> None:
    """Display a PNG from bytes without asking IPython to re-encode it.

    ``IPython.display.Image(filename=...)`` can produce a nested base64
    payload in some direct-execution paths.  Passing decoded PNG bytes keeps
    the saved notebook output a normal ``image/png`` object.
    """
    from IPython.display import Image, display

    png_path = Path(path)
    if not png_path.is_file():
        raise FileNotFoundError(png_path)
    payload = png_path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {png_path}")
    display(Image(data=payload, format="png"))


def show_paper_panel(path: str | Path) -> None:
    """Display a manuscript panel in the current notebook cell."""
    display_png_file(path)


def enable_live_plot_output() -> None:
    """Display figures at the moment original MOFI code closes them.

    Most publication plotting functions save a file and then call
    ``plt.close(fig)``.  This temporary hook preserves that original plotting
    route while also sending the freshly created Matplotlib figure to the
    current notebook output. It never reloads a saved PNG.
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    if getattr(plt, "_mofi_live_close_enabled", False):
        return
    original_close = plt.close

    def close_and_display(fig=None):
        if getattr(plt, "_mofi_live_display_enabled", True) and fig != "all":
            target = plt.gcf() if fig is None else fig
            if hasattr(target, "savefig"):
                display(target)
        return original_close(fig)

    plt.close = close_and_display
    plt._mofi_live_close_enabled = True


@contextmanager
def suppress_live_plot_output():
    """Keep an auxiliary original plot on disk without adding notebook output.

    The plotting function itself still runs unchanged.  Tutorials use this for
    supporting tables and intermediate figures that are needed for a later
    panel but would otherwise crowd the reader's notebook.
    """
    import matplotlib.pyplot as plt

    previous = getattr(plt, "_mofi_live_display_enabled", True)
    previous_show = plt.show
    plt._mofi_live_display_enabled = False
    # The direct notebook executor captures plt.show() independently of the
    # close-hook above.  Silence it here as well so an auxiliary original
    # plotting call cannot leak an extra inline panel.
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt._mofi_live_display_enabled = previous
        plt.show = previous_show


def find_repository(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook, working directory, or script."""
    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "src" / "CytoBridge").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise RuntimeError(
        "MOFI repository not found. Start Jupyter from the repository root "
        "or pass root=... to setup_notebook()."
    )


def setup_notebook(root: str | Path | None = None, *, device: str = "auto"):
    """Configure one tutorial notebook and return ``ROOT, DEVICE, runner``.

    ``device='auto'`` selects CUDA when available and otherwise CPU.  All
    runtime caches are kept in the repository's .cache directory.
    """
    root_path = find_repository(root)
    src_path = root_path / "src"
    for path in (src_path, root_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    cache_dirs = {
        "NUMBA_CACHE_DIR": root_path / ".cache" / "numba",
        "MPLCONFIGDIR": root_path / ".cache" / "matplotlib",
    }
    defaults = {
        **{key: str(value) for key, value in cache_dirs.items()},
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "42",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        # Keep tutorial runs predictable on workstations that expose hundreds
        # of logical CPUs; unrestricted joblib workers can hit Windows ACLs.
        "LOKY_MAX_CPU_COUNT": "1",
        "NUMBA_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    for path in cache_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    import CytoBridge
    import torch
    enable_live_plot_output()

    loaded_library = Path(CytoBridge.__file__).resolve()
    expected_library = (src_path / "CytoBridge").resolve()
    if expected_library not in loaded_library.parents:
        raise RuntimeError(
            f"Loaded CytoBridge from {loaded_library}, expected the release copy in {expected_library}"
        )
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"

    def run_original_script(relative_script: str | Path, *args, inline: bool = True):
        """Run a released script, optionally in this kernel for live figures."""
        script = root_path / relative_script
        command = [sys.executable, str(script), *map(str, args)]
        print("$", " ".join(command))
        if inline:
            previous_argv = sys.argv
            try:
                sys.argv = [str(script), *map(str, args)]
                runpy.run_path(str(script), run_name="__main__")
            finally:
                sys.argv = previous_argv
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            # Prefer the active environment's scientific stack.  The vendor
            # tree remains a fallback for lightweight utilities, but its
            # bundled Plotly/Kaleido can be incompatible with a user's CUDA
            # environment and force a fragile browser export on Windows.
            [str(src_path), str(root_path), str(root_path / ".vendor"), env.get("PYTHONPATH", "")]
        )
        subprocess.run(command, cwd=root_path, env=env, check=True)

    return root_path, selected_device, run_original_script
