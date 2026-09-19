"""Small, user-facing helpers shared by the five MOFI tutorial notebooks.

The notebooks deliberately keep this call short.  Environment configuration,
repository discovery, library provenance checks, and display helpers live here
so that readers can focus on the biological analysis rather than notebook
plumbing.
"""

from __future__ import annotations

import json
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
    """Display exactly one manuscript panel in the current notebook cell."""
    display_png_file(path)


def _legacy_show_paper_figure(
    panels: list[str | Path],
    *,
    columns: int = 2,
    title: str | None = None,
    tile_width: int = 900,
) -> None:
    """Show a manuscript panel contract as one notebook figure.

    This is a layout-only operation: it loads the exact saved panel PNGs
    produced by the released analysis, preserves their pixels, and places
    them on one readable contact sheet.  It does not resample data, redraw
    trajectories, or invent additional perturbation conditions.
    """
    from io import BytesIO

    from PIL import Image as PILImage, ImageDraw
    from IPython.display import Image, display

    paths = [Path(panel) for panel in panels]
    if not paths:
        raise ValueError("show_paper_figure requires at least one panel")
    if columns < 1:
        raise ValueError("columns must be positive")

    source_images = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        if payload[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG file: {path}")
        source_images.append(PILImage.open(BytesIO(payload)).convert("RGB"))

    tile_height = max(1, int(tile_width * 0.72))
    margin = 24
    title_height = 56 if title else 0
    rows = (len(source_images) + columns - 1) // columns
    canvas = PILImage.new(
        "RGB",
        (columns * tile_width + (columns + 1) * margin,
         title_height + rows * tile_height + (rows + 1) * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((margin, 16), title, fill="black")

    for index, source in enumerate(source_images):
        row, column = divmod(index, columns)
        tile = source.copy()
        tile.thumbnail((tile_width - 2 * margin, tile_height - 2 * margin))
        x = margin + column * tile_width + (tile_width - tile.width) // 2
        y = title_height + margin + row * tile_height + (tile_height - tile.height) // 2
        canvas.paste(tile, (x, y))

    output = BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    display(Image(data=output.getvalue(), format="png"))


def _paper_figure_panels(
    figure: int | str,
    *,
    root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> tuple[str, list[Path]]:
    """Resolve the released, manuscript-selected PNG contract for one figure."""
    figure_id = str(figure).strip().lower().replace("figure", "").strip()
    if figure_id not in {"2", "3", "4", "5", "6"}:
        raise ValueError("figure must be one of 2, 3, 4, 5, or 6")
    root_path = find_repository(root) if root is not None else None
    base = Path(run_dir).resolve() if run_dir is not None else None

    if figure_id == "2":
        base = base or ((root_path or find_repository()) / "results" / "figure2_reproduced")
        paths = [
            base / "Potential_landscape.png",
            base / "ode_results" / "ode_trajectories_v2.png",
            base / "predicted_growth.png",
            base / "true_growth_from_generator.png",
        ]
    elif figure_id == "3":
        if base is None:
            raise ValueError("Figure 3 requires run_dir=...")
        paths = [
            base / "figures" / "synchronized_corridor" / "sync_multimodal_time_corridor.png",
            base / "figures" / "cell_type_fate_dynamics.png",
            base / "figures" / "manuscript_markers" / "HNF1B_PAX8.png",
            base / "figures" / "manuscript_markers" / "EYA1_PAX8.png",
            base / "figures" / "paper_ode_v3" / "ode_results" / "ode_trajectories_primary.png",
            base / "figures" / "paper_ode_v3" / "ode_results" / "ode_trajectories_secondary.png",
            base / "figures" / "paper_ode_v3" / "growth_primary.png",
            base / "figures" / "paper_ode_v3" / "growth_secondary.png",
        ]
    elif figure_id == "4":
        if base is None:
            raise ValueError("Figure 4 requires run_dir=...")
        trajectory_dir = base / "figures" / "tf" / "paper_mkp_trajectory"
        paths = [
            base / "figures" / "time_slices" / "time_slice_panels.png",
            trajectory_dir / "perturb_index_traj_primary__g-mk-union__z-40__dr-umap.png",
            trajectory_dir / "perturb_index_traj_primary__g-mk-union__z-0__dr-umap.png",
            trajectory_dir / "perturb_index_traj_primary__g-mk-union__z-m40__dr-umap.png",
            base / "figures" / "tf" / "paper_mkp_fate" / "perturbation_sweep_heatmap_prob.png",
            base / "figures" / "tf" / "paper_primary" / "primary" / "batch" / "perturb_batch_curve__task-a-itga2b-cd41__allz.png",
        ]
    elif figure_id == "5":
        if base is None:
            raise ValueError("Figure 5 requires run_dir=...")
        ode_dir = base / "figures" / "paper_ode_v3"
        paths = [
            base / "figures" / "synchronized_corridor" / "sync_multimodal_time_corridor.png",
            base / "figures" / "program_dynamics_both.png",
            base / "figures" / "nkx2_1" / "NKX2-1_RNA.png",
            base / "figures" / "nkx2_1" / "NKX2-1_ATAC.png",
            base / "figures" / "key_molecule_volcano.png",
            ode_dir / "ode_results" / "ode_trajectories_primary.png",
            ode_dir / "ode_results" / "ode_trajectories_secondary.png",
            ode_dir / "growth_primary.png",
            ode_dir / "growth_secondary.png",
        ]
    else:
        if base is None:
            raise ValueError("Figure 6 requires run_dir=...")
        trajectory_dir = base / "figures" / "tf" / "erbb4" / "trajectory"
        paths = [
            base / "figures" / "time_slices" / "author5_time_slice_panels_umap.png",
            base / "figures" / "author5_lineage" / "author5_lineage_sankey.png",
            trajectory_dir / "perturb_index_traj_primary__g-mge-erbb4__z-0__dr-umap.png",
            trajectory_dir / "perturb_index_traj_primary__g-mge-erbb4__z-9__dr-umap.png",
            trajectory_dir / "perturb_index_traj_secondary__g-mge-erbb4__z-0__dr-umap.png",
            trajectory_dir / "perturb_index_traj_secondary__g-mge-erbb4__z-9__dr-umap.png",
            base / "figures" / "tf" / "erbb4" / "fate" / "perturbation_sweep_heatmap_prob.png",
        ]
    return f"Figure {figure_id}", paths


def show_paper_figure(
    figure: int | str,
    *,
    root: str | Path | None = None,
    run_dir: str | Path | None = None,
    columns: int = 2,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Display one manuscript figure through one readable notebook call.

    The function follows the STVCR tutorial pattern: the notebook calls one
    high-level plotting function and receives one Matplotlib figure. The
    function only lays out the exact PNGs produced by the released analysis;
    it does not redraw values, refit embeddings, or add perturbation scores.
    ``run_dir`` is the output directory returned by ``run_real_downstream``
    for Figures 3--6. For Figure 2 it is the ``figure2_reproduced`` folder.
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    title, paths = _paper_figure_panels(figure, root=root, run_dir=run_dir)
    if columns < 1:
        raise ValueError("columns must be positive")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing manuscript panel(s): " + ", ".join(map(str, missing)))

    rows = (len(paths) + columns - 1) // columns
    if figsize is None:
        figsize = (6.0 * columns, 4.6 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()
    for ax, path in zip(axes_flat, paths):
        ax.imshow(imread(path))
        ax.axis("off")
    for ax in axes_flat[len(paths):]:
        ax.axis("off")
    fig.suptitle(f"{title} - exact manuscript-selected panels", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()


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
    """Find the release root from a notebook, working directory, or script."""
    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "src" / "CytoBridge").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise RuntimeError(
        "MOFI repository not found. Start Jupyter from the MOFI-release root "
        "or pass root=... to setup_notebook()."
    )


def setup_notebook(root: str | Path | None = None, *, device: str = "auto"):
    """Configure one tutorial notebook and return ``ROOT, DEVICE, runner``.

    ``device='auto'`` selects CUDA when available and otherwise CPU.  All
    runtime caches are kept under the release directory, never in a protected
    Python installation.
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

    print(
        json.dumps(
            {
                "repository": root_path.name,
                "python": sys.executable,
                "device": selected_device,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if selected_device == "cuda" and torch.cuda.is_available()
                    else None
                ),
                "CytoBridge": str(loaded_library),
            },
            ensure_ascii=False,
        )
    )
    return root_path, selected_device, run_original_script
