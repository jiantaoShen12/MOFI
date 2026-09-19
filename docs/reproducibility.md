# Reproducibility and extension guide

This guide explains what is reproduced by the release notebooks, which files are downloaded separately, and where to extend the workflow for a new dataset.

## Reproduction contract

The release separates three layers:

1. **Library layer** — `src/CytoBridge/` contains preprocessing, mapping, neural ODE, growth, perturbation, and plotting code.
2. **Analysis layer** — `CytoBridge.reproducibility` and `CytoBridge.manuscript` resolve dataset-specific inputs and connect the released downstream renderers in `downstream/plotting/`.
3. **Presentation layer** — `notebooks/` displays one exact manuscript panel per code cell and keeps supporting tables available for inspection.

The real-data notebooks begin with the released paper-trained objects and use public `CytoBridge` calls for each downstream task. Training from raw paired AnnData is shown as a reusable pattern, while the manuscript panels remain tied to the fitted objects used for the paper.

## Run order

From the repository root:

```bash
python scripts/download_data.py --all
jupyter lab
```

Open the notebooks in this order:

1. `00_simulation_training_and_figure2.ipynb`
2. `01_gse213152_and_figure3.ipynb`
3. `02_hspc_31800_and_figure4.ipynb`
4. `03_human_organoid_and_figure5.ipynb`
5. `04_cortical_development_and_figure6.ipynb`

Each notebook explains its required inputs before analysis. If a paper-data asset is missing, download it with `scripts/download_data.py`; do not substitute a guessed dataset or add an unreported perturbation condition.

## Notebook/library boundary

The notebooks are readable callers, not a second implementation of MOFI. Training, evaluation, corridor rendering, time-slice rendering, marker bridging, and transformation summaries are called through `CytoBridge`. The command-line scripts remain useful for shell users, but notebook cells do not invoke them through `sys.argv` or copy their analysis functions.

```python
from pathlib import Path
from CytoBridge.manuscript import render_synchronized_corridor

panel_dir = render_synchronized_corridor(
    "gse213152",
    output_dir=Path("results") / "gse213152" / "figures" / "corridor",
    device="cuda",
)
```

The returned directory contains the image, PDF/SVG, HTML preview, and manifest. Use `show_paper_panel` in a separate cell when the goal is to display exactly one saved image.

## Figure contracts

- **Figure 2** uses the public simulation generator and reports the saved landscape, trajectories, growth, and benchmark outputs.
- **Figure 3** uses the paper-selected GSE213152 UMAP/projection assets and the original ODE-v3 plotting route.
- **Figure 4** keeps the manuscript MkP conditions and uses the ITGA2B–CD41 response source for the transformation/response panel. Transformation outputs are also available as tables.
- **Figure 5** keeps the manuscript brain-organoid panels. The volcano result is backed by the released source tables rather than being redrawn with invented values.
- **Figure 6** contains only the manuscript-defined cortex time slices, lineage Sankey, ERBB4 perturbation, and fate-composition outputs. Additional perturbation states are not added merely to make a larger plot.

## One-result display pattern

Use the single-panel helper when adding a new notebook display:

```python
from pathlib import Path
from mofi_notebook import show_paper_panel

panel = Path("results") / "my_analysis" / "figures" / "panel.png"
show_paper_panel(panel)
```

Keep one result in one code cell. A table may be displayed instead of a figure when that is the more useful downstream object. Avoid loops that emit several manuscript panels at once; a reader should be able to rerun and modify each analysis independently.

## Adding a new dataset

1. Add a small, human-readable configuration under `configs/`.
2. Put large inputs and trained objects in an external release or object store, not in ordinary Git history.
3. Add each external file to `data_manifest.yaml` with a relative target path, asset name, byte count, and SHA-256 checksum.
4. Add a downloader call or dataset-specific preparation function.
5. Add a small public function to `src/CytoBridge/manuscript.py` when a new high-level operation is needed; keep the notebook as a named-argument call to that function.
6. Keep analysis outputs in named intermediate paths and expose the returned paths/tables in the notebook.
7. Add a focused notebook section for each manuscript panel or new analysis output.

Avoid machine-specific paths such as `/home/...` or `<machine-root>/Users/...`. Use repository-relative paths, `Path(__file__).resolve()`, or an explicit user-supplied data root.

## Downstream analysis pattern

The recommended extension pattern is:

```python
from pathlib import Path

from CytoBridge.pl.plot_ode_v3 import plot_ode_v3

run_dir = Path("results") / "my_dataset"
trajectory = run_dir / "tables" / "trajectory.csv"
figure = run_dir / "figures" / "trajectory.png"

# Load the table, call the library plotting function, and save one named panel.
# Then display it in its own notebook cell with show_paper_panel(figure).
```

Replace the input table, model, or downstream function as needed, but keep the provenance visible in the notebook. A reader should be able to identify which object was loaded, which function was called, and which file contains the resulting table or image.
