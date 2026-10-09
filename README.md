# MOFI: Multi-Omics Fate Inference

MOFI reconstructs continuous cell-state dynamics from time-series single-cell multi-omics data. It combines cross-modality mapping with models of cell-state change and population growth to study developmental trajectories, cell fates and molecular perturbations.

![MOFI framework](web/assets/Figure1.png)

## Getting started

From the repository root:

```bash
git clone https://github.com/jiantaoShen12/MOFI.git
cd MOFI
conda create -n mofi python -y
conda activate mofi
python -m pip install -r requirements.txt
python -m pip install -e .
jupyter lab
```

The simulation notebook includes model fitting and evaluation. The biological examples use fitted models and processed data placed under `paper_data/`; see the [data notes](paper_data/README.md). These biological data and models are not included in the repository.

## Examples

The notebooks cover a simulation benchmark and four biological datasets.

| Notebook | Analysis |
| --- | --- |
| [Simulation](notebooks/00_simulation_training_and_figure2.ipynb) | Trajectories, potential landscape and growth rates |
| [Kidney organoids](notebooks/01_gse213152_and_figure3.ipynb) | RNA–ATAC dynamics, cell fates and marker expression |
| [HSPCs](notebooks/02_hspc_31800_and_figure4.ipynb) | RNA–protein dynamics and MkP perturbation |
| [Human organoids](notebooks/03_human_organoid_and_figure5.ipynb) | Developmental programs and NKX2-1 dynamics |
| [Cortical development](notebooks/04_cortical_development_and_figure6.ipynb) | Lineage transitions and ERBB4 perturbation |

## Code

Paths below are relative to the repository root.

| Directory | Contents |
| --- | --- |
| `src/` | MOFI implementation in `CytoBridge/`, with the `mofi/` import alias. `Map/` handles cross-modality mapping, `tl/` model fitting and dynamics, `pp/` preprocessing, and `pl/` plotting and perturbation analysis. |
| `configs/` | Training, analysis and perturbation settings, with selected cell indices and gene lists in `analysis_inputs/`. |
| `notebooks/` | Simulation training and analyses for Figures 2–6; `mofi_notebook.py` provides shared notebook helpers. |
| `scripts/` | Notebook execution and simulation data generation, training, evaluation and plotting. |
| `datasets/simulation/` | Simulation inputs and reference gene-expression values. |
| `models/simulation/` | Fitted simulation models and evaluation results. |
| `downstream/plotting/` | Additional plotting routines for the biological analyses. |
| `paper_data/` | Notes on the external biological data and fitted models needed by the notebooks. |
| `web/` | Interactive website, paper figures in `assets/`, and sampled trajectories in `demo_data/`. |

## Interactive examples

The [web demo](https://jiantaoshen12.github.io/MOFI/) shows simulated and kidney organoid trajectories. To view it locally, run this command from the repository root:

```bash
python -m http.server 8000 --directory web
```

Open <http://localhost:8000/> in your browser. The demo uses sampled trajectories and illustrative animations; it does not run model inference.

## Citation

The paper citation will be added after publication.

## License

[MIT](LICENSE).
