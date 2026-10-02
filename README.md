# MOFI: Multi-Omics Fate Inference

MOFI reconstructs continuous cell-state dynamics from time-series single-cell multi-omics data. It combines cross-modality mapping with models of cell-state change and population growth to study developmental trajectories, cell fates and molecular perturbations.

![MOFI framework](web/assets/Figure1.png)

The full implementation and training code will be released after publication. This repository currently contains analysis examples with saved results, simulation data and the interactive web demo.

## Examples

The notebooks cover a simulation benchmark and four biological datasets.

| Notebook | Analysis |
| --- | --- |
| [Simulation](notebooks/00_simulation_training_and_figure2.ipynb) | Trajectories, potential landscape and growth rates |
| [Kidney organoids](notebooks/01_gse213152_and_figure3.ipynb) | RNA–ATAC dynamics, cell fates and marker expression |
| [HSPCs](notebooks/02_hspc_31800_and_figure4.ipynb) | RNA–protein dynamics and MkP perturbation |
| [Human organoids](notebooks/03_human_organoid_and_figure5.ipynb) | Developmental programs and NKX2-1 dynamics |
| [Cortical development](notebooks/04_cortical_development_and_figure6.ipynb) | Lineage transitions and ERBB4 perturbation |

The notebooks can be read on GitHub. Re-running the biological analyses requires the full implementation and fitted models, which are not included in this pre-publication version.

## Data

The simulation inputs are in [`datasets/simulation/`](datasets/simulation/). Data and model links for the biological examples will be added with the full release.

## Interactive examples

The [web demo](web/) shows simulated and kidney organoid trajectories. To view it locally, run this command from the repository root:

```bash
python -m http.server 8000 --directory web
```

Open <http://localhost:8000/> in your browser. The demo uses sampled trajectories and illustrative animations; it does not run model inference.

## Citation

The paper citation will be added after publication.

## License

[MIT](LICENSE).
