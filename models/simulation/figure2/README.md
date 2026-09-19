# Figure 2 simulation model

This directory contains the public fitted model used as a reproducible reference for the MOFI simulation benchmark.

## Files

- `adata.h5ad` — fitted simulation AnnData object.
- `Pretrain1/last_model.pth` — checkpoint after the first training stage.
- `Pretrain2/last_model.pth` — checkpoint after the second training stage.
- `evaluation.json` — W1, TMV, and training-time summary from the recorded run.
- `reproduction_manifest.json` — portable inputs and training metadata.

The source simulation data are tracked under [`../../../datasets/simulation/`](../../../datasets/simulation/), and the portable training configuration is [`../../../configs/training/simulation_figure2.yaml`](../../../configs/training/simulation_figure2.yaml).

The tutorial [`../../../notebooks/00_simulation_training_and_figure2.ipynb`](../../../notebooks/00_simulation_training_and_figure2.ipynb) demonstrates training from zero. These files provide a convenient fitted reference for downstream exploration.
