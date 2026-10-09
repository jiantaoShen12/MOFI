# Data and models

The biological examples use processed RNA, ATAC or protein data and fitted dynamics and mapping models. These files are stored locally under `paper_data/` and are not included in Git.

The dataset names used in the configurations are `gse213152` (kidney organoid), `hspc_31800` (HSPC), `ho` (human organoid) and `hc_author5` (cortical development).

```text
paper_data/
├── processed/   # RNA, ATAC and protein AnnData objects
├── dynamics/    # Fitted dynamics objects and configurations
├── maps/        # Cross-modality maps
├── umap/        # Fitted UMAP projections
├── projected/   # Saved coordinates for Figure 5
└── archived/    # Source tables and original figure panels
```

Input paths are defined in `configs/analysis/` and `src/CytoBridge/reproducibility.py`. Most datasets use `maps/<dataset>/best_model.pt`; the kidney example uses separate `gse213152_training` and `gse213152_analysis` maps. Figure 5 also requires the saved coordinates and original RNA trajectory panel from the original analysis.

Simulation inputs are included in `datasets/simulation/`, with fitted models in `models/simulation/figure2/`.
