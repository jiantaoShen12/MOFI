# Real paper data and models

This directory is intentionally kept free of real biological datasets and their fitted model files in normal Git history.

Place downloaded assets at the exact repository-relative paths listed in [`../data_manifest.yaml`](../data_manifest.yaml). The four dataset identifiers are:

- `gse213152`
- `hspc_31800`
- `ho`
- `hc_author5`

The expected layout is:

```text
paper_data/
├── processed/    # Processed RNA, ATAC, or protein AnnData inputs
├── dynamics/     # Fitted MOFI dynamics AnnData objects and configurations
├── maps/         # Cross-modality mapping checkpoints
├── umap/         # Paper projection models and parameters
├── projected/    # Small precomputed paper projections where specified
└── archived/     # Released manuscript source tables or panels where specified
```

Use the downloader when release assets are available:

```bash
python scripts/download_data.py --all
```

To check manually placed files without downloading:

```bash
python scripts/download_data.py --all --verify-only
```

Do not commit the downloaded contents of this directory. Simulation inputs and the fitted Figure 2 simulation model are public under [`../datasets/simulation/`](../datasets/simulation/) and [`../models/simulation/figure2/`](../models/simulation/figure2/), respectively.
