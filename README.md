<div align=center>
<h1>MOFI: Multi-Omics Fate Inference</h1>
</div>

<div align=center>

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

**MOFI** is a computational framework that reconstructs coordinated multi-omics developmental trajectories from discrete paired snapshots. It integrates synchronized unbalanced optimal transport with a learned cross-omics mapper to jointly infer velocity fields, growth rates, and fate trajectories across two omics.

## 🌟 Highlights

- **Synchronized Multi-Omics Dynamics**: Reconstructs continuous developmental trajectories that remain coherent across transcriptomic and epigenomic/proteomic spaces
- **Cross-Omics Regulatory Decoding**: Quantifies bidirectional gene-protein regulatory interactions at single-cell resolution via transport map Jacobians
- **In Silico Perturbation**: Simulates gene perturbations and predicts fate shifts across omics
- **Flexible Data Requirements**: Works with as few as 2 paired time points; supports asymmetric temporal sampling across omics

## 🛠 Installation (The code has not been made public yet.)

```bash
# Clone the repository
git clone https://github.com/MOFI/MOFI.git
cd MOFI

# Install in development mode
pip install -e .
```

## 📖 Quick Start

### Single-Modal Training (Simulation)

```python
import sys; sys.path.insert(0, 'src')
import scanpy as sc
import CytoBridge.pp as cb_pp
import CytoBridge.tl as cb_tl
from CytoBridge.utils.utils import set_seed

set_seed(42)
adata = sc.read_h5ad('datasets/simulation/simulation4_2d.h5ad')

# Preprocess
adata = cb_pp.preprocess(adata, time_key='samples',
                         time_mapping={0.0: 0.0, 0.75: 1.0, 1.5: 2.0, 2.25: 3.0, 3.0: 4.0},
                         dim_reduction='none', normalization=False, log1p=False, select_hvg=False)

# Train dynamics model
adata = cb_tl.fit(adata, config='examples/configs/simu/unbalanced_ot_simu.yaml',
                  batch_size=400, device='cuda')
```

### Dual-Modal Training (RNA + ATAC/Protein)

```python
import scanpy as sc
import CytoBridge.tl as cb_tl

# Load preprocessed data with X_latent
adata_rna = sc.read_h5ad('datasets/GSE213152/GSE213152_rna_pca50.h5ad')
adata_atac = sc.read_h5ad('datasets/GSE213152/GSE213152_atac_gene_pca50.h5ad')

# Train dual-modal dynamics (requires pretrained Map model)
adata = cb_tl.fit(adata_rna,
                  config='examples/configs/GSE213152/unbalanced_ot_gse213152_12_cycle.yaml',
                  adata_sec=adata_atac, device='cuda')
```

### Full Pipeline (Map Training → Dynamics → Evaluation → Visualization)

```bash
python examples/scripts/run_pipeline.py \
    --config examples/configs/GSE213152/pipeline_GSE213152_12.yaml
```

## 📚 Tutorials

| Notebook | Description | Data |
|----------|-------------|------|
| [00_simulation.ipynb](tutorial/00_simulation.ipynb) | Single-modal simulation benchmark | simulation4_2d |
| [01_geo_kidney_GSE213152.ipynb](tutorial/01_geo_kidney_GSE213152.ipynb) | Kidney organoid differentiation (RNA+ATAC) | GSE213152 |
| [02_geo_kidney_GSE233134.ipynb](tutorial/02_geo_kidney_GSE233134.ipynb) | Kidney organoid differentiation (RNA+ATAC) | GSE233134 |
| [03_hspcs_31800.ipynb](tutorial/03_hspcs_31800.ipynb) | HSPC fate decisions (RNA+Protein) + perturbation | 31800 |

## 📁 Repository Structure

```
MOFI/
├── src/
│   ├── CytoBridge/          # Core library (engine)
│   │   ├── pp/              # Preprocessing
│   │   ├── tl/              # Training & analysis (models, losses, ODE)
│   │   ├── pl/              # Plotting & visualization
│   │   ├── Map/             # Cross-omics mapper (AE, Moscot)
│   │   ├── utils/           # Utilities
│   │   └── configs/         # Training config YAMLs
│   └── mofi/                # Thin wrapper (import mofi)
├── datasets/                # Example datasets
│   ├── simulation/
│   ├── GSE213152/
│   ├── GSE233134/
│   └── 31800/
├── tutorial/                # Jupyter notebook tutorials
├── examples/
│   ├── configs/             # Pipeline & training configs
│   └── scripts/             # Pipeline runner scripts
├── setup.py
├── requirements.txt
└── README.md
```

## 🔬 Biological Systems Demonstrated

1. **In Silico Benchmark**: Controlled simulation with ground-truth dynamics
2. **Kidney Organoid Differentiation** (GSE213152, GSE233134): RNA + ATAC, multi-lineage fate commitment
3. **Hematopoietic Progenitor Cells** (31800): RNA + Protein, perturbation-responsive fate prediction
4. **Human Brain Organoid Maturation**: RNA + ATAC, gradual compositional redistribution
5. **Developing Human Cerebral Cortex**: RNA + ATAC, continuous corticogenesis with lineage allocation

## 📖 Citation



## 📧 Contact

2601111702@stu.pku.edu.cn

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
