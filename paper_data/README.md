# Data used by the manuscript notebooks

The following identifiers link to the original public data. Cell counts and feature counts refer to the inputs used by the MOFI notebooks, not each original study. Time indices are the values in the shared input.

| Data | Cells and modalities | Time points | Features before PCA | Original source |
| --- | --- | --- | --- | --- |
| Kidney organoid | 55,177 nuclei; RNA and ATAC gene activity. Broad labels: nephron progenitor, developing tubule, mature tubule and podocyte. | Days 7, 12, 16, 19 and 26 (indices 0–4). | RNA 2,041; ATAC 2,068. | [GSE213152](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213152); Yoshimura et al., *A single-cell multiomic analysis of kidney organoid differentiation*, PNAS (2023), [PMID 37155865](https://pubmed.ncbi.nlm.nih.gov/37155865/). |
| Hematopoiesis | 28,087 cells from donor 31800; RNA and protein. Labels: HSC, EryP, NeuP, MasP, MkP and MoP. | Days 2, 4, 7 and 10 (indices 0–3). | RNA 2,000; protein 103. | [GSE305370](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305370); [2022 Open Problems multimodal challenge](https://openproblems.bio/events/2022-08_neurips/). The GEO record links [PMID 41129612](https://pubmed.ncbi.nlm.nih.gov/41129612/). |
| Brain organoid | 9,894 observations; RNA and ATAC gene activity. Labels: EB, nect, nepi and organoid. | Days 4, 7, 9, 11, 12, 18, 21, 26, 31 and 61 (indices 0–9). | RNA 2,000; ATAC 2,000. | [E-MTAB-12001](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-12001), [E-MTAB-11998](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11998); *Inferring and perturbing cell fate regulomes in human brain organoids*, Nature (2022), [PMID 36198796](https://pubmed.ncbi.nlm.nih.gov/36198796/). |
| Human cortex | 6,163 nuclei; RNA and ATAC gene activity. Labels: RG, IPC, IN-fetal, IN-CGE and IN-MGE. | Post-fertilization weeks 18, 19, 23 and 24, and infancy (indices 0–4). | RNA 2,000; ATAC 2,000. | [GSE204684](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE204684); *Multi-omic profiling of the developing human cerebral cortex at the single-cell level*, Science Advances (2023), [PMID 37824614](https://pubmed.ncbi.nlm.nih.gov/37824614/). |

The historical brain-organoid preprocessing removed day 16 (106 records in the 10,000-row pre-filter input); day 12 was retained. The brain-organoid time-course RNA and ATAC data come from separate experiments; matching row IDs in these prepared files should not be interpreted as proof of same-cell multiome measurements. Dataset terms remain those of the original sources; the software license does not automatically apply to those data.

The HSPC CITE-seq time course was used in the 2022 Open Problems competition and later analyzed by [DeMeo et al.](https://doi.org/10.1126/science.adi8577), whose data-availability statement identifies GSE305370. Our prepared input uses donor 31800 at days 2, 4, 7 and 10 (not day 3); it is a filtered subset, not the complete four-donor, five-time-point study.

For kidney organoids, `rna.h5ad` and `atac.h5ad` use the historical manuscript annotations in `obs["cell_type"]`, including `dev`. The corresponding fitted five-class classifier was originally fitted using 60,845 cells; the release contains the 55,177-cell subset.

For the kidney ATAC trajectory panel, the historical full-data model uses 60,845 RNA observations and an ATAC background of 60,845 observations. The release includes only the ten selected 50-dimensional starting cells for this panel, together with the time grid and the background time/cell labels; its fitted UMAP model is provided separately.

The shared feature matrices are the processed values supplied to PCA in the historical analysis, not raw counts and not values recovered exactly by reversing 50 PCs. The 50-dimensional representation, feature names and PCA transformation arrays are retained for the notebook analyses. Dataset-specific normalization settings have not yet been independently verified.

The analyses include fitted cell-type classifier coefficients from the historical analyses. For brain organoids, the nepi and nect classifier coefficient rows were exchanged to match the corrected nepi/nect labels in the shared inputs.

The kidney ATAC UMAP is supplied for projection and figure reproduction. Its `graph_`, `_knn_indices` and `_knn_dists` attributes have been removed; the projection index still retains 50-dimensional reference data, search structures and the fitted 2D embedding.
