# Data used by the manuscript notebooks

The following identifiers link to the original public data. Cell counts and feature counts refer to the inputs used by the MOFI notebooks, not each original study. Time indices are the values in the shared input.

| Data | Cells and modalities | Time points | Features before PCA | Original source |
| --- | --- | --- | --- | --- |
| Kidney organoid | 55,177 nuclei; RNA and ATAC gene activity. Broad labels: nephron progenitor, developing tubule, mature tubule and podocyte. | Days 7, 12, 16, 19 and 26 (indices 0–4). | RNA 2,041; ATAC 2,068. | [GSE213152](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213152); Yoshimura et al., *A single-cell multiomic analysis of kidney organoid differentiation*, PNAS (2023), [PMID 37155865](https://pubmed.ncbi.nlm.nih.gov/37155865/). |
| Hematopoiesis | 28,087 cells from donor 31800; RNA and protein. Labels: HSC, EryP, NeuP, MasP, MkP and MoP. | Days 2, 4, 7 and 10 (indices 0–3). | RNA 2,000; protein 103. | [GSE305370](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305370); [2022 Open Problems multimodal challenge](https://openproblems.bio/events/2022-08_neurips/). The GEO record links [PMID 41129612](https://pubmed.ncbi.nlm.nih.gov/41129612/). |
| Brain organoid | 9,894 observations; RNA and ATAC gene activity. Labels: EB, nect, nepi and organoid. | Days 4, 7, 9, 11, 12, 18, 21, 26, 31 and 61 (indices 0–9). | RNA 2,000; ATAC 2,000. | [E-MTAB-12001](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-12001), [E-MTAB-11998](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11998); *Inferring and perturbing cell fate regulomes in human brain organoids*, Nature (2022), [PMID 36198796](https://pubmed.ncbi.nlm.nih.gov/36198796/). |
| Human cortex | 6,163 nuclei; RNA and ATAC gene activity. Labels: RG, IPC, IN-fetal, IN-CGE and IN-MGE. | Post-fertilization weeks 18, 19, 23 and 24, and infancy (indices 0–4). | RNA 2,000; ATAC 2,000. | [GSE204684](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE204684); *Multi-omic profiling of the developing human cerebral cortex at the single-cell level*, Science Advances (2023), [PMID 37824614](https://pubmed.ncbi.nlm.nih.gov/37824614/). |

The brain-organoid time-course RNA and ATAC data come from separate experiments; matching row IDs in these prepared files should not be interpreted as proof of same-cell multiome measurements. Dataset terms remain those of the original sources; the software license does not automatically apply to those data.

For the kidney ATAC trajectory panel, the historical full-data model uses 60,845 RNA observations and an ATAC background of 60,845 observations. Its 50-dimensional inputs and necessary time/cell labels are included separately from the 55,177-cell balanced model inputs.

The shared feature matrices are the processed values supplied to PCA in the historical analysis, not raw counts and not values recovered exactly by reversing 50 PCs. The 50-dimensional representation, feature names and PCA transformation arrays are retained for the notebook analyses. Dataset-specific normalization settings have not yet been independently verified.

The analyses include fitted cell-type classifier coefficients from the historical analyses. For brain organoids, the nepi and nect classifier coefficient rows were exchanged to match the corrected nepi/nect labels in the shared inputs.
