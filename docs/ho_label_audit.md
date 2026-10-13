# HO label correction audit (2026-10-13)

## Manuscript convention and scope

The supplied manuscript `MOFI(6).docx`, HO Results section (page 9), describes
neuroectoderm (`nect`) followed by its derivative neuroepithelium (`nepi`).
This is the reference convention for interpretation. Temporal order alone is
not a rule assigning every cell at an early/late time to one of these types.

The existing release-copy mapping (original early `nepi` -> `nect`, original
late `nect` -> `nepi`) is retained. The original per-cell annotations and their
relationship to developmental time cannot be independently verified without
the HO data. Do not apply the swap to arbitrary correctly labelled datasets.
No original data, processed biological files, saved figures, or model weights
were rewritten during this audit.

## Root cause and repair

The old `changed == 0` guard was unreachable whenever either swap label was
present. Repeated calls inverted the labels while retaining `status=applied`.
The helper now checks the applied audit **before** replacement, verifies its
label column, and preserves the first audit on subsequent calls. Existing
release audits are supported. Copying/subsetting and H5AD round trips preserve
this behavior when `uns` is retained.

An old applied audit does not prove that its current labels are correct: the
old implementation could have swapped them an even number of times. The new
code prevents further inversions; it does not silently repair historical data.
An object whose audit was discarded cannot be distinguished from an original
swapped object from these labels alone.

## Call-path review

| Entry/path | Objects and old behavior | New behavior / remaining check |
| --- | --- | --- |
| `apply_ho_release_label_fix` | Loads, swaps, and saves RNA, ATAC, dynamics; a second invocation inverted all three | Idempotent; only intended for verified swapped release copies. Not run on biological data in this audit |
| `run_real_training_smoke('ho')` | Loads RNA and ATAC, then swaps again even if release-fixed | Applied audits prevent inversion; checks shared-cell labels and times before sampling |
| `CytoBridge.tl.fit` / HO training config | Fits latent arrays grouped by time; no additional stage-label swap found | No dynamics objective changed; real training was not rerun |
| `run_real_downstream('ho')` | Reads release files through `infer_dense_time_dual`; no swap in the wrapper | Read-only validation of all three release objects before inference |
| `infer_dense_time_dual` / `run_dense_time_workflow` | Loads processed and dynamics objects; classifier training uses processed labels, initial sampling uses main processed metadata; old cache signature did not include actual stage labels | When correction metadata is present, validates all three objects and includes label/time/ID fingerprints in the inference cache signature. Old signatures cannot match the new signature |
| `run_paper_ode_v3('ho')` | Loads dynamics and swaps again for live ATAC backgrounds and growth plotting; RNA trajectory PNG is copied from an archived panel | Validates release objects, then the load-time correction is a no-op for an applied audit. The unreachable second HO branch after the HO return was removed |
| `notebooks/03_human_organoid_and_figure5.ipynb` | Calls downstream analysis, corridor renderer and `run_paper_ode_v3`; no direct stage swap in notebook source | Inherits downstream and Figure 5 checks |
| `manuscript.render_multimodal_corridor` / generic corridor renderer | Reads dynamics and two processed objects; no label correction found | No additional swap; standalone users should run the read-only validator first |

The repository was searched across Python, notebook source cells, configurations,
and documentation for both helper names, the audit key, and `nepi`/`nect`.
No other direct calls or explicit two-way swaps were found. The release helper
has no internal caller in the current tracked repository; previous external
release preparation executions are not recoverable from source alone.

## Read-only consistency review

With the verified release copies available, run:

```python
from CytoBridge.reproducibility import validate_ho_release_labels
report = validate_ho_release_labels()
```

The report gives labels by time for RNA, ATAC, and dynamics. Shared unique cell
IDs are compared after alignment, rather than by row order. Missing applied
audits, conflicting labels, conflicting times, and duplicate IDs raise errors.
Disjoint IDs are explicitly reported as `no_shared_ids_unverified`; matching
population counts are not evidence of per-cell agreement. Distinct RNA/ATAC
metacell IDs require the original pairing/correspondence table for verification.

## Tests and limits

Run `python -m pytest -q` after installing `anndata`, `pytest`, and `pyyaml`.
The label helpers do not require importing torch; the training helper imports
it when invoked. Sixteen synthetic tests pass, including first correction,
repetition, historical applied audits, copies/subsets, H5AD save/reload, two
release-fix passes over all three files, reordered matched cells, mixed
provenance, label/time conflicts, and cache fingerprints across save/reload.
No full GPU training, dense-time inference, or Figure 5 rendering was performed.

Biological HO assets are absent from Git (`paper_data/README.md` documents the
external data/model dependencies). The following remain unverified:

1. Per-cell original labels against the manuscript convention and experimental
   stage metadata; correction history of each RNA/ATAC/dynamics release copy.
2. Original metacell correspondence and label agreement across both modalities
   and dynamics, including objects with disjoint IDs.
3. Figure 5A annotations, classifier-derived cell proportions, and stage-named
   downstream outputs generated with the old files/caches.
4. Figure 5E archived RNA panel legends versus live ATAC panel labels; applying
   this code cannot repair text already embedded in an archived PNG/PDF.
5. Alignment of projected Figure 5 arrays to dynamics cell IDs. Existing code
   checks cell count, which does not prove matching row order.
6. Figure 5B/C/D/F interpretations after checking original labels and regenerating
   affected analyses with fresh caches where appropriate. Programs, features,
   EB/organoid contrasts and model arrays are not changed by this patch; that
   alone does not certify the previously submitted figures.

This audit establishes a reproducible code defect and its repair. It does not
establish that any submitted/published biological result has incorrect labels.
