# Contributing to MOFI

Thank you for helping improve MOFI. Contributions that improve portability, documentation, dataset adapters, plotting utilities, or reproducibility are welcome.

## Before opening a pull request

- Run the relevant notebook or a focused automated test.
- Keep all paths repository-relative or user-configurable.
- Do not commit raw datasets, trained checkpoints, validation dumps, notebook checkpoints, or local environment folders.
- Keep one manuscript panel per display cell when changing the release notebooks.
- Preserve the source-data provenance of a figure; do not manually shift, reverse, smooth, or reshape a curve to make it look closer to a target image.
- Add or update a short README/doc section when a new user-facing workflow is introduced.

## Pull request checklist

- [ ] The change is scoped and documented.
- [ ] A clean environment can import the package.
- [ ] New examples contain no machine-specific absolute paths.
- [ ] Large data and model files are referenced through a manifest or release asset.
- [ ] Figures and tables are generated from named inputs and reproducible calls.
- [ ] The web page still loads its local assets when `index.html` is served from the repository root.

Please include the command used to reproduce the change and describe any data release asset that is required.

