# MOFI web demo

This folder contains the complete static browser demo:

- `index.html` — landing page and interactive panels;
- `style.css` — page styles;
- `script.js` — browser interaction and visualizations;
- `assets/` — website figures, icons, and logo;
- `demo_data/` — compact sampled and illustrative JSON assets used only by the browser demo.

The files under `demo_data/` are not the complete manuscript datasets. They are
small web assets for exploration and page demonstrations. The full paper-data
release is described in the repository-level `README.md` and `data_manifest.yaml`.

## Run locally

From the repository root:

```bash
python -m http.server 8000 --directory web
```

Then open <http://127.0.0.1:8000/>.
