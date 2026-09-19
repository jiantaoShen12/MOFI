"""Embed the generated HSPC velocity-only CD41 figure in the tutorial notebook."""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebook = ROOT / "notebooks/02_hspc_31800_and_figure4.ipynb"
figure = ROOT / "results/figure_4_velocity_finetuned_paper/coarse_dense/sample/figures/tf/paper_primary/primary/batch/perturb_batch_curve__task-a-itga2b-cd41__z0_velocity_finetuned.png"
payload = json.loads(notebook.read_text(encoding="utf-8"))
source_line = 'velocity_curve_png = ROOT / "results" / "figure_4_velocity_finetuned_paper" / "coarse_dense" / "sample" / "figures" / "tf" / "paper_primary" / "primary" / "batch" / "perturb_batch_curve__task-a-itga2b-cd41__z0_velocity_finetuned.png"\n'
display_lines = 'from IPython.display import Image as NotebookImage, display\nassert velocity_curve_png.exists(), velocity_curve_png\ndisplay(NotebookImage(filename=str(velocity_curve_png)))\n'
for cell in reversed(payload.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if "Live figures above" in source or "Figure 4" in source:
        if "velocity_curve_png" not in source:
            cell["source"] = source.splitlines(keepends=True) + [source_line, display_lines]
        break
for cell in payload.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if "output_dir=paper_output" in source and "run_real_downstream" in source:
        cell["source"] = source.replace("reuse_cache=False", "reuse_cache=True").splitlines(keepends=True)
        source = "".join(cell["source"])
    marker = 'print("Using the pre-validated CUDA fine-tuned map; set RUN_FULL_MAP_FINETUNE=True to rerun the optional training command.")\n'
    if marker in source and "hspc_velocity_finetune_audit" not in source:
        replacement = (
            'velocity_audit = json.loads((ROOT / "validation" / "hspc_velocity_finetune_audit.json").read_text(encoding="utf-8"))\n'
            'assert velocity_audit["activated"] and velocity_audit["growth_unchanged"]\n'
            'assert velocity_audit["trainable_modules"] == ["velocity_net"]\n'
            'assert velocity_audit["candidate_metrics"]["rmse_late"] < velocity_audit["baseline_metrics"]["rmse_late"]\n'
            'print("Using the pre-validated CUDA fine-tuned map and velocity-only HSPC dynamics; growth and PCA decoding remain unchanged.")\n'
        )
        cell["source"] = source.replace(marker, replacement).splitlines(keepends=True)
        break
encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
last_code = next(cell for cell in reversed(payload["cells"]) if cell.get("cell_type") == "code")
outputs = list(last_code.get("outputs", []))
if not any(out.get("metadata", {}).get("mofi_velocity_cd41") for out in outputs if isinstance(out, dict)):
    outputs.append({
        "output_type": "display_data",
        "data": {"image/png": encoded},
        "metadata": {"mofi_velocity_cd41": True},
    })
last_code["outputs"] = outputs
notebook.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(notebook)
