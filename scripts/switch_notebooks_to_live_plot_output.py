#!/usr/bin/env python3
"""Remove saved-file preview calls from already-executed tutorial notebooks."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        updated = re.sub(
            r"\n\s*show_pngs\(\[.*?\], heading=.*?\)\n?",
            "\nprint(\"Live figures above were emitted by the original plotting functions.\")\n",
            source,
            flags=re.DOTALL,
        )
        updated = updated.replace(
            "ROOT, DEVICE, show_pngs, run_original_script = setup_notebook()",
            "ROOT, DEVICE, run_original_script = setup_notebook()",
        )
        updated = re.sub(r"\n\"\)\s*$", "", updated)
        if updated != source:
            cell["source"] = updated
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path.name)
