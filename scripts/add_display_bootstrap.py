"""Mechanical migration helper for the five tutorial notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    bootstrap = code_cells[0]
    source = "".join(bootstrap.get("source", []))
    if "from IPython.display import display" not in source:
        source = source.replace(
            "import json, time\n",
            "import json, time\nfrom IPython.display import display\n",
        )
        bootstrap["source"] = source
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path.name)
