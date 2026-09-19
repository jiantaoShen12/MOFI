import json
from pathlib import Path

for path in sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    errors = [
        (i, output.get("ename"), output.get("evalue"))
        for i, cell in enumerate(notebook["cells"])
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    counts = [
        (i, len(cell.get("outputs", [])), cell.get("execution_count"))
        for i, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    ]
    print(path.name, counts, "errors", errors)
