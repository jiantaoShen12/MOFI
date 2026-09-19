#!/usr/bin/env python3
"""Validate the public MOFI tutorial release without creating audit artifacts."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMAGES = {
    "00_simulation_training_and_figure2.ipynb": 4,
    "01_gse213152_and_figure3.ipynb": 8,
    "02_hspc_31800_and_figure4.ipynb": 6,
    "03_human_organoid_and_figure5.ipynb": 9,
    "04_cortical_development_and_figure6.ipynb": 7,
}
ABSOLUTE_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]|/(?:home|Users|mnt|tmp)/")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def validate_png(output: dict, notebook_name: str) -> None:
    encoded = output.get("data", {}).get("image/png")
    assert isinstance(encoded, str), f"{notebook_name}: image output is not base64 text"
    decoded = base64.b64decode(encoded, validate=True)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n", f"{notebook_name}: invalid PNG payload"


def validate_notebooks() -> dict:
    notebook_dir = ROOT / "notebooks"
    paths = sorted(notebook_dir.glob("*.ipynb"))
    assert [path.name for path in paths] == list(EXPECTED_IMAGES), [path.name for path in paths]

    report: dict[str, dict] = {}
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert int(notebook.get("nbformat", 0)) == 4, path.name
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
        assert not [output for output in outputs if output.get("output_type") == "error"], path.name

        for index, cell in enumerate(code_cells, start=1):
            ast.parse(source_text(cell), filename=f"{path.name}:code-cell-{index}")

        image_cells = [
            cell for cell in code_cells
            if any("image/png" in output.get("data", {}) for output in cell.get("outputs", []))
        ]
        images = sum(
            "image/png" in output.get("data", {})
            for cell in image_cells
            for output in cell.get("outputs", [])
        )
        assert images == EXPECTED_IMAGES[path.name], (path.name, images)
        assert len(image_cells) == images, f"{path.name}: each result must have its own display cell"
        for cell in image_cells:
            png_outputs = [
                output for output in cell.get("outputs", [])
                if "image/png" in output.get("data", {})
            ]
            assert len(png_outputs) == 1, f"{path.name}: multiple figures in one display cell"
            validate_png(png_outputs[0], path.name)

        readable_copy = json.loads(json.dumps(notebook, ensure_ascii=False))
        for cell in readable_copy.get("cells", []):
            for output in cell.get("outputs", []):
                data = output.get("data")
                if isinstance(data, dict):
                    data.pop("image/png", None)
                    data.pop("image/jpeg", None)
        readable_text = json.dumps(readable_copy, ensure_ascii=False)
        assert not ABSOLUTE_PATH.search(readable_text), f"{path.name}: absolute machine path leaked"

        tutorial_text = "\n".join(source_text(cell) for cell in notebook["cells"])
        lowered = tutorial_text.lower()
        for disallowed in ("smoke test", "panel contract", "release audit"):
            assert disallowed not in lowered, f"{path.name}: tutorial contains {disallowed!r}"
        assert "setup_notebook" in tutorial_text, path.name
        assert "show_paper_panel" in tutorial_text, path.name
        assert "show_pngs" not in tutorial_text, path.name

        if path.name.startswith("00_"):
            assert "train_figure2" in tutorial_text and "evaluate_figure2" in tutorial_text
        elif path.name.startswith("01_"):
            assert "render_synchronized_corridor" in tutorial_text
            assert "plot_marker_bridging" in tutorial_text and "run_paper_ode_v3" in tutorial_text
        elif path.name.startswith("02_"):
            assert "run_perturbation_pipeline_for_run" in tutorial_text
            assert "build_transformation_summary" in tutorial_text
            assert "ITGA2B" in tutorial_text and "CD41" in tutorial_text
        elif path.name.startswith("03_"):
            assert "key_molecule_volcano" in tutorial_text
            assert "plot_marker_bridging" in tutorial_text and "run_paper_ode_v3" in tutorial_text
        elif path.name.startswith("04_"):
            assert "author5_lineage_sankey" in tutorial_text and "erbb4" in lowered

        report[path.name] = {
            "bytes": path.stat().st_size,
            "cells": len(notebook["cells"]),
            "code_cells": len(code_cells),
            "image_outputs": images,
        }
    return report


def validate_web_links() -> dict:
    page = ROOT / "web" / "index.html"
    html = page.read_text(encoding="utf-8")
    local_links = re.findall(r'(?:src|href)=["\']([^"\']+)["\']', html)
    checked = 0
    for link in local_links:
        if link.startswith(("#", "http://", "https://", "mailto:", "javascript:")):
            continue
        target = (page.parent / link.split("?", 1)[0].split("#", 1)[0]).resolve()
        assert target.exists(), f"web/index.html: missing local target {link}"
        checked += 1
    return {"page": str(page.relative_to(ROOT)), "local_links_checked": checked}


def validate_manifest(check_hashes: bool) -> dict:
    payload = yaml.safe_load((ROOT / "data_manifest.yaml").read_text(encoding="utf-8"))
    checked = 0
    total_bytes = 0
    for dataset in payload["datasets"].values():
        for item in dataset["files"].values():
            path = ROOT / item["path"]
            assert path.exists(), path
            assert path.stat().st_size == int(item["bytes"]), path
            if check_hashes:
                assert digest(path) == item["sha256"], path
            checked += 1
            total_bytes += path.stat().st_size
    return {"files": checked, "bytes": total_bytes, "hashes_checked": check_hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--skip-paper-data", action="store_true")
    args = parser.parse_args()

    report = {
        "notebooks": validate_notebooks(),
        "web": validate_web_links(),
    }
    if not args.skip_paper_data:
        report["paper_data"] = validate_manifest(not args.skip_hashes)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
