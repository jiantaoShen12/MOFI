#!/usr/bin/env python3
"""Compact embedded PNG previews while preserving full-resolution result files."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import nbformat
from PIL import Image


def compact_png(encoded: str, max_width: int, max_height: int) -> tuple[str, int, int]:
    raw = base64.b64decode(encoded)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        original_size = len(raw)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        if image.mode not in {"RGB", "RGBA", "L", "P"}:
            image = image.convert("RGBA")
        stream = io.BytesIO()
        image.save(stream, format="PNG", optimize=True, compress_level=9)
    compacted = stream.getvalue()
    return base64.b64encode(compacted).decode("ascii"), original_size, len(compacted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--max-width", type=int, default=1200)
    parser.add_argument("--max-height", type=int, default=900)
    args = parser.parse_args()

    for path in args.notebooks:
        notebook = nbformat.read(path, as_version=4)
        before = after = count = 0
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                encoded = data.get("image/png")
                if not encoded:
                    continue
                compacted, old_bytes, new_bytes = compact_png(
                    encoded, args.max_width, args.max_height
                )
                data["image/png"] = compacted
                before += old_bytes
                after += new_bytes
                count += 1
        nbformat.write(notebook, path)
        print(
            f"{path.name}: images={count}, embedded_png={before / 1e6:.2f}MB"
            f" -> {after / 1e6:.2f}MB"
        )


if __name__ == "__main__":
    main()
