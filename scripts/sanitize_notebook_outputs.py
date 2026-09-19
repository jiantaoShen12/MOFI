#!/usr/bin/env python3
"""Remove workstation-specific paths from executed notebook outputs."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re


WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:[\\/](?:[^\\/\r\n]+[\\/])*[^\\/\r\n]*")
POSIX_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:home|Users|mnt|tmp)/[^\s\"']+")
PNG_SIGNATURE = bytes.fromhex("89504e470d0a1a0a")


def scrub_text(value: str) -> str:
    value = WINDOWS_PATH.sub("<local-path>", value)
    return POSIX_PATH.sub("<local-path>", value)


def normalize_png(value: str) -> str:
    """Collapse accidental base64-of-base64 PNG outputs."""
    candidate = value.encode()
    for _ in range(3):
        try:
            decoded = base64.b64decode(candidate)
        except Exception:
            return value
        if decoded[:8] == PNG_SIGNATURE:
            return base64.b64encode(decoded).decode("ascii")
        candidate = decoded
    return value


def scrub_tree(value, *, binary_key: str | None = None):
    if isinstance(value, str):
        if binary_key == "image/png":
            return normalize_png(value)
        return value if binary_key and binary_key.startswith("image/") else scrub_text(value)
    if isinstance(value, list):
        return [scrub_tree(item, binary_key=binary_key) for item in value]
    if isinstance(value, dict):
        return {
            key: scrub_tree(item, binary_key=key if key in {"image/png", "image/jpeg", "image/svg+xml"} else binary_key)
            for key, item in value.items()
        }
    return value


def sanitize(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    notebook["metadata"] = scrub_tree(notebook.get("metadata", {}))
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            scrubbed = scrub_tree(output)
            output.clear()
            output.update(scrubbed)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.notebooks:
        sanitize(path)
        print(path)


if __name__ == "__main__":
    main()
