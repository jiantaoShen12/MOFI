#!/usr/bin/env python3
"""Check the size-independent SHA-256 manifest of the manuscript inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = ROOT / "paper_data" / "SHA256SUMS"
    failures = []
    entries = manifest.read_text(encoding="ascii").splitlines()
    for line in entries:
        expected, name = line.split("  ", 1)
        path = ROOT / name
        if not path.is_file():
            failures.append(f"missing: {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            failures.append(f"checksum mismatch: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(entries)} manuscript input files.")


if __name__ == "__main__":
    main()
