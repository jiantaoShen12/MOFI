#!/usr/bin/env python3
"""Download and verify MOFI paper-data assets from the repository release."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "MOFI-data-downloader"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="Dataset IDs; omit with --all")
    parser.add_argument("--all", action="store_true", help="Download all four real datasets")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = yaml.safe_load((REPO_ROOT / "data_manifest.yaml").read_text(encoding="utf-8"))
    available = manifest["datasets"]
    selected = sorted(available) if args.all else args.datasets
    if not selected:
        parser.error("Choose one or more dataset IDs, or use --all")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        parser.error(f"Unknown dataset IDs: {unknown}; available={sorted(available)}")
    base = manifest["release_base_url"].rstrip("/")
    failures = []
    for dataset_id in selected:
        for role, record in available[dataset_id]["files"].items():
            path = REPO_ROOT / record["path"]
            if not path.exists() and not args.verify_only:
                url = f"{base}/{record['asset']}"
                print(f"Downloading {dataset_id}/{role}: {url}")
                download(url, path)
            if not path.exists():
                failures.append(f"missing: {path}")
                continue
            actual = sha256(path)
            if actual != record["sha256"]:
                failures.append(f"checksum mismatch: {path}")
            else:
                print(f"OK {dataset_id}/{role}: {path}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

