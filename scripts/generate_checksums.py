#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for every public release file."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS"
EXCLUDED_NAMES = {".DS_Store", "SHA256SUMS"}
EXCLUDED_PARTS = {".git", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    )
    content = "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files)
    MANIFEST.write_text(content, encoding="utf-8", newline="\n")
    print(f"Wrote {MANIFEST.name}: {len(files)} files")


if __name__ == "__main__":
    main()
