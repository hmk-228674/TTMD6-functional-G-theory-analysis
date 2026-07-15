#!/usr/bin/env python3
"""Fail closed when a public-release repository contract is violated."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".rar", ".npy"}
FORBIDDEN_TEXT = {
    "AUTHOR_INPUT",
    "AUTHOR INPUT",
    "G_ratio_",
    "required_m_ratio",
    "action_label_zh",
    "/Users/mac/",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yml", ".yaml", ".csv", ".txt", ".cff", ".sha256"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    )


def check() -> list[str]:
    failures: list[str] = []
    files = release_files()

    for path in files:
        rel = path.relative_to(ROOT)
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.name.startswith("._") or "__MACOSX" in rel.parts:
            failures.append(f"Forbidden macOS archive metadata: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"Forbidden third-party/cache file: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8-sig", errors="strict")
            for token in FORBIDDEN_TEXT:
                if token in text:
                    failures.append(f"Forbidden token {token!r} in {rel}")

    status_path = ROOT / "reference_results/reproduction/REPRODUCTION_STATUS.json"
    figure_qa_path = ROOT / "figure_source_data/Figure_QA.json"
    if not status_path.is_file() or json.loads(status_path.read_text())["status"] != "PASS":
        failures.append("Reference reproduction status is missing or not PASS")
    if not figure_qa_path.is_file():
        failures.append("Figure QA record is missing")
    else:
        qa = json.loads(figure_qa_path.read_text())
        if not qa.get("hard_checks_pass") or qa.get("text_boundary_violations_total") != 0:
            failures.append(f"Figure QA failed: {qa}")

    primary = ROOT / "reference_results/primary/Table_R2_FullBalanced_Integrated.csv"
    if not primary.is_file():
        failures.append("Primary integrated results are missing")
    else:
        table = pd.read_csv(primary)
        required = {"waveform", "action_id", "action_label", "R_L2_m50", "required_n_R_L2_90"}
        missing = required.difference(table.columns)
        if missing:
            failures.append(f"Primary table is missing columns: {sorted(missing)}")
        if set(table.action_label) != {
            "forehand attack", "forehand drive", "forehand push",
            "backhand attack", "backhand drive", "backhand push",
        }:
            failures.append("Primary table action labels are not the canonical six English labels")

    figure_bases = {path.stem for path in (ROOT / "figures").glob("*.png")}
    if len(figure_bases) != 7:
        failures.append(f"Expected seven figures, found {len(figure_bases)}")
    for base in figure_bases:
        for suffix in (".png", ".pdf", ".svg", ".tif"):
            if not (ROOT / "figures" / f"{base}{suffix}").is_file():
                failures.append(f"Missing figure format: figures/{base}{suffix}")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if 'family-names: "Han"' not in citation or 'given-names: "Mingke"' not in citation:
        failures.append("CITATION.cff does not identify the software author as Mingke Han")
    if 'family-names: "hmk-228674"' in citation:
        failures.append("CITATION.cff uses a GitHub username as the software author")
    citation_lines = citation.splitlines()
    has_version = any(line.startswith("version:") for line in citation_lines)
    has_release_date = any(line.startswith("date-released:") for line in citation_lines)
    if has_version != has_release_date:
        failures.append("CITATION.cff version and date-released must be present or absent together")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "Copyright (c) 2026 Mingke Han" not in license_text:
        failures.append("LICENSE copyright holder is not Mingke Han")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "python -m pip install -r requirements-lock.txt" not in readme:
        failures.append("README exact-reproduction command does not install requirements-lock.txt")
    if "use Python 3.12 and `requirements-lock.txt`" not in readme:
        failures.append("README does not state the Python 3.12 exact-environment requirement")
    if "not covered by either repository license" not in readme:
        failures.append("README does not state the raw-archive license boundary")

    manifest = ROOT / "SHA256SUMS"
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = ROOT / relative
            if not path.is_file():
                failures.append(f"Manifest file missing: {relative}")
            elif sha256(path) != expected:
                failures.append(f"Checksum mismatch: {relative}")

    return failures


def main() -> None:
    failures = check()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("Repository release contract: PASS")


if __name__ == "__main__":
    main()
