#!/usr/bin/env python3
"""One-command, clean reproduction of the TTMD6 waveform analyses.

The program verifies the official archive byte-for-byte, extracts it to a
temporary directory, and invokes only the versioned scripts shipped beside
this file.  It never reads an earlier result directory, README, QC_WARNING,
or manuscript table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ARCHIVE_BYTES = 341_074_031
ARCHIVE_MD5 = "1c9ce9cbf79dd35dd22f16a7199e2a8c"
ARCHIVE_SHA256 = "93d1b52a470f14b9dc0ba0600959bff921be891a3da1b71e609bd328224b354d"
DEFAULT_SEED = 20260712

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
SCRIPTS = {
    "prepare": SCRIPT_DIR / "01_prepare_ttmd6_waveforms.py",
    "global": SCRIPT_DIR / "02_global_context_and_bootstrap.py",
    "coordinate_qc": SCRIPT_DIR / "03_audit_coordinate_missingness.py",
    "derive": SCRIPT_DIR / "04_prepare_structural_qc_arrays.py",
    "figures": SCRIPT_DIR / "06_make_publication_figures.py",
    "complete_case": SCRIPT_DIR / "07_complete_case_zero_marker_sensitivity.py",
    "continuous": SCRIPT_DIR / "08_continuous_threshold_bootstrap.py",
    "assumptions": SCRIPT_DIR / "09_assumption_influence_sensitivity.py",
    "pointwise": SCRIPT_DIR / "10_global_pointwise_bootstrap.py",
    "cohort_estimand": SCRIPT_DIR / "11_cohort_estimand_sensitivity.py",
    "reml": SCRIPT_DIR / "run_action_specific_reml.py",
}

PRIMARY_N90 = {
    "racket": [6, 7, 8, 17, 27, 10],
    "body_configuration": [6, 6, 4, 7, 8, 4],
}
HAMPEL_N90 = {
    "racket": [6, 7, 8, 17, 18, 10],
    "body_configuration": [5, 5, 4, 7, 8, 4],
}
FIXED8_N90 = {
    "racket": [6, 7, 8, 17, 27, 10],
    "body_configuration": [6, 6, 3, 6, 8, 4],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rar", required=True, type=Path, help="Official TTMD6.rar archive")
    p.add_argument("--out", required=True, type=Path, help="New reproduction output directory")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--n-balanced-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Allow an existing output directory; every analysis step is still rerun and overwritten.",
    )
    p.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep extracted coordinates under OUT/_extracted_source (normally omitted).",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Verify input, software and script presence, then print the analysis plan without running it.",
    )
    return p.parse_args()


def file_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()  # nosec B324 -- archival identity, not cryptographic security
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preflight(a: argparse.Namespace) -> dict:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    missing_scripts = [str(x) for x in SCRIPTS.values() if not x.is_file()]
    if missing_scripts:
        raise FileNotFoundError(f"Missing release scripts: {missing_scripts}")
    versions = {}
    for module in ("numpy", "pandas", "scipy", "matplotlib", "PIL"):
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            raise RuntimeError(
                f"Missing Python dependency {module!r}; install requirements.txt first"
            ) from exc
        versions[module] = getattr(mod, "__version__", "unknown")
    import numpy as np

    if not hasattr(np, "trapezoid"):
        raise RuntimeError("NumPy >=2.0 is required (numpy.trapezoid is unavailable)")
    rar = a.rar.expanduser().resolve()
    if not rar.is_file():
        raise FileNotFoundError(rar)
    size = rar.stat().st_size
    md5, sha = file_hashes(rar)
    if (size, md5, sha) != (ARCHIVE_BYTES, ARCHIVE_MD5, ARCHIVE_SHA256):
        raise RuntimeError(
            "TTMD6.rar identity mismatch. "
            f"Observed bytes={size}, MD5={md5}, SHA256={sha}; "
            "refusing to analyse an unverified archive."
        )
    bsdtar = shutil.which("bsdtar")
    if not bsdtar:
        raise RuntimeError("bsdtar with RAR5 support is required for archive extraction")
    return {
        "archive": str(rar),
        "bytes": size,
        "md5": md5,
        "sha256": sha,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": versions,
        "bsdtar": bsdtar,
    }


def locate_ttmd6_root(extract_dir: Path) -> Path:
    direct = [extract_dir / "TTMD6", extract_dir]
    candidates = direct + [p.parent for p in extract_dir.rglob("TTMD_cut_bat")]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        bat = candidate / "TTMD_cut_bat"
        hum = candidate / "TTMD_cut_hum"
        if bat.is_dir() and hum.is_dir():
            nb = sum(1 for _ in bat.glob("*.csv"))
            nh = sum(1 for _ in hum.glob("*.csv"))
            if nb != 12_000 or nh != 12_000:
                raise RuntimeError(
                    f"Extracted archive has unexpected CSV counts: racket={nb}, human={nh}"
                )
            return candidate
    raise RuntimeError("Could not locate TTMD_cut_bat and TTMD_cut_hum after extraction")


class Runner:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    def __call__(self, command: list[object]) -> None:
        cmd = [str(x) for x in command]
        printable = shlex.join(cmd)
        stamp = datetime.now(timezone.utc).isoformat()
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{stamp}] START {printable}\n")
        print(f"\n>>> {printable}", flush=True)
        completed = subprocess.run(cmd, check=False)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{datetime.now(timezone.utc).isoformat()}] END exit={completed.returncode}\n")
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, cmd)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def command_plan(a: argparse.Namespace, root: Path, out: Path) -> list[list[object]]:
    py = sys.executable
    w = out / "work"
    prep = w / "preparation"
    manifest = prep / "tables" / "Table_02_PrimaryManifest_9000.csv"
    structural = w / "structural_arrays"
    hampel = w / "hampel_arrays"
    fixed8 = w / "fixed8_arrays"
    fixed13 = w / "fixed13_arrays"
    reml_common = [
        "--manifest", manifest,
        "--n-bootstrap", a.n_bootstrap,
        "--n-balanced-resamples", a.n_balanced_resamples,
        "--seed", a.seed,
    ]
    return [
        [py, SCRIPTS["prepare"], "--ttmd6-root", root, "--out", prep],
        [py, SCRIPTS["coordinate_qc"], "--ttmd6-root", root, "--out", w / "coordinate_qc"],
        [py, SCRIPTS["derive"], "--ttmd6-root", root, "--out", structural],
        [py, SCRIPTS["derive"], "--ttmd6-root", root, "--out", hampel, "--hampel-high-spikes"],
        [
            py, SCRIPTS["derive"], "--ttmd6-root", root, "--out", fixed8,
            "--fixed-human-markers", "1,2,3,4,5,6,11,14",
        ],
        [
            py, SCRIPTS["derive"], "--ttmd6-root", root, "--out", fixed13,
            "--fixed-human-markers", "1,2,3,4,5,6,7,9,10,11,12,13,14",
            "--fixed-min-valid", 9,
        ],
        [
            py, SCRIPTS["reml"], "--arrays-dir", structural,
            "--out", w / "reanalysis_structural" / "results", *reml_common,
        ],
        [
            py, SCRIPTS["reml"], "--arrays-dir", hampel,
            "--out", w / "reanalysis_hampel" / "results", *reml_common,
        ],
        [
            py, SCRIPTS["reml"], "--arrays-dir", fixed8,
            "--out", w / "reanalysis_fixed8" / "results", *reml_common,
        ],
        [
            py, SCRIPTS["global"], "--cache-dir", w / "reanalysis_structural" / "cache",
            "--out", w / "reanalysis_structural" / "global",
            "--n-bootstrap", a.n_bootstrap, "--seed", a.seed,
        ],
        [
            py, SCRIPTS["global"], "--cache-dir", w / "reanalysis_fixed8" / "cache",
            "--out", w / "reanalysis_fixed8" / "global",
            "--n-bootstrap", a.n_bootstrap, "--seed", a.seed,
        ],
        [
            py, SCRIPTS["complete_case"],
            "--arrays", structural / "body_configuration_speed_verified9000.npy",
            "--audit", structural / "structural_missingness_audit_9000.csv",
            "--reml-script", SCRIPTS["reml"],
            "--out", w / "reanalysis_structural" / "Table_S_CompleteCaseZeroMarkerSensitivity.csv",
        ],
        [
            py, SCRIPTS["complete_case"],
            "--arrays", fixed13 / "body_configuration_speed_verified9000.npy",
            "--audit", fixed13 / "structural_missingness_audit_9000.csv",
            "--reml-script", SCRIPTS["reml"],
            "--out", w / "reanalysis_structural" / "Table_S_Fixed13JointCompleteCaseSensitivity.csv",
        ],
        [
            py, SCRIPTS["continuous"],
            "--draws", w / "reanalysis_structural" / "results" / "Table_R4_ClusterBootstrap5000_Draws.csv",
            "--out-draws", w / "reanalysis_structural" / "Table_S_ContinuousThresholdBootstrap5000_Draws.csv",
            "--out-summary", w / "reanalysis_structural" / "Table_S_ContinuousThresholdBootstrap5000_Summary.csv",
        ],
        [
            py, SCRIPTS["assumptions"],
            "--main-arrays-dir", structural,
            "--hampel-arrays-dir", hampel,
            "--manifest", manifest,
            "--main-summary", w / "reanalysis_structural" / "results" / "Table_R2_FullBalanced_Integrated.csv",
            "--bootstrap-draws", w / "reanalysis_structural" / "results" / "Table_R4_ClusterBootstrap5000_Draws.csv",
            "--hampel-audit", hampel / "structural_missingness_audit_9000.csv",
            "--out", w / "reanalysis_structural" / "assumption_influence",
            "--seed", a.seed + 1,
            "--bootstrap-seed", a.seed,
        ],
        [
            py, SCRIPTS["pointwise"],
            "--cache-dir", w / "reanalysis_structural" / "cache",
            "--out", w / "reanalysis_structural" / "global" / "tables" / "Table_S_GlobalPointwiseBootstrapBands.csv",
            "--n-bootstrap", a.n_bootstrap,
            "--seed", a.seed,
        ],
        [
            py, SCRIPTS["cohort_estimand"],
            "--ttmd6-root", root,
            "--manifest", prep / "tables" / "Table_01_ArchiveManifest_All12000.csv",
            "--duplicates", prep / "tables" / "Table_03_ExactPairDuplicates.csv",
            "--out", w / "reanalysis_structural" / "cohort_estimand",
            "--n-bootstrap", a.n_bootstrap,
            "--seed", a.seed + 2,
        ],
        [py, SCRIPTS["figures"], "--root", out, "--out", out / "figures_final"],
    ]


def stage_global_cache(out: Path, source_dir: Path, analysis: str) -> None:
    cache = out / "work" / analysis / "cache"
    link_or_copy(
        source_dir / "racket_speed_verified9000.npy",
        cache / "racket_displacement_9000.npy",
    )
    link_or_copy(
        source_dir / "body_configuration_speed_verified9000.npy",
        cache / "body14mean_displacement_9000.npy",
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def assert_n90(path: Path, expected: dict[str, list[int]]) -> None:
    rows = read_csv_rows(path)
    if len(rows) != 12:
        raise AssertionError(f"Expected 12 action-specific rows in {path}, observed {len(rows)}")
    got: dict[str, list[int]] = {}
    for waveform in expected:
        subset = sorted(
            (r for r in rows if r["waveform"] == waveform),
            key=lambda r: int(r["action_id"]),
        )
        got[waveform] = [int(float(r["required_n_R_L2_90"])) for r in subset]
    if got != expected:
        raise AssertionError(f"R_L2=.90 trial-count regression check failed for {path}: {got} != {expected}")


def validate_outputs(out: Path, a: argparse.Namespace, preflight_info: dict) -> dict:
    w = out / "work"
    prep_summary = json.loads((w / "preparation" / "PREPARATION_SUMMARY.json").read_text(encoding="utf-8"))
    expected_prep = {
        "matched_pairs_all_codes": 12000,
        "primary_pairs_codes_1_30": 9000,
        "quarantined_pairs_codes_31_40": 3000,
        "primary_nominal_gt_200": 1180,
        "all_bat_400_duplicate_files": 50,
        "all_human_400_duplicate_files": 50,
        "exact_duplicate_groups_entirely_within_primary": 0,
    }
    for key, value in expected_prep.items():
        if prep_summary.get(key) != value:
            raise AssertionError(f"Archive audit regression failed: {key}={prep_summary.get(key)} != {value}")

    coordinate = json.loads((w / "coordinate_qc" / "coordinate_qc_summary.json").read_text(encoding="utf-8"))
    if coordinate.get("body_trials_with_internal_or_boundary_zero_triplet") != 190:
        raise AssertionError("Expected 190 body trials containing structural zero triplets")
    if coordinate.get("racket_trials_with_internal_or_boundary_zero_triplet") != 0:
        raise AssertionError("Racket observable segments unexpectedly contain structural zero triplets")

    assert_n90(w / "reanalysis_structural" / "results" / "Table_R2_FullBalanced_Integrated.csv", PRIMARY_N90)
    assert_n90(w / "reanalysis_hampel" / "results" / "Table_R2_FullBalanced_Integrated.csv", HAMPEL_N90)
    assert_n90(w / "reanalysis_fixed8" / "results" / "Table_R2_FullBalanced_Integrated.csv", FIXED8_N90)

    cohort_summary = json.loads(
        (w / "reanalysis_structural" / "cohort_estimand" / "COHORT_ESTIMAND_SENSITIVITY_SUMMARY.json")
        .read_text(encoding="utf-8")
    )
    if cohort_summary.get("exact_pair_records_removed") != 8:
        raise AssertionError("Expected eight exact paired-record removals in the deduplication scenario")
    if cohort_summary.get("backhand_drive_racket_n90") != {
        "L2_trace_ratio": 27,
        "phase_mean_pointwise_ratio": 17,
    }:
        raise AssertionError("Backhand-drive aggregation regression check failed")

    for analysis in ("reanalysis_structural", "reanalysis_hampel", "reanalysis_fixed8"):
        run = json.loads((w / analysis / "results" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        if not run.get("self_tests_all_passed"):
            raise AssertionError(f"REML self-tests failed in {analysis}")
        if run.get("n_bootstrap_whole_athlete") != a.n_bootstrap:
            raise AssertionError(f"Bootstrap count mismatch in {analysis}")
        if run.get("n_balanced_subsamples_without_replacement") != a.n_balanced_resamples:
            raise AssertionError(f"Balanced-resampling count mismatch in {analysis}")
        if not (w / analysis / "results" / "QC_NOTE.txt").is_file():
            raise AssertionError(f"Current QC_NOTE.txt missing in {analysis}")

    forbidden = list(out.rglob("QC_WARNING.txt"))
    if forbidden:
        raise AssertionError(f"Forbidden stale QC_WARNING files detected: {forbidden}")

    required = [
        w / "reanalysis_structural" / "Table_S_CompleteCaseZeroMarkerSensitivity.csv",
        w / "reanalysis_structural" / "Table_S_Fixed13JointCompleteCaseSensitivity.csv",
        w / "reanalysis_structural" / "Table_S_ContinuousThresholdBootstrap5000_Summary.csv",
        w / "reanalysis_structural" / "global" / "tables" / "Table_04_GlobalVarianceSummary.csv",
        w / "reanalysis_structural" / "global" / "tables" / "Table_S_GlobalPointwiseBootstrapBands.csv",
        w / "reanalysis_fixed8" / "global" / "tables" / "Table_04_GlobalVarianceSummary.csv",
        w / "reanalysis_structural" / "assumption_influence" / "Table_S_ArchiveOrderResidualACF.csv",
        w / "reanalysis_structural" / "assumption_influence" / "Table_S_AR1CorrelationScenarios.csv",
        w / "reanalysis_structural" / "assumption_influence" / "Table_S_PeakRegistrationSensitivity.csv",
        w / "reanalysis_structural" / "assumption_influence" / "Table_S_BackhandDriveInfluenceAndFlags.csv",
        w / "reanalysis_structural" / "assumption_influence" / "Table_S_BootstrapDiagnostics.csv",
        w / "reanalysis_structural" / "cohort_estimand" / "Table_S_CohortAndDedupSensitivity.csv",
        w / "reanalysis_structural" / "cohort_estimand" / "Table_S_CohortBootstrapThresholds.csv",
        w / "reanalysis_structural" / "cohort_estimand" / "Table_S_FunctionalAggregationComparison.csv",
        w / "reanalysis_structural" / "cohort_estimand" / "Table_S_ExactDuplicateRemovalAudit.csv",
        w / "reanalysis_structural" / "cohort_estimand" / "Table_S_14JointAnatomicalMapping.csv",
    ]
    required.extend(
        out / "figures_final" / f"{name}.tif"
        for name in (
            "Figure1_study_design_and_inference_boundary",
            "Figure2_pointwise_dispersion_for_six_fixed_labels",
            "Figure3_action_specific_relative_reliability",
            "Figure4_dependence_registration_and_quality_sensitivity",
            "FigureS1_archive_boundary_and_resampling_robustness",
            "FigureS2_distribution_of_Hampel_type_flags",
            "FigureS3_pointwise_marginal_bootstrap_intervals",
        )
    )
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise AssertionError(f"Required outputs missing: {missing}")

    figure_qa = json.loads((out / "figures_final" / "Figure_QA.json").read_text(encoding="utf-8"))
    if not figure_qa.get("hard_checks_pass") or figure_qa.get("text_boundary_violations_total") != 0:
        raise AssertionError(f"Figure QA failed: {figure_qa}")

    return {
        "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input": preflight_info,
        "seed": a.seed,
        "n_bootstrap_whole_athlete": a.n_bootstrap,
        "n_balanced_resamples": a.n_balanced_resamples,
        "regression_checks": {
            "archive_structure": "PASS",
            "coordinate_structural_zero_audit": "PASS",
            "primary_action_specific_R_L2_90": "PASS",
            "hampel_sensitivity_R_L2_90": "PASS",
            "fixed8_common_composition_R_L2_90": "PASS",
            "cohort_boundary_and_exact_deduplication_sensitivity": "PASS",
            "trace_vs_phase_mean_pointwise_thresholds": "PASS",
            "reml_self_tests": "PASS",
            "forbidden_stale_QC_WARNING": "ABSENT",
            "required_tables_and_figures": "PASS",
            "figure_dimensions_formats_and_text_bounds": "PASS",
        },
    }


def write_inventory(out: Path) -> None:
    rows = []
    excluded = (out / "_extracted_source").resolve()
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.name == "FILE_INVENTORY.csv":
            continue
        try:
            path.resolve().relative_to(excluded)
            continue
        except ValueError:
            pass
        rows.append({
            "relative_path": str(path.relative_to(out)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    with (out / "FILE_INVENTORY.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def execute(a: argparse.Namespace, preflight_info: dict) -> None:
    out = a.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise RuntimeError(f"Output directory is not empty: {out}. Use a new path or pass --resume.")
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(out / "REPRODUCTION_COMMANDS.log")

    if a.keep_extracted:
        extract_dir = out / "_extracted_source"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if not (a.resume and any(extract_dir.iterdir())):
            runner([preflight_info["bsdtar"], "-xf", a.rar.expanduser().resolve(), "-C", extract_dir])
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="ttmd6_reproduction_")
        extract_dir = Path(temp_context.name)
        runner([preflight_info["bsdtar"], "-xf", a.rar.expanduser().resolve(), "-C", extract_dir])

    try:
        root = locate_ttmd6_root(extract_dir)
        commands = command_plan(a, root, out)
        for index, command in enumerate(commands):
            # Global scripts use neutral cache filenames.  Hard links avoid a
            # second in-memory or on-disk transformation of the derived arrays.
            if index == 9:
                stage_global_cache(out, out / "work" / "structural_arrays", "reanalysis_structural")
                stage_global_cache(out, out / "work" / "fixed8_arrays", "reanalysis_fixed8")
            runner(command)
        status = validate_outputs(out, a, preflight_info)
        (out / "REPRODUCTION_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_inventory(out)
        print(f"\nReproduction PASS: {out}", flush=True)
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def main() -> None:
    a = parse_args()
    if a.n_bootstrap < 1 or a.n_balanced_resamples < 1:
        raise ValueError("Resampling counts must be positive")
    info = preflight(a)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if a.plan_only:
        placeholder = Path("<EXTRACTED_TTMD6_ROOT>")
        print("\nValidated analysis plan:")
        for i, command in enumerate(command_plan(a, placeholder, a.out.expanduser().resolve()), 1):
            print(f"{i:02d}. {shlex.join(str(x) for x in command)}")
        return
    execute(a, info)


if __name__ == "__main__":
    main()
