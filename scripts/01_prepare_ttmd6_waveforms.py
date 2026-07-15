#!/usr/bin/env python3
"""Prepare deterministic TTMD6 waveform caches and a complete archive audit.

The public archive contains action-segmented coordinate files stored at a
200-row boundary.  This script does not call them unprocessed raw MOCAP data.
It collapses the known duplicated 400-row blocks, removes zero padding using
the filename length field, computes frame-to-frame displacement magnitudes,
and phase-normalises each observable curve to a common grid.

Primary inferential cohort: participant codes 1--30, actions 1--6, 50 trials
per participant-action cell (9000 matched racket/human pairs).
Codes 31--40 are audited but never silently added to the primary cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BAT_RE = re.compile(
    r"bat_120_(?P<index>\d+)_(?P<action>\d+)_(?P<code>\d+)_200_(?P<rawlen>\d+)\.csv$"
)
ACTION_LABEL = {
    1: "forehand attack",
    2: "forehand drive",
    3: "forehand push",
    4: "backhand attack",
    5: "backhand drive",
    6: "backhand push",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttmd6-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-phase", type=int, default=200)
    return ap.parse_args()


def read_coordinate_file(path: Path) -> tuple[np.ndarray, dict]:
    arr = pd.read_csv(path, header=None, encoding="utf-8-sig").to_numpy(dtype=float)
    original_rows = int(arr.shape[0])
    duplicated_400 = False
    if original_rows == 400:
        if arr[:200].shape == arr[200:400].shape and np.array_equal(arr[:200], arr[200:400]):
            arr = arr[:200]
            duplicated_400 = True
        else:
            raise ValueError(f"Non-identical 400-row file cannot be repaired deterministically: {path}")
    if arr.shape[0] != 200:
        raise ValueError(f"Expected 200 rows after deterministic repair, got {arr.shape}: {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite coordinate detected: {path}")
    return arr, {"rows_on_disk": original_rows, "duplicated_400": duplicated_400}


def observable_segment(arr: np.ndarray, rawlen: int) -> tuple[np.ndarray, dict]:
    n = min(int(rawlen), 200)
    tail_zero_ok = True
    if rawlen < 200:
        tail_zero_ok = bool(np.all(arr[n:] == 0))
        if not tail_zero_ok:
            raise ValueError("Filename length and zero-padded tail disagree")
    return arr[:n], {
        "observable_rows": n,
        "nominal_gt_200": bool(rawlen > 200),
        "zero_padding_rows": max(0, 200 - int(rawlen)),
        "zero_padding_verified": tail_zero_ok,
    }


def phase_normalise(y: np.ndarray, n_phase: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 2 or not np.isfinite(y).all():
        return np.full(n_phase, np.nan)
    return np.interp(
        np.linspace(0.0, 1.0, n_phase),
        np.linspace(0.0, 1.0, y.size),
        y,
    )


def curve_pair(bat: np.ndarray, human: np.ndarray, n_phase: int) -> tuple[np.ndarray, np.ndarray]:
    if bat.shape[1] != 3:
        raise ValueError(f"Racket file must have 3 columns, got {bat.shape}")
    if human.shape[1] != 42:
        raise ValueError(f"Human file must have 42 columns, got {human.shape}")
    rb = np.linalg.norm(np.diff(bat, axis=0), axis=1)
    hc = human.reshape(human.shape[0], 14, 3)
    hb = np.linalg.norm(np.diff(hc, axis=0), axis=2).mean(axis=1)
    return phase_normalise(rb, n_phase), phase_normalise(hb, n_phase)


def effective_pair_digest(bat: np.ndarray, human: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(bat, dtype="<f8").tobytes())
    h.update(b"TTMD6_PAIR_SEPARATOR")
    h.update(np.ascontiguousarray(human, dtype="<f8").tobytes())
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.ttmd6_root
    bat_dir = root / "TTMD_cut_bat"
    hum_dir = root / "TTMD_cut_hum"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)

    records = []
    for bat_path in bat_dir.glob("*.csv"):
        m = BAT_RE.fullmatch(bat_path.name)
        if not m:
            continue
        meta = {k: int(v) for k, v in m.groupdict().items()}
        hum_path = hum_dir / bat_path.name.replace("bat_", "human_", 1)
        if not hum_path.exists():
            raise FileNotFoundError(f"Missing matched human file for {bat_path.name}")
        records.append((meta, bat_path, hum_path))

    records.sort(key=lambda z: (z[0]["code"], z[0]["action"], z[0]["index"]))
    if len(records) != 12000:
        raise RuntimeError(f"Expected 12000 matched pairs, observed {len(records)}")

    Xr = np.full((30, 6, 50, args.n_phase), np.nan)
    Xh = np.full_like(Xr, np.nan)
    rawlen = np.full((30, 6, 50), -1, dtype=np.int16)
    global_index = np.full((30, 6, 50), -1, dtype=np.int32)
    counters = defaultdict(int)
    digest_groups: dict[str, list[dict]] = defaultdict(list)
    manifest_rows = []

    for meta, bat_path, hum_path in records:
        bat_all, bat_qc = read_coordinate_file(bat_path)
        hum_all, hum_qc = read_coordinate_file(hum_path)
        bat, seg_qc_b = observable_segment(bat_all, meta["rawlen"])
        hum, seg_qc_h = observable_segment(hum_all, meta["rawlen"])
        if bat.shape[0] != hum.shape[0]:
            raise RuntimeError(f"Paired effective length mismatch: {bat_path.name}")
        digest = effective_pair_digest(bat, hum)
        digest_groups[digest].append({
            "global_index": meta["index"],
            "participant_code": meta["code"],
            "action_id": meta["action"],
            "raw_length": meta["rawlen"],
            "bat_file": bat_path.name,
            "human_file": hum_path.name,
        })

        trial_in_cell = counters[(meta["code"], meta["action"])]
        counters[(meta["code"], meta["action"])] += 1
        in_primary = 1 <= meta["code"] <= 30 and 1 <= meta["action"] <= 6
        if in_primary:
            if trial_in_cell >= 50:
                raise RuntimeError("More than 50 trials in a primary participant-action cell")
            rc, hc = curve_pair(bat, hum, args.n_phase)
            p, a, t = meta["code"] - 1, meta["action"] - 1, trial_in_cell
            Xr[p, a, t] = rc
            Xh[p, a, t] = hc
            rawlen[p, a, t] = meta["rawlen"]
            global_index[p, a, t] = meta["index"]

        manifest_rows.append({
            "global_index": meta["index"],
            "participant_code": meta["code"],
            "action_id": meta["action"],
            "action_label": ACTION_LABEL[meta["action"]],
            "trial_in_cell": trial_in_cell + 1,
            "raw_length_from_filename": meta["rawlen"],
            "observable_rows": seg_qc_b["observable_rows"],
            "nominal_gt_200": seg_qc_b["nominal_gt_200"],
            "zero_padding_rows": seg_qc_b["zero_padding_rows"],
            "bat_rows_on_disk": bat_qc["rows_on_disk"],
            "human_rows_on_disk": hum_qc["rows_on_disk"],
            "bat_duplicated_400": bat_qc["duplicated_400"],
            "human_duplicated_400": hum_qc["duplicated_400"],
            "effective_pair_sha256": digest,
            "primary_cohort": in_primary,
            "bat_file": bat_path.name,
            "human_file": hum_path.name,
        })

    bad_cells = {k: v for k, v in counters.items() if v != 50}
    if bad_cells:
        raise RuntimeError(f"Every archive participant-action cell must have 50 trials: {bad_cells}")
    if not np.isfinite(Xr).all() or not np.isfinite(Xh).all():
        raise RuntimeError("Primary waveform cache contains missing/non-finite values")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out / "tables" / "Table_01_ArchiveManifest_All12000.csv", index=False)
    manifest[manifest.primary_cohort].to_csv(
        out / "tables" / "Table_02_PrimaryManifest_9000.csv", index=False
    )

    duplicate_rows = []
    group_id = 0
    for digest, items in sorted(digest_groups.items()):
        if len(items) <= 1:
            continue
        group_id += 1
        primary_n = sum(1 <= x["participant_code"] <= 30 for x in items)
        for item in items:
            duplicate_rows.append({
                "duplicate_group": group_id,
                "effective_pair_sha256": digest,
                "group_size": len(items),
                "primary_records_in_group": primary_n,
                **item,
            })
    dup_df = pd.DataFrame(duplicate_rows)
    dup_df.to_csv(out / "tables" / "Table_03_ExactPairDuplicates.csv", index=False)

    summary = {
        "matched_pairs_all_codes": int(len(manifest)),
        "primary_pairs_codes_1_30": int(manifest.primary_cohort.sum()),
        "quarantined_pairs_codes_31_40": int((~manifest.primary_cohort).sum()),
        "primary_nominal_gt_200": int(
            (manifest.primary_cohort & manifest.nominal_gt_200).sum()
        ),
        "all_bat_400_duplicate_files": int(manifest.bat_duplicated_400.sum()),
        "all_human_400_duplicate_files": int(manifest.human_duplicated_400.sum()),
        "exact_duplicate_pair_groups_all_codes": int(group_id),
        "exact_duplicate_groups_entirely_within_primary": int(
            sum(
                all(1 <= x["participant_code"] <= 30 for x in items)
                for items in digest_groups.values()
                if len(items) > 1
            )
        ),
        "n_phase": int(args.n_phase),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    with open(out / "PREPARATION_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    np.save(out / "cache" / "racket_displacement_9000.npy", Xr)
    np.save(out / "cache" / "body14mean_displacement_9000.npy", Xh)
    np.save(out / "cache" / "raw_length_9000.npy", rawlen)
    np.save(out / "cache" / "global_index_9000.npy", global_index)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
