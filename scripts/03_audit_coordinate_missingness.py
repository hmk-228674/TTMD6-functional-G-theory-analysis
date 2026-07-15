#!/usr/bin/env python3
"""Audit structural all-zero triplets and coordinate discontinuities in TTMD6.

This is a read-only quality-control pass over the prespecified primary cohort
(participant codes 1--30).  It does not exclude trials or select thresholds
from reliability results.  The filename length field is used only to remove
the archive's trailing zero padding; nominal lengths above 200 are inspected
over the 200 rows available in the archive.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BAT_RE = re.compile(
    r"bat_120_(?P<index>\d+)_(?P<action>\d+)_(?P<code>\d+)_200_(?P<rawlen>\d+)\.csv$"
)
JUMP_THRESHOLDS = (25.0, 50.0, 100.0, 200.0, 500.0)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ttmd6-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def read(path: Path) -> np.ndarray:
    a = pd.read_csv(path, header=None, encoding="utf-8-sig").to_numpy(float)
    if a.shape[0] == 400 and np.array_equal(a[:200], a[200:]):
        a = a[:200]
    if a.shape[0] != 200 or not np.isfinite(a).all():
        raise ValueError(f"Unexpected coordinate file: {path} {a.shape}")
    return a


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive zero-based runs for a one-dimensional boolean mask."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(int(x[0]), int(x[-1])) for x in np.split(idx, cuts)]


def main() -> None:
    a = args()
    a.out.mkdir(parents=True, exist_ok=True)
    bat_dir = a.ttmd6_root / "TTMD_cut_bat"
    hum_dir = a.ttmd6_root / "TTMD_cut_hum"

    files = []
    for p in bat_dir.glob("*.csv"):
        m = BAT_RE.fullmatch(p.name)
        if not m:
            continue
        d = {k: int(v) for k, v in m.groupdict().items()}
        if d["code"] <= 30:
            files.append((d, p, hum_dir / p.name.replace("bat_", "human_", 1)))
    files.sort(key=lambda x: x[0]["index"])
    if len(files) != 9000:
        raise RuntimeError(f"Expected 9000 primary pairs, found {len(files)}")

    trial_rows: list[dict] = []
    zero_run_rows: list[dict] = []
    top_jump_rows: list[dict] = []
    jump_counts_body = Counter()
    jump_counts_racket = Counter()
    marker_missing_counts = Counter()

    for k, (meta, bp, hp) in enumerate(files, 1):
        n = min(meta["rawlen"], 200)
        b = read(bp)[:n]
        h = read(hp)[:n].reshape(n, 14, 3)
        bz = np.all(b == 0.0, axis=1)
        hz = np.all(h == 0.0, axis=2)

        for start, end in runs(bz):
            zero_run_rows.append({
                **meta, "waveform": "racket", "marker": 0,
                "start_row_1based": start + 1, "end_row_1based": end + 1,
                "run_length": end - start + 1,
                "touches_start": start == 0, "touches_end": end == n - 1,
                "file": bp.name,
            })
        for marker in range(14):
            if hz[:, marker].any():
                marker_missing_counts[marker + 1] += int(hz[:, marker].sum())
            for start, end in runs(hz[:, marker]):
                zero_run_rows.append({
                    **meta, "waveform": "body", "marker": marker + 1,
                    "start_row_1based": start + 1, "end_row_1based": end + 1,
                    "run_length": end - start + 1,
                    "touches_start": start == 0, "touches_end": end == n - 1,
                    "file": hp.name,
                })

        bd = np.linalg.norm(np.diff(b, axis=0), axis=1)
        bvalid = ~(bz[:-1] | bz[1:])
        bd_valid = bd[bvalid]
        hd = np.linalg.norm(np.diff(h, axis=0), axis=2)
        hvalid = ~(hz[:-1] | hz[1:])
        hd_valid = hd[hvalid]

        for t in JUMP_THRESHOLDS:
            jump_counts_racket[t] += int(np.sum(bd_valid > t))
            jump_counts_body[t] += int(np.sum(hd_valid > t))

        # Store the largest valid coordinate jumps with their exact source row.
        if bd_valid.size:
            masked = np.where(bvalid, bd, -np.inf)
            i = int(np.argmax(masked))
            top_jump_rows.append({
                **meta, "waveform": "racket", "marker": 0,
                "from_row_1based": i + 1, "to_row_1based": i + 2,
                "jump": float(masked[i]), "file": bp.name,
            })
        if hd_valid.size:
            masked = np.where(hvalid, hd, -np.inf)
            flat = int(np.argmax(masked))
            i, marker = np.unravel_index(flat, masked.shape)
            top_jump_rows.append({
                **meta, "waveform": "body", "marker": int(marker) + 1,
                "from_row_1based": int(i) + 1, "to_row_1based": int(i) + 2,
                "jump": float(masked[i, marker]), "file": hp.name,
            })

        valid_markers_per_step = hvalid.sum(axis=1)
        trial_rows.append({
            **meta,
            "observable_rows": n,
            "racket_zero_rows": int(bz.sum()),
            "racket_zero_runs": len(runs(bz)),
            "body_zero_triplets": int(hz.sum()),
            "body_markers_affected": int(hz.any(axis=0).sum()),
            "body_zero_runs": int(sum(len(runs(hz[:, m])) for m in range(14))),
            "body_frames_any_zero_marker": int(hz.any(axis=1).sum()),
            "body_frames_all_zero_markers": int(hz.all(axis=1).sum()),
            "min_valid_markers_per_step": int(valid_markers_per_step.min()) if len(valid_markers_per_step) else 0,
            "steps_lt_14_valid_markers": int(np.sum(valid_markers_per_step < 14)),
            "steps_lt_13_valid_markers": int(np.sum(valid_markers_per_step < 13)),
            "steps_lt_10_valid_markers": int(np.sum(valid_markers_per_step < 10)),
            "max_valid_racket_jump": float(bd_valid.max()) if bd_valid.size else np.nan,
            "p99_valid_racket_jump": float(np.quantile(bd_valid, .99)) if bd_valid.size else np.nan,
            "max_valid_body_marker_jump": float(hd_valid.max()) if hd_valid.size else np.nan,
            "p99_valid_body_marker_jump": float(np.quantile(hd_valid, .99)) if hd_valid.size else np.nan,
            "bat_file": bp.name,
            "human_file": hp.name,
        })
        if k % 1000 == 0:
            print(f"audited {k}/9000", flush=True)

    trials = pd.DataFrame(trial_rows)
    zruns = pd.DataFrame(zero_run_rows)
    jumps = pd.DataFrame(top_jump_rows)
    trials.to_csv(a.out / "trial_coordinate_qc_9000.csv", index=False)
    zruns.to_csv(a.out / "zero_triplet_runs.csv", index=False)
    jumps.sort_values("jump", ascending=False).to_csv(a.out / "largest_valid_jump_per_trial_waveform.csv", index=False)

    affected_body = trials.body_zero_triplets > 0
    affected_racket = trials.racket_zero_rows > 0
    summary = {
        "primary_trials": 9000,
        "body_trials_with_internal_or_boundary_zero_triplet": int(affected_body.sum()),
        "racket_trials_with_internal_or_boundary_zero_triplet": int(affected_racket.sum()),
        "body_zero_triplets_total": int(trials.body_zero_triplets.sum()),
        "racket_zero_rows_total": int(trials.racket_zero_rows.sum()),
        "body_trials_with_fewer_than_13_valid_markers_on_any_step": int((trials.min_valid_markers_per_step < 13).sum()),
        "body_trials_with_fewer_than_10_valid_markers_on_any_step": int((trials.min_valid_markers_per_step < 10).sum()),
        "body_zero_run_length_counts": (
            zruns.loc[zruns.waveform == "body", "run_length"].value_counts().sort_index().to_dict()
            if len(zruns) else {}
        ),
        "racket_zero_run_length_counts": (
            zruns.loc[zruns.waveform == "racket", "run_length"].value_counts().sort_index().to_dict()
            if len(zruns) else {}
        ),
        "body_zero_triplets_by_marker": {str(k): int(v) for k, v in sorted(marker_missing_counts.items())},
        "body_valid_jump_counts_above": {str(k): int(v) for k, v in jump_counts_body.items()},
        "racket_valid_jump_counts_above": {str(k): int(v) for k, v in jump_counts_racket.items()},
        "body_trial_max_valid_jump_quantiles": {
            str(q): float(trials.max_valid_body_marker_jump.quantile(q))
            for q in (0.5, .9, .95, .99, .999, 1.0)
        },
        "racket_trial_max_valid_jump_quantiles": {
            str(q): float(trials.max_valid_racket_jump.quantile(q))
            for q in (0.5, .9, .95, .99, .999, 1.0)
        },
        "affected_body_by_action": {
            str(k): int(v) for k, v in trials.loc[affected_body].groupby("action").size().to_dict().items()
        },
        "affected_racket_by_action": {
            str(k): int(v) for k, v in trials.loc[affected_racket].groupby("action").size().to_dict().items()
        },
    }
    with (a.out / "coordinate_qc_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
