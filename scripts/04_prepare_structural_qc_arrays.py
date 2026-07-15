#!/usr/bin/env python3
"""Rebuild TTMD6 displacement waveforms with structural missing-marker handling.

The archive represents a missing human joint at some frames as the exact
triplet [0, 0, 0].  A transition to/from that sentinel is not motion.  For
each adjacent-frame interval this script averages displacement only across
joint trajectories observed (nonzero) at both endpoints.  No coordinate is
gap-filled and no trial is removed.  The archive audit established that at
least 10 of 14 joints remain usable on every interval in the 9,000-trial
primary cohort.  Racket trajectories contain no all-zero triplets in their
observable segments and are therefore derived unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BAT_RE = re.compile(
    r"bat_120_(?P<index>\d+)_(?P<action>\d+)_(?P<code>\d+)_200_(?P<rawlen>\d+)\.csv$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ttmd6-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-phase", type=int, default=200)
    p.add_argument(
        "--hampel-high-spikes",
        action="store_true",
        help="Sensitivity only: replace isolated high displacement spikes before phase normalisation.",
    )
    p.add_argument("--hampel-radius", type=int, default=5)
    p.add_argument("--hampel-z", type=float, default=8.0)
    p.add_argument("--hampel-ratio", type=float, default=4.0)
    p.add_argument(
        "--fixed-human-markers",
        type=str,
        default="",
        help="Sensitivity only: comma-separated 1-based human-marker indices kept for every frame.",
    )
    p.add_argument(
        "--fixed-min-valid",
        type=int,
        default=0,
        help="Minimum usable selected markers; default requires every selected marker.",
    )
    return p.parse_args()


def read_coordinates(path: Path) -> np.ndarray:
    a = pd.read_csv(path, header=None, encoding="utf-8-sig").to_numpy(float)
    if a.shape[0] == 400 and np.array_equal(a[:200], a[200:400]):
        a = a[:200]
    if a.shape[0] != 200 or not np.isfinite(a).all():
        raise ValueError(f"Unexpected file structure: {path} {a.shape}")
    return a


def phase_normalise(y: np.ndarray, n_phase: int) -> np.ndarray:
    if y.size < 2 or not np.isfinite(y).all():
        raise ValueError("A finite displacement sequence with >=2 points is required")
    return np.interp(
        np.linspace(0.0, 1.0, n_phase),
        np.linspace(0.0, 1.0, y.size),
        y,
    )


def replace_isolated_high_spikes(
    y: np.ndarray,
    radius: int,
    z_cut: float,
    ratio_cut: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided local Hampel sensitivity filter for displacement magnitudes.

    A point must exceed both the local median by ``z_cut`` scaled MADs and
    ``ratio_cut`` times that median.  The centre point is excluded from its
    own reference window.  Flagged values are linearly interpolated from the
    nearest unflagged displacement samples.  This is deliberately a
    sensitivity transformation, never the structural-missingness primary.
    """
    y = np.asarray(y, float)
    flag = np.zeros(y.size, dtype=bool)
    eps = np.finfo(float).eps
    for i in range(y.size):
        lo, hi = max(0, i - radius), min(y.size, i + radius + 1)
        ref = np.concatenate((y[lo:i], y[i + 1 : hi]))
        if ref.size < 4:
            continue
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med))) * 1.4826
        threshold = max(med + z_cut * max(mad, eps), ratio_cut * max(med, eps))
        flag[i] = y[i] > threshold
    if not flag.any():
        return y.copy(), flag
    good = ~flag
    if good.sum() < 2:
        raise ValueError("Hampel sensitivity left fewer than two usable displacement samples")
    out = y.copy()
    out[flag] = np.interp(np.flatnonzero(flag), np.flatnonzero(good), y[good])
    return out, flag


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    a = parse_args()
    fixed_markers = (
        [int(x) - 1 for x in a.fixed_human_markers.split(",") if x.strip()]
        if a.fixed_human_markers else None
    )
    if fixed_markers is not None and (not fixed_markers or min(fixed_markers) < 0 or max(fixed_markers) >= 14):
        raise ValueError("--fixed-human-markers must contain 1-based indices from 1 to 14")
    a.out.mkdir(parents=True, exist_ok=True)
    bat_dir = a.ttmd6_root / "TTMD_cut_bat"
    hum_dir = a.ttmd6_root / "TTMD_cut_hum"

    records = []
    for bp in bat_dir.glob("*.csv"):
        m = BAT_RE.fullmatch(bp.name)
        if not m:
            continue
        meta = {k: int(v) for k, v in m.groupdict().items()}
        if 1 <= meta["code"] <= 30:
            records.append((meta, bp, hum_dir / bp.name.replace("bat_", "human_", 1)))
    records.sort(key=lambda z: (z[0]["code"], z[0]["action"], z[0]["index"]))
    if len(records) != 9000:
        raise RuntimeError(f"Expected 9000 pairs, observed {len(records)}")

    xr = np.empty((30, 6, 50, a.n_phase), float)
    xh = np.empty_like(xr)
    counters = defaultdict(int)
    audit = []

    for k, (meta, bp, hp) in enumerate(records, 1):
        n = min(meta["rawlen"], 200)
        bat = read_coordinates(bp)[:n]
        human = read_coordinates(hp)[:n].reshape(n, 14, 3)
        if fixed_markers is not None:
            human = human[:, fixed_markers, :]
        if n < 3:
            raise ValueError(f"Too-short observable sequence: {bp.name}")

        bat_zero = np.all(bat == 0.0, axis=1)
        if bat_zero.any():
            raise ValueError(f"Unexpected racket all-zero triplet in observable segment: {bp.name}")
        racket_step = np.linalg.norm(np.diff(bat, axis=0), axis=1)

        human_zero = np.all(human == 0.0, axis=2)
        valid_pair = ~(human_zero[:-1] | human_zero[1:])
        marker_step = np.linalg.norm(np.diff(human, axis=0), axis=2)
        valid_count = valid_pair.sum(axis=1)
        min_required = (
            a.fixed_min_valid if fixed_markers is not None and a.fixed_min_valid > 0
            else (len(fixed_markers) if fixed_markers is not None else 10)
        )
        if np.any(valid_count < min_required):
            raise ValueError(f"Fewer than {min_required} usable joints on an interval: {hp.name}")
        body_step = np.divide(
            np.where(valid_pair, marker_step, 0.0).sum(axis=1),
            valid_count,
        )
        racket_spikes = np.zeros(racket_step.size, dtype=bool)
        body_spikes = np.zeros(body_step.size, dtype=bool)
        if a.hampel_high_spikes:
            racket_step, racket_spikes = replace_isolated_high_spikes(
                racket_step, a.hampel_radius, a.hampel_z, a.hampel_ratio
            )
            body_step, body_spikes = replace_isolated_high_spikes(
                body_step, a.hampel_radius, a.hampel_z, a.hampel_ratio
            )

        trial = counters[(meta["code"], meta["action"])]
        counters[(meta["code"], meta["action"])] += 1
        p, act = meta["code"] - 1, meta["action"] - 1
        xr[p, act, trial] = phase_normalise(racket_step, a.n_phase)
        xh[p, act, trial] = phase_normalise(body_step, a.n_phase)
        audit.append({
            "global_index": meta["index"],
            "participant_code": meta["code"],
            "action_id": meta["action"],
            "trial_in_cell": trial + 1,
            "raw_length_from_filename": meta["rawlen"],
            "observable_rows": n,
            "body_zero_triplets": int(human_zero.sum()),
            "body_markers_affected": int(human_zero.any(axis=0).sum()),
            "min_usable_markers_per_step": int(valid_count.min()),
            "steps_with_13_usable_markers": int(np.sum(valid_count == 13)),
            "steps_with_12_or_fewer_usable_markers": int(np.sum(valid_count <= 12)),
            "racket_hampel_high_spikes": int(racket_spikes.sum()),
            "body_hampel_high_spikes": int(body_spikes.sum()),
            "racket_hampel_flag_steps_0based": ";".join(
                str(int(x)) for x in np.flatnonzero(racket_spikes)
            ),
            "body_hampel_flag_steps_0based": ";".join(
                str(int(x)) for x in np.flatnonzero(body_spikes)
            ),
            "racket_hampel_flag_time_pct": ";".join(
                f"{100.0 * int(x) / max(1, racket_spikes.size - 1):.6f}"
                for x in np.flatnonzero(racket_spikes)
            ),
            "body_hampel_flag_time_pct": ";".join(
                f"{100.0 * int(x) / max(1, body_spikes.size - 1):.6f}"
                for x in np.flatnonzero(body_spikes)
            ),
            "bat_file": bp.name,
            "human_file": hp.name,
        })
        if k % 1000 == 0:
            print(f"prepared {k}/9000", flush=True)

    if any(v != 50 for v in counters.values()) or len(counters) != 180:
        raise RuntimeError("Primary cohort is not a complete 30 x 6 x 50 design")
    if not np.isfinite(xr).all() or not np.isfinite(xh).all():
        raise RuntimeError("Non-finite output")

    rp = a.out / "racket_speed_verified9000.npy"
    hp = a.out / "body_configuration_speed_verified9000.npy"
    np.save(rp, xr)
    np.save(hp, xh)
    pd.DataFrame(audit).to_csv(a.out / "structural_missingness_audit_9000.csv", index=False)
    summary = {
        "policy": "For each adjacent-frame interval, omit a human joint if either endpoint is exactly [0,0,0]; average the remaining joint displacement magnitudes. No coordinate gap fill or trial deletion.",
        "hampel_high_spike_sensitivity": bool(a.hampel_high_spikes),
        "hampel_parameters": {
            "radius": a.hampel_radius,
            "window": 2 * a.hampel_radius + 1,
            "scaled_mad_cut": a.hampel_z,
            "local_median_ratio_cut": a.hampel_ratio,
            "replacement": "linear interpolation of displacement magnitude from nearest unflagged samples",
        } if a.hampel_high_spikes else None,
        "primary_trials": 9000,
        "phase_nodes": a.n_phase,
        "affected_body_trials": int(sum(x["body_zero_triplets"] > 0 for x in audit)),
        "body_zero_triplets": int(sum(x["body_zero_triplets"] for x in audit)),
        "minimum_usable_markers_any_step": int(min(x["min_usable_markers_per_step"] for x in audit)),
        "fixed_human_markers_1based": [x + 1 for x in fixed_markers] if fixed_markers is not None else None,
        "racket_all_zero_triplets": 0,
        "racket_hampel_high_spikes": int(sum(x["racket_hampel_high_spikes"] for x in audit)),
        "body_hampel_high_spikes": int(sum(x["body_hampel_high_spikes"] for x in audit)),
        "racket_array_sha256": sha256(rp),
        "body_array_sha256": sha256(hp),
    }
    (a.out / "STRUCTURAL_QC_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
