#!/usr/bin/env python3
"""Action-specific functional reliability reanalysis for TTMD6.

The cached arrays have shape athlete x action x trial x phase node.  This
script fits, separately for each of six actions and two waveform families,

    y_ij(q) = mu(q) + athlete_i(q) + error_ij(q).

It provides four analyses requested for the manuscript revision:

1. Full 30 x 50 balanced, non-negative REML at every phase node.  In a
   balanced one-way model the interior REML solution is the usual ANOVA
   solution.  When MS_between < MS_within, the random-effect variance is put
   on its zero boundary and the two sums of squares are pooled for the
   residual REML estimate.
2. A 5,000-replicate whole-athlete cluster bootstrap and D-study.
3. Exclusion of trials whose filename reports raw_length > 200, followed by
   exact unbalanced random-intercept profile REML at every phase node.
4. A 1,000-replicate within-athlete/action balanced subsampling sensitivity
   analysis after exclusion, sampling without replacement down to the
   action-specific minimum cell size.

The implementation uses only sufficient statistics for REML and resampling.
All random draws are reproducible from one explicit seed.  The same athlete
bootstrap and the same within-cell trial selections are reused across the two
waveform families, preserving paired comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar


SEED_DEFAULT = 20260712
M_GRID = (1, 2, 5, 10, 15, 20, 30, 40, 50)
TARGETS = (0.80, 0.90)
ACTION_EN = {
    1: "forehand_attack",
    2: "forehand_drive",
    3: "forehand_push",
    4: "backhand_attack",
    5: "backhand_drive",
    6: "backhand_push",
}
ACTION_LABEL = {
    1: "forehand attack",
    2: "forehand drive",
    3: "forehand push",
    4: "backhand attack",
    5: "backhand drive",
    6: "backhand push",
}
ARRAY_FILES = {
    "racket": "racket_speed_verified9000.npy",
    "body_configuration": "body_configuration_speed_verified9000.npy",
}


@dataclass
class Components:
    """Pointwise one-way random-effect REML quantities."""

    between: np.ndarray
    within: np.ndarray
    ss_between: np.ndarray
    ss_within: np.ndarray
    ms_between: np.ndarray
    ms_within: np.ndarray
    interior: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrays-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-balanced-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument(
        "--skip-numerical-self-test",
        action="store_true",
        help="Skip the slower actual-data profile-REML equivalence check.",
    )
    return parser.parse_args()


def setup_logging(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ttmd6_action_reml")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """UTF-8 BOM makes the Chinese labels open cleanly in spreadsheet apps."""
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def integration_weights(n_phase: int) -> np.ndarray:
    """Weights exactly matching trapezoidal integration on 0--100 divided by 100."""
    if n_phase < 2:
        raise ValueError("At least two phase nodes are required")
    w = np.ones(n_phase, dtype=float) / (n_phase - 1)
    w[[0, -1]] *= 0.5
    return w


def integrate(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(y) @ weights


def reliability_curve(between: np.ndarray, within: np.ndarray, m: int) -> np.ndarray:
    den = between + within / float(m)
    return np.divide(between, den, out=np.zeros_like(den, dtype=float), where=den > 0)


def integrated_relative_reliability(between_i: np.ndarray, within_i: np.ndarray, m: int) -> np.ndarray:
    den = between_i + within_i / float(m)
    return np.divide(between_i, den, out=np.zeros_like(den, dtype=float), where=den > 0)


def required_trials(between_i: np.ndarray | float, within_i: np.ndarray | float, target: float) -> np.ndarray:
    b = np.asarray(between_i, dtype=float)
    w = np.asarray(within_i, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = target * w / ((1.0 - target) * b)
    # An algebraically integral threshold can land one ULP above the integer
    # (for example 16.000000000000004).  Moving one representable value toward
    # -inf prevents a spurious extra trial while leaving genuinely non-integral
    # thresholds on the same side of the ceiling.
    corrected = np.nextafter(raw, -np.inf)
    ans = np.where(b > 0, np.maximum(1.0, np.ceil(corrected)), np.inf)
    return ans


def required_m_pointwise(
    between: np.ndarray,
    within: np.ndarray,
    weights: np.ndarray,
    target: float,
    max_m: int = 1_000_000,
) -> float:
    """Smallest integer m whose phase-averaged pointwise G reaches target."""
    limit = integrate(np.where(between > 0, 1.0, 0.0), weights)
    if limit + 1e-14 < target:
        return math.inf
    lo, hi = 1, 1
    while hi < max_m and integrate(reliability_curve(between, within, hi), weights) < target:
        hi *= 2
    if hi > max_m:
        hi = max_m
    if integrate(reliability_curve(between, within, hi), weights) < target:
        return math.inf
    while lo < hi:
        mid = (lo + hi) // 2
        if integrate(reliability_curve(between, within, mid), weights) >= target:
            hi = mid
        else:
            lo = mid + 1
    return float(lo)


def required_m_pointwise_batch(
    between: np.ndarray,
    within: np.ndarray,
    weights: np.ndarray,
    target: float,
    max_m: int = 1_000_000,
) -> np.ndarray:
    """Vectorised integer search for phase-averaged G thresholds."""
    b = np.asarray(between, dtype=float)
    w = np.asarray(within, dtype=float)
    if b.ndim != 2 or w.shape != b.shape:
        raise ValueError("Batch pointwise threshold input must be replicate x phase")

    def evaluate(m: np.ndarray) -> np.ndarray:
        den = b + w / m[:, None]
        g = np.divide(b, den, out=np.zeros_like(den), where=den > 0)
        return g @ weights

    n_rep = b.shape[0]
    limit = (b > 0).astype(float) @ weights
    reachable = limit + 1e-14 >= target
    lo = np.ones(n_rep, dtype=np.int64)
    hi = np.ones(n_rep, dtype=np.int64)
    g_hi = evaluate(hi)
    active = reachable & (g_hi < target)
    while np.any(active):
        hi[active] = np.minimum(hi[active] * 2, max_m)
        g_hi = evaluate(hi)
        stuck = active & (hi >= max_m) & (g_hi < target)
        reachable[stuck] = False
        active = reachable & (g_hi < target)
    # Binary search each reachable replicate inside [1, first successful hi].
    active = reachable & (lo < hi)
    while np.any(active):
        mid = (lo + hi) // 2
        g_mid = evaluate(mid)
        success = active & (g_mid >= target)
        failure = active & ~success
        hi[success] = mid[success]
        lo[failure] = mid[failure] + 1
        active = reachable & (lo < hi)
    return np.where(reachable, lo.astype(float), np.inf)


def components_from_sums_of_squares(
    ss_between: np.ndarray,
    ss_within: np.ndarray,
    n_groups: int,
    m: int,
) -> Components:
    """Constrained balanced REML, allowing arbitrary leading replicate axes."""
    msb = ss_between / (n_groups - 1)
    msw = ss_within / (n_groups * (m - 1))
    interior = msb >= msw
    between = np.where(interior, (msb - msw) / m, 0.0)
    pooled = (ss_between + ss_within) / (n_groups * m - 1)
    within = np.where(interior, msw, pooled)
    return Components(between, within, ss_between, ss_within, msb, msw, interior)


def balanced_reml(data: np.ndarray) -> tuple[Components, np.ndarray, np.ndarray]:
    """Fit balanced athlete x trial x phase data and return group sufficient stats."""
    if data.ndim != 3:
        raise ValueError(f"Expected athlete x trial x phase, got {data.shape}")
    n_groups, m, _ = data.shape
    group_mean = data.mean(axis=1)
    within_by_group = ((data - group_mean[:, None, :]) ** 2).sum(axis=1)
    grand = group_mean.mean(axis=0)
    ss_between = m * ((group_mean - grand) ** 2).sum(axis=0)
    ss_within = within_by_group.sum(axis=0)
    return components_from_sums_of_squares(ss_between, ss_within, n_groups, m), group_mean, within_by_group


def bootstrap_balanced_from_counts(
    group_mean: np.ndarray,
    within_by_group: np.ndarray,
    m: int,
    counts: np.ndarray,
) -> Components:
    """Whole-cluster bootstrap via multinomial cluster multiplicities."""
    n_groups = group_mean.shape[0]
    if counts.shape[1] != n_groups or not np.all(counts.sum(axis=1) == n_groups):
        raise ValueError("Invalid athlete bootstrap count matrix")
    sum_mean = counts @ group_mean
    sum_mean_sq = counts @ (group_mean * group_mean)
    ss_between = m * np.maximum(sum_mean_sq - sum_mean * sum_mean / n_groups, 0.0)
    ss_within = counts @ within_by_group
    return components_from_sums_of_squares(ss_between, ss_within, n_groups, m)


def component_summary(
    comp: Components,
    weights: np.ndarray,
    m_grid: Iterable[int] = M_GRID,
) -> dict[str, float]:
    b_i = float(integrate(comp.between, weights))
    w_i = float(integrate(comp.within, weights))
    out: dict[str, float] = {
        "integrated_sigma2_athlete": b_i,
        "integrated_sigma2_trial": w_i,
        "boundary_nodes": int(np.size(comp.interior) - np.count_nonzero(comp.interior)),
        "boundary_fraction": float(1.0 - np.mean(comp.interior)),
    }
    for m in m_grid:
        out[f"R_L2_m{m}"] = float(integrated_relative_reliability(np.array(b_i), np.array(w_i), m))
        out[f"mean_pointwise_reliability_m{m}"] = float(integrate(reliability_curve(comp.between, comp.within, m), weights))
    for target in TARGETS:
        suffix = str(int(target * 100))
        out[f"required_n_R_L2_{suffix}"] = float(required_trials(b_i, w_i, target))
        out[f"required_m_pointwise_G{suffix}"] = required_m_pointwise(
            comp.between, comp.within, weights, target
        )
    return out


def dstudy_rows(
    waveform: str,
    action_id: int,
    comp: Components,
    weights: np.ndarray,
    analysis: str,
) -> list[dict]:
    b_i = float(integrate(comp.between, weights))
    w_i = float(integrate(comp.within, weights))
    rows = []
    for m in M_GRID:
        rows.append(
            {
                "analysis": analysis,
                "waveform": waveform,
                "action_id": action_id,
                "action_label_en": ACTION_EN[action_id],
                "action_label": ACTION_LABEL[action_id],
                "trials_m": m,
                "R_L2_integrated_components": float(integrated_relative_reliability(np.array(b_i), np.array(w_i), m)),
                "mean_pointwise_reliability": float(integrate(reliability_curve(comp.between, comp.within, m), weights)),
            }
        )
    return rows


def pointwise_rows(
    waveform: str,
    action_id: int,
    comp: Components,
    n_groups: int,
    n_total: int,
    min_trials: int,
    max_trials: int,
    method: str,
) -> pd.DataFrame:
    n_phase = comp.between.size
    data: dict[str, np.ndarray | str | int] = {
        "analysis": method,
        "waveform": waveform,
        "action_id": action_id,
        "action_label_en": ACTION_EN[action_id],
        "action_label": ACTION_LABEL[action_id],
        "phase_node": np.arange(n_phase),
        "phase_percent": np.linspace(0.0, 100.0, n_phase),
        "n_athletes": n_groups,
        "n_trials_total": n_total,
        "min_trials_per_athlete": min_trials,
        "max_trials_per_athlete": max_trials,
        "sigma2_athlete": comp.between,
        "sigma2_trial": comp.within,
        "boundary_sigma2_athlete_zero": ~comp.interior,
        "ss_between": comp.ss_between,
        "ss_within": comp.ss_within,
        "ms_between": comp.ms_between,
        "ms_within": comp.ms_within,
    }
    for m in M_GRID:
        data[f"pointwise_relative_reliability_m{m}"] = reliability_curve(comp.between, comp.within, m)
    return pd.DataFrame(data)


def validate_and_load(arrays_dir: Path, manifest_path: Path, logger: logging.Logger):
    arrays: dict[str, np.ndarray] = {}
    for label, filename in ARRAY_FILES.items():
        path = arrays_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        arrays[label] = np.load(path, mmap_mode="r")
        if arrays[label].shape != (30, 6, 50, 200):
            raise ValueError(f"Unexpected {label} array shape: {arrays[label].shape}")
        if not np.isfinite(arrays[label]).all():
            raise ValueError(f"Non-finite values in {path}")
        logger.info("Loaded %s: shape=%s, dtype=%s", label, arrays[label].shape, arrays[label].dtype)

    manifest = pd.read_csv(manifest_path)
    needed = {
        "participant_code",
        "action_id",
        "trial_in_cell",
        "raw_length_from_filename",
    }
    if len(manifest) != 9000 or not needed.issubset(manifest.columns):
        raise ValueError("Manifest is not the expected verified 9,000-trial manifest")
    duplicated = manifest.duplicated(["participant_code", "action_id", "trial_in_cell"]).sum()
    if duplicated:
        raise ValueError(f"Manifest contains {duplicated} duplicated cell/trial rows")
    expected = pd.MultiIndex.from_product(
        [range(1, 31), range(1, 7), range(1, 51)],
        names=["participant_code", "action_id", "trial_in_cell"],
    )
    observed = pd.MultiIndex.from_frame(manifest[list(expected.names)])
    if len(expected.difference(observed)) or len(observed.difference(expected)):
        raise ValueError("Manifest does not map exactly to the 30 x 6 x 50 cached design")

    raw_length = np.empty((30, 6, 50), dtype=int)
    for row in manifest.itertuples(index=False):
        raw_length[row.participant_code - 1, row.action_id - 1, row.trial_in_cell - 1] = (
            row.raw_length_from_filename
        )
    keep = raw_length <= 200
    logger.info(
        "Manifest mapping verified. rawlen>200 exclusions=%d; retained=%d",
        int((~keep).sum()),
        int(keep.sum()),
    )
    return arrays, manifest, raw_length, keep


def waveform_trial_qc(
    arrays: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    out: Path,
    logger: logging.Logger,
) -> None:
    """Audit extreme cached curves without silently adding an exclusion rule."""
    p = manifest["participant_code"].to_numpy(dtype=int) - 1
    a = manifest["action_id"].to_numpy(dtype=int) - 1
    t = manifest["trial_in_cell"].to_numpy(dtype=int) - 1
    id_cols = [
        "global_index",
        "participant_code",
        "action_id",
        "action_label",
        "trial_in_cell",
        "raw_length_from_filename",
        "bat_file",
        "human_file",
    ]
    frames = []
    for waveform, full in arrays.items():
        curves = np.asarray(full[p, a, t, :])
        frame = manifest[id_cols].copy()
        frame.insert(0, "waveform", waveform)
        frame["curve_mean"] = curves.mean(axis=1)
        frame["curve_rms"] = np.sqrt((curves * curves).mean(axis=1))
        frame["curve_p99"] = np.percentile(curves, 99.0, axis=1)
        frame["curve_max"] = curves.max(axis=1)
        frames.append(frame)
    qc = pd.concat(frames, ignore_index=True)
    grouped = qc.groupby(["waveform", "action_id"])["curve_max"]
    medians = grouped.transform("median")
    mad = grouped.transform(lambda x: np.median(np.abs(x - np.median(x))))
    qc["action_median_curve_max"] = medians
    qc["action_mad_curve_max"] = mad
    qc["robust_z_curve_max"] = np.divide(
        qc["curve_max"] - medians,
        1.4826 * mad,
        out=np.full(len(qc), np.nan),
        where=mad.to_numpy() > 0,
    )
    qc["descending_rank_within_waveform_action"] = qc.groupby(
        ["waveform", "action_id"]
    )["curve_max"].rank(method="first", ascending=False).astype(int)
    extreme = qc[qc["descending_rank_within_waveform_action"] <= 20].sort_values(
        ["waveform", "action_id", "descending_rank_within_waveform_action"]
    )
    write_csv(qc, out / "Table_R13_WaveformTrialQC.csv")
    write_csv(extreme, out / "Table_R14_ExtremeTrialTop20PerAction.csv")

    top_body = qc[qc.waveform == "body_configuration"].nlargest(1, "curve_max").iloc[0]
    top_racket = qc[qc.waveform == "racket"].nlargest(1, "curve_max").iloc[0]
    note = f"""TTMD6 cached-waveform QC note
===============================

No trial is removed from the primary action-specific model on the basis of its
curve maximum.  The most extreme cached aggregated-human curve is
{top_body.human_file} (maximum {top_body.curve_max:.6g}); the most extreme
cached racket curve is {top_racket.bat_file} (maximum
{top_racket.curve_max:.6g}).  Tables R13--R14 provide a complete audit and are
not a post-hoc exclusion list.

The publication analysis handles exact [0,0,0] human-joint triplets upstream
as structural missingness.  A separately labelled local-Hampel transformation
is used only as a quality-control sensitivity analysis.  Consequently, all
results remain conditional on the archived coordinate representation and on
the explicitly selected upstream derivation; they are not physical-speed or
tracking-validity claims.
"""
    (out / "QC_NOTE.txt").write_text(note, encoding="utf-8")
    logger.info(
        "Trial-level QC audit complete (body max %.3f; racket max %.3f). See QC_NOTE.txt and Tables R13-R14.",
        top_body.curve_max,
        top_racket.curve_max,
    )


def profile_objective_from_sufficient(
    lam: float,
    n: np.ndarray,
    sums: np.ndarray,
    sums_sq: np.ndarray,
) -> tuple[float, float, float]:
    """Restricted profile objective, Q and GLS mean for one phase node."""
    den = 1.0 + n * lam
    a = np.sum(n / den)
    b = np.sum(sums / den)
    y_ri_y = np.sum(sums_sq - lam * sums * sums / den)
    q = y_ri_y - b * b / a
    q = max(float(q), np.finfo(float).tiny)
    df = int(n.sum()) - 1
    objective = df * math.log(q / df) + float(np.log1p(n * lam).sum()) + math.log(a)
    return objective, q, float(b / a)


def unbalanced_profile_reml(
    n: np.ndarray,
    sums: np.ndarray,
    sums_sq: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exact profile REML for a one-way unbalanced random-intercept model.

    R(lambda) is block diagonal with blocks I + lambda J.  Sherman-Morrison
    identities reduce the REML likelihood to group sizes, sums, and sums of
    squares, avoiding any dense N x N matrix.
    """
    n = np.asarray(n, dtype=float)
    sums = np.asarray(sums, dtype=float)
    sums_sq = np.asarray(sums_sq, dtype=float)
    if sums.shape != sums_sq.shape or sums.shape[0] != n.size:
        raise ValueError("Incompatible unbalanced sufficient statistics")
    n_phase = sums.shape[1]
    theta_grid = np.linspace(-20.0, 20.0, 81)
    between = np.empty(n_phase)
    within = np.empty(n_phase)
    lambdas = np.empty(n_phase)
    means = np.empty(n_phase)
    objective = np.empty(n_phase)
    objective_at_zero = np.empty(n_phase)
    interior = np.empty(n_phase, dtype=bool)
    optimizer_success = np.empty(n_phase, dtype=bool)
    upper_limit = np.empty(n_phase, dtype=bool)
    df = int(n.sum()) - 1

    for q in range(n_phase):
        s = sums[:, q]
        z = sums_sq[:, q]

        def f_theta(theta: float) -> float:
            return profile_objective_from_sufficient(math.exp(theta), n, s, z)[0]

        f0, q0, mu0 = profile_objective_from_sufficient(0.0, n, s, z)
        grid_values = np.array([f_theta(theta) for theta in theta_grid])
        k = int(np.argmin(grid_values))
        if k == 0:
            bounds = (-32.0, theta_grid[1])
        elif k == theta_grid.size - 1:
            bounds = (theta_grid[-2], 32.0)
        else:
            bounds = (theta_grid[k - 1], theta_grid[k + 1])
        result = minimize_scalar(
            f_theta,
            bounds=bounds,
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 300},
        )
        improvement_tol = 1e-11 * (1.0 + abs(f0))
        use_boundary = (not result.success) or f0 <= result.fun + improvement_tol
        if use_boundary:
            lam = 0.0
            obj, q_reml, mu = f0, q0, mu0
        else:
            lam = math.exp(float(result.x))
            obj, q_reml, mu = profile_objective_from_sufficient(lam, n, s, z)

        sigma2 = q_reml / df
        between[q] = lam * sigma2
        within[q] = sigma2
        lambdas[q] = lam
        means[q] = mu
        objective[q] = obj
        objective_at_zero[q] = f0
        interior[q] = lam > 0
        optimizer_success[q] = bool(result.success) or use_boundary
        upper_limit[q] = bool(lam > math.exp(31.5))

    # Populate ANOVA slots with NaN because they are undefined for unequal n_i.
    nan = np.full(n_phase, np.nan)
    return {
        "components": Components(between, within, nan, nan, nan, nan, interior),
        "lambda": lambdas,
        "fixed_intercept": means,
        "profile_objective": objective,
        "objective_at_lambda_zero": objective_at_zero,
        "optimizer_success": optimizer_success,
        "upper_limit_warning": upper_limit,
    }


def unbalanced_sufficient(data: np.ndarray, keep: np.ndarray):
    n_groups, _, n_phase = data.shape
    n = keep.sum(axis=1).astype(int)
    if np.any(n < 2):
        raise ValueError("Every athlete needs at least two retained trials")
    sums = np.empty((n_groups, n_phase))
    sums_sq = np.empty((n_groups, n_phase))
    for p in range(n_groups):
        x = np.asarray(data[p, keep[p], :])
        sums[p] = x.sum(axis=0)
        sums_sq[p] = (x * x).sum(axis=0)
    return n, sums, sums_sq


def bootstrap_draw_frame(
    waveform: str,
    action_id: int,
    comp: Components,
    weights: np.ndarray,
) -> pd.DataFrame:
    b_i = integrate(comp.between, weights)
    w_i = integrate(comp.within, weights)
    b = comp.between
    w = comp.within
    rows: dict[str, np.ndarray | str | int] = {
        "waveform": waveform,
        "action_id": action_id,
        "action_label_en": ACTION_EN[action_id],
        "action_label": ACTION_LABEL[action_id],
        "bootstrap_replicate": np.arange(1, b.shape[0] + 1),
        "integrated_sigma2_athlete": b_i,
        "integrated_sigma2_trial": w_i,
        "boundary_nodes": b.shape[1] - comp.interior.sum(axis=1),
        "boundary_fraction": 1.0 - comp.interior.mean(axis=1),
    }
    for m in M_GRID:
        rows[f"R_L2_m{m}"] = integrated_relative_reliability(b_i, w_i, m)
        rows[f"mean_pointwise_reliability_m{m}"] = integrate(reliability_curve(b, w, m), weights)
    for target in TARGETS:
        rows[f"required_n_R_L2_{int(target * 100)}"] = required_trials(b_i, w_i, target)
        rows[f"required_m_pointwise_G{int(target * 100)}"] = required_m_pointwise_batch(
            b, w, weights, target
        )
    return pd.DataFrame(rows)


def resample_draw_frame(
    waveform: str,
    action_id: int,
    comp: Components,
    weights: np.ndarray,
    m_min: int,
) -> pd.DataFrame:
    b_i = integrate(comp.between, weights)
    w_i = integrate(comp.within, weights)
    rows: dict[str, np.ndarray | str | int] = {
        "waveform": waveform,
        "action_id": action_id,
        "action_label_en": ACTION_EN[action_id],
        "action_label": ACTION_LABEL[action_id],
        "resample_replicate": np.arange(1, comp.between.shape[0] + 1),
        "balanced_m": m_min,
        "integrated_sigma2_athlete": b_i,
        "integrated_sigma2_trial": w_i,
        "boundary_nodes": comp.between.shape[1] - comp.interior.sum(axis=1),
        "boundary_fraction": 1.0 - comp.interior.mean(axis=1),
    }
    for m in M_GRID:
        rows[f"R_L2_m{m}"] = integrated_relative_reliability(b_i, w_i, m)
        rows[f"mean_pointwise_reliability_m{m}"] = integrate(reliability_curve(comp.between, comp.within, m), weights)
    for target in TARGETS:
        rows[f"required_n_R_L2_{int(target * 100)}"] = required_trials(b_i, w_i, target)
        rows[f"required_m_pointwise_G{int(target * 100)}"] = required_m_pointwise_batch(
            comp.between, comp.within, weights, target
        )
    return pd.DataFrame(rows)


def percentile_summary(
    draws: pd.DataFrame,
    point_lookup: dict[tuple[str, int, str], float],
    replicate_col: str,
) -> pd.DataFrame:
    id_cols = {
        "waveform",
        "action_id",
        "action_label_en",
        "action_label",
        replicate_col,
        "balanced_m",
    }
    metric_cols = [c for c in draws.columns if c not in id_cols]
    rows = []
    for (waveform, action_id), part in draws.groupby(["waveform", "action_id"], sort=True):
        for metric in metric_cols:
            values = part[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                q025, q50, q975 = np.percentile(finite, [2.5, 50.0, 97.5])
                mean = float(finite.mean())
                sd = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            else:
                q025 = q50 = q975 = mean = sd = math.inf
            rows.append(
                {
                    "waveform": waveform,
                    "action_id": int(action_id),
                    "action_label_en": ACTION_EN[int(action_id)],
                    "action_label": ACTION_LABEL[int(action_id)],
                    "metric": metric,
                    "reference_estimate": point_lookup.get((waveform, int(action_id), metric), np.nan),
                    "n_replicates": len(values),
                    "finite_replicates": len(finite),
                    "infinite_fraction": float(1.0 - len(finite) / len(values)),
                    "resampling_mean_finite": mean,
                    "resampling_sd_finite": sd,
                    "percentile_2_5_finite": float(q025),
                    "percentile_50_finite": float(q50),
                    "percentile_97_5_finite": float(q975),
                }
            )
    return pd.DataFrame(rows)


def point_lookup_from_summary(summary: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    id_cols = {
        "analysis",
        "waveform",
        "action_id",
        "action_label_en",
        "action_label",
        "n_athletes",
        "n_trials_total",
        "min_trials_per_athlete",
        "max_trials_per_athlete",
        "mean_trials_per_athlete",
    }
    lookup = {}
    for row in summary.to_dict("records"):
        for key, value in row.items():
            if key not in id_cols and isinstance(value, (int, float, np.integer, np.floating)):
                lookup[(row["waveform"], int(row["action_id"]), key)] = float(value)
    return lookup


def make_selection_masks(
    keep_action: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
) -> tuple[int, list[np.ndarray]]:
    n = keep_action.sum(axis=1).astype(int)
    m_min = int(n.min())
    masks: list[np.ndarray] = []
    for n_i in n:
        # Independent continuous random ranks give a uniform subset without replacement.
        keys = rng.random((n_replicates, int(n_i)))
        chosen = np.argpartition(keys, kth=m_min - 1, axis=1)[:, :m_min]
        selection = np.zeros((n_replicates, int(n_i)), dtype=float)
        selection[np.arange(n_replicates)[:, None], chosen] = 1.0
        masks.append(selection)
    return m_min, masks


def balanced_subsample_components(
    data: np.ndarray,
    keep_action: np.ndarray,
    selection_masks: list[np.ndarray],
    m_min: int,
) -> Components:
    n_groups, _, n_phase = data.shape
    n_replicates = selection_masks[0].shape[0]
    group_mean = np.empty((n_replicates, n_groups, n_phase))
    ss_within = np.zeros((n_replicates, n_phase))
    for p in range(n_groups):
        x = np.asarray(data[p, keep_action[p], :])
        augmented = np.concatenate([x, x * x], axis=1)
        selected = selection_masks[p] @ augmented
        sums = selected[:, :n_phase]
        sums_sq = selected[:, n_phase:]
        group_mean[:, p, :] = sums / m_min
        ss_within += np.maximum(sums_sq - sums * sums / m_min, 0.0)
    grand = group_mean.mean(axis=1)
    ss_between = m_min * ((group_mean - grand[:, None, :]) ** 2).sum(axis=1)
    return components_from_sums_of_squares(ss_between, ss_within, n_groups, m_min)


def dense_profile_objective(lam: float, groups: list[np.ndarray]) -> float:
    """Small dense reference implementation used only by the self-test."""
    y = np.concatenate(groups)
    n_total = y.size
    r = np.eye(n_total)
    start = 0
    for group in groups:
        stop = start + group.size
        r[start:stop, start:stop] += lam
        start = stop
    one = np.ones(n_total)
    r_inv_y = np.linalg.solve(r, y)
    r_inv_one = np.linalg.solve(r, one)
    a = one @ r_inv_one
    q = y @ r_inv_y - (one @ r_inv_y) ** 2 / a
    sign, logdet = np.linalg.slogdet(r)
    if sign <= 0:
        raise AssertionError("Dense test covariance was not positive definite")
    return (n_total - 1) * math.log(q / (n_total - 1)) + logdet + math.log(a)


def self_tests(
    arrays: dict[str, np.ndarray],
    weights: np.ndarray,
    run_actual_profile: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict] = []

    # Sufficient-statistic likelihood must exactly match a dense calculation.
    groups = [np.array([1.0, 1.6, 0.8]), np.array([2.2, 2.8, 1.9, 2.5]), np.array([0.1, 0.4])]
    n = np.array([len(x) for x in groups], dtype=float)
    sums = np.array([x.sum() for x in groups])
    sums_sq = np.array([(x * x).sum() for x in groups])
    differences = []
    for lam in (0.0, 0.1, 1.0, 10.0):
        sufficient = profile_objective_from_sufficient(lam, n, sums, sums_sq)[0]
        dense = dense_profile_objective(lam, groups)
        differences.append(abs(sufficient - dense))
    dense_error = max(differences)
    rows.append(
        {
            "test": "unbalanced_sufficient_objective_equals_dense_REML",
            "waveform": "synthetic",
            "action_id": np.nan,
            "max_abs_error": dense_error,
            "max_relative_error": dense_error / 10.0,
            "tolerance": 1e-10,
            "passed": dense_error < 1e-10,
        }
    )

    if run_actual_profile:
        # Requested quick check: all 50 trials and every phase node, for all 12 models.
        for waveform, full in arrays.items():
            for action_id in range(1, 7):
                data = np.asarray(full[:, action_id - 1, :, :])
                analytic, _, _ = balanced_reml(data)
                n_full = np.full(data.shape[0], data.shape[1], dtype=int)
                sums_full = data.sum(axis=1)
                sums_sq_full = (data * data).sum(axis=1)
                profile = unbalanced_profile_reml(n_full, sums_full, sums_sq_full)["components"]
                interior = analytic.interior
                rel_b = np.max(
                    np.abs(profile.between[interior] - analytic.between[interior])
                    / np.maximum(np.abs(analytic.between[interior]), 1e-10)
                )
                rel_w = np.max(
                    np.abs(profile.within - analytic.within)
                    / np.maximum(np.abs(analytic.within), 1e-10)
                )
                max_rel = float(max(rel_b, rel_w))
                max_abs = float(
                    max(
                        np.max(np.abs(profile.between - analytic.between)),
                        np.max(np.abs(profile.within - analytic.within)),
                    )
                )
                rows.append(
                    {
                        "test": "actual_30x50_profile_REML_equals_balanced_ANOVA_REML",
                        "waveform": waveform,
                        "action_id": action_id,
                        "max_abs_error": max_abs,
                        "max_relative_error": max_rel,
                        "tolerance": 2e-5,
                        "passed": max_rel < 2e-5,
                        "interior_nodes": int(interior.sum()),
                        "phase_nodes": len(interior),
                    }
                )

    tests = pd.DataFrame(rows)
    if not tests["passed"].all():
        raise AssertionError(f"REML self-test failure:\n{tests.loc[~tests.passed]}")
    logger.info("REML self-tests passed (%d checks)", len(tests))
    return tests


def main() -> None:
    args = parse_args()
    start = time.time()
    out = args.out.resolve()
    logger = setup_logging(out)
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Seed=%d; athlete bootstrap=%d; balanced subsamples=%d", args.seed, args.n_bootstrap, args.n_balanced_resamples)
    if args.n_bootstrap < 1 or args.n_balanced_resamples < 1:
        raise ValueError("Resampling counts must be positive")

    arrays, manifest, raw_length, keep = validate_and_load(
        args.arrays_dir.resolve(), args.manifest.resolve(), logger
    )
    waveform_trial_qc(arrays, manifest, out, logger)
    weights = integration_weights(next(iter(arrays.values())).shape[-1])
    tests = self_tests(arrays, weights, not args.skip_numerical_self_test, logger)
    write_csv(tests, out / "Table_R12_SelfTests.csv")

    rng_boot = np.random.default_rng(args.seed)
    bootstrap_counts = rng_boot.multinomial(
        30, np.full(30, 1.0 / 30.0), size=args.n_bootstrap
    )

    full_pointwise: list[pd.DataFrame] = []
    full_integrated: list[dict] = []
    full_dstudy: list[dict] = []
    bootstrap_draws: list[pd.DataFrame] = []
    full_components: dict[tuple[str, int], Components] = {}

    logger.info("Running full balanced action-specific REML and whole-athlete bootstrap")
    for waveform, full in arrays.items():
        for action_id in range(1, 7):
            data = np.asarray(full[:, action_id - 1, :, :])
            comp, group_mean, within_by_group = balanced_reml(data)
            full_components[(waveform, action_id)] = comp
            full_pointwise.append(
                pointwise_rows(
                    waveform, action_id, comp, 30, 1500, 50, 50, "full_balanced_constrained_REML"
                )
            )
            row = {
                "analysis": "full_balanced_constrained_REML",
                "waveform": waveform,
                "action_id": action_id,
                "action_label_en": ACTION_EN[action_id],
                "action_label": ACTION_LABEL[action_id],
                "n_athletes": 30,
                "n_trials_total": 1500,
                "min_trials_per_athlete": 50,
                "max_trials_per_athlete": 50,
                "mean_trials_per_athlete": 50.0,
                **component_summary(comp, weights),
            }
            full_integrated.append(row)
            full_dstudy.extend(
                dstudy_rows(waveform, action_id, comp, weights, "full_balanced_constrained_REML")
            )
            boot_comp = bootstrap_balanced_from_counts(
                group_mean, within_by_group, 50, bootstrap_counts
            )
            bootstrap_draws.append(
                bootstrap_draw_frame(waveform, action_id, boot_comp, weights)
            )
            logger.info(
                "Full/boot complete: %s action %d (%s), R_L2(50)=%.4f",
                waveform,
                action_id,
                ACTION_LABEL[action_id],
                row["R_L2_m50"],
            )

    full_pointwise_df = pd.concat(full_pointwise, ignore_index=True)
    full_integrated_df = pd.DataFrame(full_integrated)
    full_dstudy_df = pd.DataFrame(full_dstudy)
    boot_draws_df = pd.concat(bootstrap_draws, ignore_index=True)
    boot_summary_df = percentile_summary(
        boot_draws_df,
        point_lookup_from_summary(full_integrated_df),
        "bootstrap_replicate",
    )
    write_csv(full_pointwise_df, out / "Table_R1_FullBalanced_Pointwise.csv")
    write_csv(full_integrated_df, out / "Table_R2_FullBalanced_Integrated.csv")
    write_csv(full_dstudy_df, out / "Table_R3_FullBalanced_DStudy.csv")
    write_csv(boot_draws_df, out / "Table_R4_ClusterBootstrap5000_Draws.csv")
    write_csv(boot_summary_df, out / "Table_R5_ClusterBootstrap5000_Percentiles.csv")

    exclusion_rows = []
    for action_id in range(1, 7):
        n_by_athlete = keep[:, action_id - 1, :].sum(axis=1)
        exclusion_rows.append(
            {
                "action_id": action_id,
                "action_label_en": ACTION_EN[action_id],
                "action_label": ACTION_LABEL[action_id],
                "original_trials": 1500,
                "excluded_rawlen_gt_200": int((~keep[:, action_id - 1, :]).sum()),
                "retained_trials": int(n_by_athlete.sum()),
                "retained_percent": float(100.0 * n_by_athlete.sum() / 1500),
                "min_trials_per_athlete": int(n_by_athlete.min()),
                "median_trials_per_athlete": float(np.median(n_by_athlete)),
                "max_trials_per_athlete": int(n_by_athlete.max()),
                "mean_trials_per_athlete": float(n_by_athlete.mean()),
            }
        )
    exclusion_df = pd.DataFrame(exclusion_rows)
    write_csv(exclusion_df, out / "Table_R6_ExclusionCounts.csv")

    logger.info("Running rawlen<=200 exact unbalanced profile REML")
    unbalanced_pointwise: list[pd.DataFrame] = []
    unbalanced_integrated: list[dict] = []
    unbalanced_dstudy: list[dict] = []
    unbalanced_components: dict[tuple[str, int], Components] = {}
    for waveform, full in arrays.items():
        for action_id in range(1, 7):
            data = np.asarray(full[:, action_id - 1, :, :])
            action_keep = keep[:, action_id - 1, :]
            n, sums, sums_sq = unbalanced_sufficient(data, action_keep)
            fit = unbalanced_profile_reml(n, sums, sums_sq)
            comp = fit["components"]
            unbalanced_components[(waveform, action_id)] = comp
            point = pointwise_rows(
                waveform,
                action_id,
                comp,
                30,
                int(n.sum()),
                int(n.min()),
                int(n.max()),
                "rawlen_le_200_unbalanced_profile_REML",
            )
            point["variance_ratio_lambda"] = fit["lambda"]
            point["fixed_intercept"] = fit["fixed_intercept"]
            point["profile_objective"] = fit["profile_objective"]
            point["objective_at_lambda_zero"] = fit["objective_at_lambda_zero"]
            point["optimizer_success"] = fit["optimizer_success"]
            point["upper_limit_warning"] = fit["upper_limit_warning"]
            unbalanced_pointwise.append(point)
            row = {
                "analysis": "rawlen_le_200_unbalanced_profile_REML",
                "waveform": waveform,
                "action_id": action_id,
                "action_label_en": ACTION_EN[action_id],
                "action_label": ACTION_LABEL[action_id],
                "n_athletes": 30,
                "n_trials_total": int(n.sum()),
                "min_trials_per_athlete": int(n.min()),
                "max_trials_per_athlete": int(n.max()),
                "mean_trials_per_athlete": float(n.mean()),
                "optimizer_success_nodes": int(np.sum(fit["optimizer_success"])),
                "upper_limit_warning_nodes": int(np.sum(fit["upper_limit_warning"])),
                **component_summary(comp, weights),
            }
            unbalanced_integrated.append(row)
            unbalanced_dstudy.extend(
                dstudy_rows(
                    waveform,
                    action_id,
                    comp,
                    weights,
                    "rawlen_le_200_unbalanced_profile_REML",
                )
            )
            logger.info(
                "Unbalanced complete: %s action %d; retained=%d; R_L2(50)=%.4f; boundary nodes=%d",
                waveform,
                action_id,
                int(n.sum()),
                row["R_L2_m50"],
                row["boundary_nodes"],
            )

    unbalanced_pointwise_df = pd.concat(unbalanced_pointwise, ignore_index=True)
    unbalanced_integrated_df = pd.DataFrame(unbalanced_integrated)
    unbalanced_dstudy_df = pd.DataFrame(unbalanced_dstudy)
    write_csv(unbalanced_pointwise_df, out / "Table_R7_Excluded_UnbalancedREML_Pointwise.csv")
    write_csv(unbalanced_integrated_df, out / "Table_R8_Excluded_UnbalancedREML_Integrated.csv")
    write_csv(unbalanced_dstudy_df, out / "Table_R9_Excluded_UnbalancedREML_DStudy.csv")

    logger.info("Running within-cell balanced subsampling after exclusion")
    rng_subsample = np.random.default_rng(args.seed + 1)
    subsample_draws: list[pd.DataFrame] = []
    selection_audit = []
    for action_id in range(1, 7):
        action_keep = keep[:, action_id - 1, :]
        m_min, selection_masks = make_selection_masks(
            action_keep, args.n_balanced_resamples, rng_subsample
        )
        for p, n_i in enumerate(action_keep.sum(axis=1), start=1):
            selection_audit.append(
                {
                    "action_id": action_id,
                    "action_label": ACTION_LABEL[action_id],
                    "participant_code": p,
                    "available_trials": int(n_i),
                    "balanced_m": m_min,
                    "sampling": "without_replacement",
                }
            )
        for waveform, full in arrays.items():
            data = np.asarray(full[:, action_id - 1, :, :])
            comp = balanced_subsample_components(
                data, action_keep, selection_masks, m_min
            )
            subsample_draws.append(
                resample_draw_frame(waveform, action_id, comp, weights, m_min)
            )
            logger.info(
                "Balanced subsampling complete: %s action %d, m=%d, replicates=%d",
                waveform,
                action_id,
                m_min,
                args.n_balanced_resamples,
            )
        del selection_masks

    subsample_draws_df = pd.concat(subsample_draws, ignore_index=True)
    subsample_summary_df = percentile_summary(
        subsample_draws_df,
        point_lookup_from_summary(unbalanced_integrated_df),
        "resample_replicate",
    )
    write_csv(subsample_draws_df, out / "Table_R10_BalancedSubsample1000_Draws.csv")
    write_csv(subsample_summary_df, out / "Table_R11_BalancedSubsample1000_Percentiles.csv")
    write_csv(pd.DataFrame(selection_audit), out / "Table_R11A_BalancedSubsample_CellAudit.csv")

    provenance_inputs = [args.manifest.resolve()] + [
        (args.arrays_dir / filename).resolve() for filename in ARRAY_FILES.values()
    ]
    provenance = pd.DataFrame(
        [
            {
                "role": "input",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in provenance_inputs
        ]
    )
    write_csv(provenance, out / "File_Provenance.csv")
    shutil.copy2(Path(__file__).resolve(), out / "run_action_specific_reml.py")

    runtime = time.time() - start
    key_results = full_integrated_df[
        [
            "waveform",
            "action_id",
            "action_label",
            "R_L2_m50",
            "required_n_R_L2_80",
            "required_n_R_L2_90",
        ]
    ].to_dict("records")
    summary = {
        "status": "complete",
        "seed": args.seed,
        "n_bootstrap_whole_athlete": args.n_bootstrap,
        "n_balanced_subsamples_without_replacement": args.n_balanced_resamples,
        "excluded_rawlen_gt_200": int((~keep).sum()),
        "retained_rawlen_le_200": int(keep.sum()),
        "action_specific_minimum_m_after_exclusion": {
            str(a): int(keep[:, a - 1, :].sum(axis=1).min()) for a in range(1, 7)
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "runtime_seconds": round(runtime, 3),
        "self_tests_all_passed": bool(tests.passed.all()),
        "key_full_balanced_results": key_results,
    }
    with (out / "RUN_SUMMARY.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    logger.info("Complete in %.2f seconds. Outputs: %s", runtime, out)
    for handler in logger.handlers:
        handler.flush()
    outputs = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "File_Inventory.csv":
            outputs.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_csv(pd.DataFrame(outputs), out / "File_Inventory.csv")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
