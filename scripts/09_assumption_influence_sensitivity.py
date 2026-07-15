#!/usr/bin/env python3
"""Assumption, registration, influence, and bootstrap diagnostics for TTMD6.

This script addresses four inferential risks that are not resolved by the
action-specific variance-component analysis alone:

1. the archive's within-cell numeric order may retain serial dependence;
2. W/n is therefore evaluated under transparent AR(1) correlation scenarios;
3. unregistered normalized waveforms mix amplitude and landmark timing;
4. athlete influence and whole-athlete bootstrap behavior require direct
   diagnostics, especially for the backhand-drive-labelled racket waveform.

The numeric file order is treated only as *archive order*.  The source article
states that 55 consecutive strokes were stored together and manually split,
but it does not document that the 50 released files preserve acquisition order
or explain which five strokes were omitted.  Archive-order autocorrelation is
therefore diagnostic rather than proof of chronological serial dependence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde, norm


ACTION_EN = {
    1: "forehand_attack",
    2: "forehand_drive",
    3: "forehand_push",
    4: "backhand_attack",
    5: "backhand_drive",
    6: "backhand_push",
}
ACTION_CODE = {1: "FA", 2: "FD", 3: "FP", 4: "BA", 5: "BD", 6: "BP"}
WAVEFORM_LABEL = {"racket": "racket", "body_configuration": "body"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--main-arrays-dir", required=True, type=Path)
    p.add_argument("--hampel-arrays-dir", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--main-summary", required=True, type=Path)
    p.add_argument("--bootstrap-draws", required=True, type=Path)
    p.add_argument("--hampel-audit", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--bootstrap-seed", type=int, default=20260712)
    p.add_argument("--n-permutations", type=int, default=5000)
    p.add_argument("--acf-max-lag", type=int, default=10)
    p.add_argument("--peak-smooth-window", type=int, default=11)
    p.add_argument("--peak-search-low-pct", type=float, default=10.0)
    p.add_argument("--peak-search-high-pct", type=float, default=90.0)
    return p.parse_args()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def integration_weights(q: int) -> np.ndarray:
    if q < 2:
        raise ValueError("At least two time nodes are required")
    w = np.ones(q, dtype=float) / (q - 1)
    w[[0, -1]] *= 0.5
    return w


def integrate_last(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(x) @ w


def balanced_components(data: np.ndarray, w: np.ndarray) -> dict[str, float | np.ndarray]:
    """Non-negative balanced one-way REML/ANOVA components by time node."""
    if data.ndim != 3:
        raise ValueError(f"Expected athlete x trial x time, observed {data.shape}")
    p, m, _ = data.shape
    group_mean = data.mean(axis=1)
    grand = group_mean.mean(axis=0)
    ss_between = m * ((group_mean - grand) ** 2).sum(axis=0)
    ss_within = ((data - group_mean[:, None, :]) ** 2).sum(axis=(0, 1))
    ms_between = ss_between / (p - 1)
    ms_within = ss_within / (p * (m - 1))
    interior = ms_between >= ms_within
    between = np.where(interior, (ms_between - ms_within) / m, 0.0)
    pooled = (ss_between + ss_within) / (p * m - 1)
    within = np.where(interior, ms_within, pooled)
    b = float(integrate_last(between, w))
    e = float(integrate_last(within, w))
    ratio = e / b if b > 0 else math.inf
    nstar90 = 9.0 * ratio
    return {
        "between": between,
        "within": within,
        "B": b,
        "W": e,
        "W_over_B": ratio,
        "continuous_n90": nstar90,
        "integer_n90": float(max(1, math.ceil(nstar90))) if np.isfinite(nstar90) else math.inf,
        "boundary_nodes": int(np.size(interior) - np.count_nonzero(interior)),
    }


def trace_lag_correlation(residual: np.ndarray, w: np.ndarray, lag: int) -> float:
    left = residual[:, :-lag, :]
    right = residual[:, lag:, :]
    numerator = float(np.sum(integrate_last(left * right, w)))
    denom = math.sqrt(
        float(np.sum(integrate_last(left * left, w)))
        * float(np.sum(integrate_last(right * right, w)))
    )
    return numerator / denom if denom > 0 else math.nan


def lag1_permutation_test(
    residual: np.ndarray,
    w: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> dict[str, float]:
    """Within-athlete order randomization using precomputed L2 Gram matrices."""
    p, m, _ = residual.shape
    gram = np.einsum("ptu,u,psu->pts", residual, w, residual, optimize=True)
    observed = trace_lag_correlation(residual, w, 1)
    null = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        numerator = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for athlete in range(p):
            order = rng.permutation(m)
            numerator += float(gram[athlete, order[:-1], order[1:]].sum())
            diag = np.diag(gram[athlete])
            left_norm += float(diag[order[:-1]].sum())
            right_norm += float(diag[order[1:]].sum())
        null[b] = numerator / math.sqrt(left_norm * right_norm)
    return {
        "rho_lag1": observed,
        "permutation_null_p025": float(np.quantile(null, 0.025)),
        "permutation_null_p975": float(np.quantile(null, 0.975)),
        "permutation_two_sided_p": float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1)),
    }


def ar1_design_effect(n: int, phi: float) -> float:
    if n <= 1:
        return 1.0
    k = np.arange(1, n, dtype=float)
    return float(1.0 + 2.0 * np.sum((1.0 - k / n) * phi**k))


def ar1_reliability(B: float, W: float, n: int, phi: float) -> float:
    error = W * ar1_design_effect(n, phi) / n
    return B / (B + error) if B + error > 0 else 0.0


def required_n_ar1(B: float, W: float, phi: float, target: float = 0.90, max_n: int = 5000) -> float:
    for n in range(1, max_n + 1):
        if ar1_reliability(B, W, n, phi) >= target:
            return float(n)
    return math.inf


def landmark_register(
    landmark_racket: np.ndarray,
    racket: np.ndarray,
    body: np.ndarray,
    smooth_window: int,
    low_pct: float,
    high_pct: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """One-landmark piecewise-linear registration using the central racket peak.

    The peak is located on an 11-node moving-average display copy but the warp
    is applied to the unsmoothed racket and body curves.  The same warp is used
    for each paired racket/body trial.  Endpoints remain fixed at 0 and 1.
    """
    if smooth_window < 3 or smooth_window % 2 == 0:
        raise ValueError("The peak smoothing window must be an odd integer >=3")
    p, a, m, q = racket.shape
    grid = np.linspace(0.0, 1.0, q)
    lo = int(math.ceil(low_pct / 100.0 * (q - 1)))
    hi = int(math.floor(high_pct / 100.0 * (q - 1))) + 1
    if not (0 < lo < hi < q):
        raise ValueError("Peak search interval must lie strictly inside (0, 100)")
    if landmark_racket.shape != racket.shape:
        raise ValueError("Landmark and analysis racket arrays must have identical shape")
    # A quality-controlled copy is used only to locate a robust central peak;
    # the warp is applied to the unaltered primary racket/body curves.  This
    # prevents a single retained high value from defining the landmark while
    # keeping the registration sensitivity distinct from the QC sensitivity.
    smooth = uniform_filter1d(landmark_racket, size=smooth_window, axis=-1, mode="nearest")
    peak = lo + np.argmax(smooth[..., lo:hi], axis=-1)
    out_r = np.empty_like(racket)
    out_b = np.empty_like(body)
    rows: list[dict] = []
    for action in range(a):
        reference_node = int(np.rint(np.median(peak[:, action, :])))
        reference_u = grid[reference_node]
        for athlete in range(p):
            for trial in range(m):
                observed_node = int(peak[athlete, action, trial])
                observed_u = grid[observed_node]
                source_u = np.where(
                    grid <= reference_u,
                    grid * observed_u / reference_u,
                    observed_u + (grid - reference_u) * (1.0 - observed_u) / (1.0 - reference_u),
                )
                out_r[athlete, action, trial] = np.interp(
                    source_u, grid, racket[athlete, action, trial]
                )
                out_b[athlete, action, trial] = np.interp(
                    source_u, grid, body[athlete, action, trial]
                )
                rows.append(
                    {
                        "participant_code": athlete + 1,
                        "action_id": action + 1,
                        "action_code": ACTION_CODE[action + 1],
                        "trial_in_cell": trial + 1,
                        "racket_peak_node_0based": observed_node,
                        "racket_peak_time_pct": 100.0 * grid[observed_node],
                        "reference_peak_node_0based": reference_node,
                        "reference_peak_time_pct": 100.0 * reference_u,
                    }
                )
    return out_r, out_b, pd.DataFrame(rows)


def bootstrap_quantile_se(x: np.ndarray, p: float) -> float:
    """Approximate Monte Carlo SE of a simulated quantile using KDE density."""
    x = np.asarray(x, float)
    q = float(np.quantile(x, p))
    density = float(gaussian_kde(x)([q])[0])
    if density <= 0 or not np.isfinite(density):
        return math.nan
    return float(math.sqrt(p * (1.0 - p) / x.size) / density)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    main_arrays = {
        "racket": np.load(args.main_arrays_dir / "racket_speed_verified9000.npy"),
        "body_configuration": np.load(args.main_arrays_dir / "body_configuration_speed_verified9000.npy"),
    }
    hampel_arrays = {
        "racket": np.load(args.hampel_arrays_dir / "racket_speed_verified9000.npy"),
        "body_configuration": np.load(args.hampel_arrays_dir / "body_configuration_speed_verified9000.npy"),
    }
    if any(x.shape != (30, 6, 50, 200) for x in [*main_arrays.values(), *hampel_arrays.values()]):
        raise ValueError("Expected four 30 x 6 x 50 x 200 arrays")
    w = integration_weights(200)

    # Verify that the numeric archive order is complete and contiguous, while
    # preserving the distinction between archive order and acquisition order.
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["primary_cohort"].astype(bool)].copy()
    order_rows = []
    for (participant, action), g in manifest.groupby(["participant_code", "action_id"]):
        g = g.sort_values("trial_in_cell")
        idx = g["global_index"].to_numpy(int)
        trials = g["trial_in_cell"].to_numpy(int)
        order_rows.append(
            {
                "participant_code": int(participant),
                "action_id": int(action),
                "action_code": ACTION_CODE[int(action)],
                "files_in_cell": int(len(g)),
                "trial_labels_1_to_50": bool(np.array_equal(trials, np.arange(1, 51))),
                "global_indices_contiguous": bool(np.array_equal(np.diff(idx), np.ones(49, int))),
                "first_global_index": int(idx.min()),
                "last_global_index": int(idx.max()),
                "chronological_order_verified": False,
            }
        )
    order_df = pd.DataFrame(order_rows)
    if len(order_df) != 180 or not order_df[["trial_labels_1_to_50", "global_indices_contiguous"]].all().all():
        raise RuntimeError("Archive-order assumptions failed")
    write_csv(order_df, args.out / "Table_S_ArchiveOrderAudit.csv")

    # Archive-order trace autocorrelation and within-athlete order permutation.
    acf_rows: list[dict] = []
    lag1_lookup: dict[tuple[str, int], float] = {}
    for waveform, full in main_arrays.items():
        for action_id in range(1, 7):
            data = full[:, action_id - 1, :, :]
            residual = data - data.mean(axis=1, keepdims=True)
            perm = lag1_permutation_test(residual, w, rng, args.n_permutations)
            lag1_lookup[(waveform, action_id)] = perm["rho_lag1"]
            for lag in range(1, args.acf_max_lag + 1):
                acf_rows.append(
                    {
                        "waveform": waveform,
                        "action_id": action_id,
                        "action_code": ACTION_CODE[action_id],
                        "action_label_en": ACTION_EN[action_id],
                        "archive_order_lag": lag,
                        "rho_L2_trace": trace_lag_correlation(residual, w, lag),
                        "lag1_permutation_replicates": args.n_permutations if lag == 1 else np.nan,
                        "lag1_permutation_null_p025": perm["permutation_null_p025"] if lag == 1 else np.nan,
                        "lag1_permutation_null_p975": perm["permutation_null_p975"] if lag == 1 else np.nan,
                        "lag1_permutation_two_sided_p": perm["permutation_two_sided_p"] if lag == 1 else np.nan,
                        "order_interpretation": "numeric_archive_order_not_verified_acquisition_chronology",
                    }
                )
    acf_df = pd.DataFrame(acf_rows)
    write_csv(acf_df, args.out / "Table_S_ArchiveOrderResidualACF.csv")

    # AR(1) working scenarios.  The empirical lag-1 scenario is explicitly
    # labeled diagnostic because chronology is not independently verified.
    summary = pd.read_csv(args.main_summary)
    ar_rows = []
    for row in summary.itertuples(index=False):
        B = float(row.integrated_sigma2_athlete)
        W = float(row.integrated_sigma2_trial)
        empirical = lag1_lookup[(row.waveform, int(row.action_id))]
        scenarios = [("fixed", 0.0), ("fixed", 0.10), ("fixed", 0.20), ("fixed", 0.30), ("archive_order_lag1", empirical)]
        for scenario, phi in scenarios:
            n90 = required_n_ar1(B, W, phi)
            ar_rows.append(
                {
                    "waveform": row.waveform,
                    "action_id": int(row.action_id),
                    "action_code": ACTION_CODE[int(row.action_id)],
                    "scenario": scenario,
                    "AR1_phi": phi,
                    "B": B,
                    "W": W,
                    "W_over_B": W / B,
                    "independence_continuous_n90": 9.0 * W / B,
                    "minimum_integer_n90": n90,
                    "design_effect_at_n90": ar1_design_effect(int(n90), phi) if np.isfinite(n90) else math.nan,
                    "reliability_at_50": ar1_reliability(B, W, 50, phi),
                    "chronology_verified": False if scenario == "archive_order_lag1" else np.nan,
                }
            )
    ar_df = pd.DataFrame(ar_rows)
    write_csv(ar_df, args.out / "Table_S_AR1CorrelationScenarios.csv")

    # One-landmark registration sensitivity using the paired racket peak.
    registered_racket, registered_body, peak_df = landmark_register(
        hampel_arrays["racket"],
        main_arrays["racket"],
        main_arrays["body_configuration"],
        args.peak_smooth_window,
        args.peak_search_low_pct,
        args.peak_search_high_pct,
    )
    write_csv(peak_df, args.out / "Table_S_PeakRegistrationLandmarks.csv")
    registration_rows = []
    for waveform, full, registered in [
        ("racket", main_arrays["racket"], registered_racket),
        ("body_configuration", main_arrays["body_configuration"], registered_body),
    ]:
        for action_id in range(1, 7):
            for rule, data in [
                ("unregistered", full[:, action_id - 1]),
                ("robust_racket_peak_registered", registered[:, action_id - 1]),
            ]:
                fit = balanced_components(data, w)
                peaks = peak_df.loc[peak_df.action_id == action_id, "racket_peak_time_pct"]
                registration_rows.append(
                    {
                        "waveform": waveform,
                        "action_id": action_id,
                        "action_code": ACTION_CODE[action_id],
                        "registration_rule": rule,
                        "B": fit["B"],
                        "W": fit["W"],
                        "W_over_B": fit["W_over_B"],
                        "continuous_n90": fit["continuous_n90"],
                        "integer_n90": fit["integer_n90"],
                        "boundary_nodes": fit["boundary_nodes"],
                        "racket_peak_median_pct": float(peaks.median()),
                        "racket_peak_p25_pct": float(peaks.quantile(0.25)),
                        "racket_peak_p75_pct": float(peaks.quantile(0.75)),
                        "peak_search_interval_pct": f"{args.peak_search_low_pct:g}-{args.peak_search_high_pct:g}",
                        "peak_smoothing_nodes_for_landmark_only": args.peak_smooth_window,
                        "landmark_source": "Hampel-type QC racket copy; warp applied to unaltered paired waveforms",
                    }
                )
    registration_df = pd.DataFrame(registration_rows)
    write_csv(registration_df, args.out / "Table_S_PeakRegistrationSensitivity.csv")

    # Leave-one-athlete-out fits for both transparent coordinate rules.
    loo_rows = []
    contribution_rows = []
    for rule, arrays in [("unaltered_nonzero_primary", main_arrays), ("hampel_type_local_high_value", hampel_arrays)]:
        for waveform, full in arrays.items():
            for action_id in range(1, 7):
                data = full[:, action_id - 1]
                group_mean = data.mean(axis=1)
                grand = group_mean.mean(axis=0)
                within_ss = integrate_last(((data - group_mean[:, None]) ** 2).sum(axis=1), w)
                between_distance = integrate_last((group_mean - grand) ** 2, w)
                for athlete in range(30):
                    fit = balanced_components(np.delete(data, athlete, axis=0), w)
                    loo_rows.append(
                        {
                            "quality_rule": rule,
                            "waveform": waveform,
                            "action_id": action_id,
                            "action_code": ACTION_CODE[action_id],
                            "excluded_participant_code": athlete + 1,
                            "B": fit["B"],
                            "W": fit["W"],
                            "W_over_B": fit["W_over_B"],
                            "continuous_n90": fit["continuous_n90"],
                            "integer_n90": fit["integer_n90"],
                            "boundary_nodes": fit["boundary_nodes"],
                        }
                    )
                    contribution_rows.append(
                        {
                            "quality_rule": rule,
                            "waveform": waveform,
                            "action_id": action_id,
                            "action_code": ACTION_CODE[action_id],
                            "participant_code": athlete + 1,
                            "integrated_within_SS": float(within_ss[athlete]),
                            "within_SS_share": float(within_ss[athlete] / within_ss.sum()),
                            "integrated_squared_group_mean_deviation": float(between_distance[athlete]),
                            "between_distance_share": float(between_distance[athlete] / between_distance.sum()),
                        }
                    )
    loo_df = pd.DataFrame(loo_rows)
    contribution_df = pd.DataFrame(contribution_rows)
    write_csv(loo_df, args.out / "Table_S_LeaveOnePlayerOut.csv")
    write_csv(contribution_df, args.out / "Table_S_PlayerVarianceContributions.csv")

    # Distribution of Hampel-type flags by player for the key BD racket cell.
    hampel_audit = pd.read_csv(args.hampel_audit)
    bd = hampel_audit[hampel_audit.action_id == 5].copy()
    bd_player = bd.groupby("participant_code").agg(
        trials_flagged=("racket_hampel_high_spikes", lambda x: int((x > 0).sum())),
        displacement_values_flagged=("racket_hampel_high_spikes", "sum"),
        maximum_flags_in_one_trial=("racket_hampel_high_spikes", "max"),
    ).reset_index()
    bd_player["trial_count"] = 50
    bd_player["flagged_trial_percent"] = 100.0 * bd_player.trials_flagged / 50.0
    key_loo = loo_df[(loo_df.waveform == "racket") & (loo_df.action_id == 5)].pivot(
        index="excluded_participant_code", columns="quality_rule", values=["continuous_n90", "integer_n90", "B", "W"]
    )
    key_loo.columns = [f"{metric}_{rule}" for metric, rule in key_loo.columns]
    key_loo = key_loo.reset_index().rename(columns={"excluded_participant_code": "participant_code"})
    bd_player = bd_player.merge(key_loo, on="participant_code", how="left")
    write_csv(bd_player, args.out / "Table_S_BackhandDriveInfluenceAndFlags.csv")

    # Bootstrap diagnostics for continuous n90 and paired action contrasts.
    boot = pd.read_csv(args.bootstrap_draws)
    boot["continuous_n90"] = 9.0 * boot.integrated_sigma2_trial / boot.integrated_sigma2_athlete
    point = summary.set_index(["waveform", "action_id"])
    diagnostic_rows = []
    for key, g in boot.groupby(["waveform", "action_id"]):
        x = g.continuous_n90.to_numpy(float)
        theta = 9.0 * float(point.loc[key, "integrated_sigma2_trial"]) / float(point.loc[key, "integrated_sigma2_athlete"])
        q025, q975 = np.quantile(x, [0.025, 0.975])
        jack = loo_df[
            (loo_df.quality_rule == "unaltered_nonzero_primary")
            & (loo_df.waveform == key[0])
            & (loo_df.action_id == key[1])
        ].sort_values("excluded_participant_code").continuous_n90.to_numpy(float)
        jack_delta = jack.mean() - jack
        acceleration_den = 6.0 * np.sum(jack_delta**2) ** 1.5
        acceleration = float(np.sum(jack_delta**3) / acceleration_den) if acceleration_den > 0 else 0.0
        less = (np.sum(x < theta) + 0.5 * np.sum(x == theta)) / x.size
        less = float(np.clip(less, 0.5 / x.size, 1.0 - 0.5 / x.size))
        bias_correction = float(norm.ppf(less))
        adjusted_probabilities = []
        for alpha in (0.025, 0.975):
            z_alpha = float(norm.ppf(alpha))
            denominator = 1.0 - acceleration * (bias_correction + z_alpha)
            adjusted = norm.cdf(
                bias_correction
                + (bias_correction + z_alpha) / denominator
            )
            adjusted_probabilities.append(float(np.clip(adjusted, 0.0, 1.0)))
        bca_low, bca_high = np.quantile(x, adjusted_probabilities)
        blocks = [z for z in np.array_split(x, min(5, x.size)) if z.size]
        block_q = np.array([np.quantile(z, [0.025, 0.975]) for z in blocks])
        diagnostic_rows.append(
            {
                "waveform": key[0],
                "action_id": int(key[1]),
                "action_code": ACTION_CODE[int(key[1])],
                "bootstrap_replicates": int(x.size),
                "point_continuous_n90": theta,
                "percentile_p025": q025,
                "percentile_p975": q975,
                "basic_bootstrap_low": 2.0 * theta - q975,
                "basic_bootstrap_high": 2.0 * theta - q025,
                "BCa_bootstrap_low": bca_low,
                "BCa_bootstrap_high": bca_high,
                "BCa_bias_correction": bias_correction,
                "BCa_acceleration": acceleration,
                "BCa_adjusted_lower_probability": adjusted_probabilities[0],
                "BCa_adjusted_upper_probability": adjusted_probabilities[1],
                "p025_monte_carlo_se": bootstrap_quantile_se(x, 0.025),
                "p975_monte_carlo_se": bootstrap_quantile_se(x, 0.975),
                "up_to_five_equal_blocks_p025_min": float(block_q[:, 0].min()),
                "up_to_five_equal_blocks_p025_max": float(block_q[:, 0].max()),
                "up_to_five_equal_blocks_p975_min": float(block_q[:, 1].min()),
                "up_to_five_equal_blocks_p975_max": float(block_q[:, 1].max()),
                "B_zero_or_negative_replicates": int((g.integrated_sigma2_athlete <= 0).sum()),
                "any_boundary_node_replicates": int((g.boundary_nodes > 0).sum()),
                "undefined_n90_replicates": int((~np.isfinite(x)).sum()),
                "maximum_continuous_n90": float(np.nanmax(x)),
            }
        )
    diagnostic_df = pd.DataFrame(diagnostic_rows)
    write_csv(diagnostic_df, args.out / "Table_S_BootstrapDiagnostics.csv")

    contrast_rows = []
    for waveform, g in boot.groupby("waveform"):
        wide = g.pivot(index="bootstrap_replicate", columns="action_id", values="continuous_n90")
        for left in range(1, 7):
            for right in range(left + 1, 7):
                delta = (wide[right] - wide[left]).to_numpy(float)
                q = np.quantile(delta, [0.025, 0.5, 0.975])
                contrast_rows.append(
                    {
                        "waveform": waveform,
                        "action_left": left,
                        "action_left_code": ACTION_CODE[left],
                        "action_right": right,
                        "action_right_code": ACTION_CODE[right],
                        "contrast": f"{ACTION_CODE[right]} minus {ACTION_CODE[left]}",
                        "continuous_n90_difference_p025": q[0],
                        "continuous_n90_difference_median": q[1],
                        "continuous_n90_difference_p975": q[2],
                        "percentile_interval_excludes_zero": bool(q[0] > 0 or q[2] < 0),
                        "multiplicity_adjusted": False,
                    }
                )
    write_csv(pd.DataFrame(contrast_rows), args.out / "Table_S_PairedActionBootstrapContrasts.csv")

    # The original bootstrap count stream is reproducible from the fixed seed.
    bootstrap_replicates = int(boot.bootstrap_replicate.nunique())
    rng_counts = np.random.default_rng(args.bootstrap_seed)
    counts = rng_counts.multinomial(
        30, np.full(30, 1.0 / 30.0), size=bootstrap_replicates
    )
    unique = (counts > 0).sum(axis=1)
    bootstrap_cluster_summary = {
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "drawn_clusters_per_replicate_with_replacement": 30,
        "unique_clusters_mean": float(unique.mean()),
        "unique_clusters_min": int(unique.min()),
        "unique_clusters_p025": float(np.quantile(unique, 0.025)),
        "unique_clusters_median": float(np.median(unique)),
        "unique_clusters_p975": float(np.quantile(unique, 0.975)),
        "unique_clusters_max": int(unique.max()),
    }
    (args.out / "BOOTSTRAP_CLUSTER_DIAGNOSTIC.json").write_text(
        json.dumps(bootstrap_cluster_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Compact machine-readable summary used by the manuscript and figure code.
    summary_json = {
        "archive_order": {
            "cells": int(len(order_df)),
            "all_numeric_indices_contiguous": bool(order_df.global_indices_contiguous.all()),
            "chronological_order_verified": False,
            "lag1_rho_min": float(acf_df.loc[acf_df.archive_order_lag == 1, "rho_L2_trace"].min()),
            "lag1_rho_max": float(acf_df.loc[acf_df.archive_order_lag == 1, "rho_L2_trace"].max()),
        },
        "backhand_drive_racket": {
            "players_with_hampel_flags": int((bd_player.displacement_values_flagged > 0).sum()),
            "total_hampel_flags": int(bd_player.displacement_values_flagged.sum()),
            "total_flagged_trials": int(bd_player.trials_flagged.sum()),
            "main_LOO_integer_n90_min": float(
                bd_player["integer_n90_unaltered_nonzero_primary"].min()
            ),
            "main_LOO_integer_n90_max": float(
                bd_player["integer_n90_unaltered_nonzero_primary"].max()
            ),
            "hampel_LOO_integer_n90_min": float(
                bd_player["integer_n90_hampel_type_local_high_value"].min()
            ),
            "hampel_LOO_integer_n90_max": float(
                bd_player["integer_n90_hampel_type_local_high_value"].max()
            ),
        },
        "settings": {
            "seed": args.seed,
            "bootstrap_seed": args.bootstrap_seed,
            "permutations": args.n_permutations,
            "acf_max_lag": args.acf_max_lag,
            "peak_smooth_window": args.peak_smooth_window,
            "peak_search_interval_pct": [args.peak_search_low_pct, args.peak_search_high_pct],
        },
    }
    (args.out / "ASSUMPTION_INFLUENCE_SUMMARY.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
