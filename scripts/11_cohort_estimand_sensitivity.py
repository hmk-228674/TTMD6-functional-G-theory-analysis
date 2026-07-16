#!/usr/bin/env python3
"""Cohort-boundary, exact-deduplication, and functional-aggregation sensitivity.

The source article reports 30 participants, whereas the publisher-hosted
archive exposes complete codes 1--40 and does not document a participant-code
crosswalk.  This script therefore treats code identity as unresolved and
compares four transparent archive-code scenarios:

1. codes 1--30 (the deterministic primary block);
2. codes 11--40 (an alternative contiguous 30-code block);
3. all codes 1--40; and
4. all codes 1--40 after removing the quarantined-block member of each exact
   paired-record duplicate group.

The scenarios are robustness checks, not attempts to relabel the archive or
to claim that 40 independent participants were recruited.  The script also
places the L2 trace-ratio threshold beside the threshold obtained by averaging
the pointwise reliability curve over phase.  This makes the estimand choice
visible rather than silently treating the two summaries as interchangeable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_action_specific_reml import (
    ACTION_EN,
    ACTION_LABEL,
    balanced_reml,
    bootstrap_balanced_from_counts,
    component_summary,
    integrate,
    integration_weights,
    required_m_pointwise_batch,
    required_trials,
    unbalanced_profile_reml,
    unbalanced_sufficient,
)


BAT_RE = re.compile(
    r"bat_120_(?P<index>\d+)_(?P<action>\d+)_(?P<code>\d+)_200_(?P<rawlen>\d+)\.csv$"
)

SCENARIOS = (
    {
        "scenario": "codes_1_30_primary",
        "codes": tuple(range(1, 31)),
        "interpretation": "deterministic_primary_archive_block_not_author_confirmed",
    },
    {
        "scenario": "codes_11_40_alternative_window",
        "codes": tuple(range(11, 41)),
        "interpretation": "alternative_contiguous_30_code_boundary_check",
    },
    {
        "scenario": "codes_1_40_all_archive_codes",
        "codes": tuple(range(1, 41)),
        "interpretation": "complete_archive_code_scenario_identity_not_established",
    },
)

JOINT_LABELS = (
    (1, "Hips", "髋部/骨盆"),
    (2, "Head", "头部"),
    (3, "LeftShoulder", "左肩"),
    (4, "LeftArm", "左上臂"),
    (5, "LeftForeArm", "左前臂"),
    (6, "RightShoulder", "右肩"),
    (7, "RightArm", "右上臂"),
    (8, "RightForeArm", "右前臂"),
    (9, "LeftUpLeg", "左大腿"),
    (10, "LeftLeg", "左小腿"),
    (11, "LeftFoot", "左足"),
    (12, "RightUpLeg", "右大腿"),
    (13, "RightLeg", "右小腿"),
    (14, "RightFoot", "右足"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ttmd6-root", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--duplicates", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-phase", type=int, default=200)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260714)
    return p.parse_args()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def read_coordinates(path: Path) -> np.ndarray:
    a = pd.read_csv(path, header=None, encoding="utf-8-sig").to_numpy(float)
    if a.shape[0] == 400 and np.array_equal(a[:200], a[200:400]):
        a = a[:200]
    if a.shape[0] != 200 or not np.isfinite(a).all():
        raise ValueError(f"Unexpected coordinate structure: {path} {a.shape}")
    return a


def phase_normalise(y: np.ndarray, n_phase: int) -> np.ndarray:
    if y.size < 2 or not np.isfinite(y).all():
        raise ValueError("A finite displacement sequence with at least two values is required")
    return np.interp(
        np.linspace(0.0, 1.0, n_phase),
        np.linspace(0.0, 1.0, y.size),
        y,
    )


def derive_all_code_arrays(root: Path, n_phase: int) -> tuple[np.ndarray, np.ndarray]:
    bat_dir = root / "TTMD_cut_bat"
    hum_dir = root / "TTMD_cut_hum"
    records: list[tuple[dict[str, int], Path, Path]] = []
    for bp in bat_dir.glob("*.csv"):
        match = BAT_RE.fullmatch(bp.name)
        if not match:
            continue
        meta = {key: int(value) for key, value in match.groupdict().items()}
        hp = hum_dir / bp.name.replace("bat_", "human_", 1)
        records.append((meta, bp, hp))
    records.sort(key=lambda z: (z[0]["code"], z[0]["action"], z[0]["index"]))
    if len(records) != 12_000:
        raise RuntimeError(f"Expected 12,000 paired records, observed {len(records)}")

    racket = np.empty((40, 6, 50, n_phase), float)
    body = np.empty_like(racket)
    counters: defaultdict[tuple[int, int], int] = defaultdict(int)
    for k, (meta, bp, hp) in enumerate(records, 1):
        n = min(meta["rawlen"], 200)
        bat = read_coordinates(bp)[:n]
        human = read_coordinates(hp)[:n].reshape(n, 14, 3)

        if np.any(np.all(bat == 0.0, axis=1)):
            raise ValueError(f"Unexpected racket zero sentinel: {bp.name}")
        racket_step = np.linalg.norm(np.diff(bat, axis=0), axis=1)

        human_zero = np.all(human == 0.0, axis=2)
        valid_pair = ~(human_zero[:-1] | human_zero[1:])
        valid_count = valid_pair.sum(axis=1)
        if np.any(valid_count < 10):
            raise ValueError(f"Fewer than 10 usable landmarks on an interval: {hp.name}")
        marker_step = np.linalg.norm(np.diff(human, axis=0), axis=2)
        body_step = np.where(valid_pair, marker_step, 0.0).sum(axis=1) / valid_count

        trial = counters[(meta["code"], meta["action"])]
        counters[(meta["code"], meta["action"])] += 1
        if trial >= 50:
            raise RuntimeError("Archive cell contains more than 50 records")
        p, a = meta["code"] - 1, meta["action"] - 1
        racket[p, a, trial] = phase_normalise(racket_step, n_phase)
        body[p, a, trial] = phase_normalise(body_step, n_phase)
        if k % 2000 == 0:
            print(f"derived {k}/12000 records", flush=True)

    if len(counters) != 240 or any(value != 50 for value in counters.values()):
        raise RuntimeError("Archive is not a complete 40 x 6 x 50 code design")
    if not np.isfinite(racket).all() or not np.isfinite(body).all():
        raise RuntimeError("Derived arrays contain non-finite values")
    return racket, body


def bootstrap_threshold_summary(
    comp,
    weights: np.ndarray,
    counts: np.ndarray,
    group_means: np.ndarray,
    within_by_group: np.ndarray,
    m: int,
    scenario: str,
    waveform: str,
    action_id: int,
) -> list[dict]:
    boot = bootstrap_balanced_from_counts(group_means, within_by_group, m, counts)
    b_i = integrate(boot.between, weights)
    w_i = integrate(boot.within, weights)
    metrics: dict[str, np.ndarray] = {}
    for target in (0.80, 0.90):
        suffix = int(target * 100)
        metrics[f"required_n_R_L2_{suffix}"] = required_trials(b_i, w_i, target)
        metrics[f"required_n_mean_pointwise_R_{suffix}"] = required_m_pointwise_batch(
            boot.between, boot.within, weights, target
        )
    rows = []
    for metric, values in metrics.items():
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        q025, q50, q975 = np.percentile(finite, [2.5, 50.0, 97.5])
        rows.append(
            {
                "scenario": scenario,
                "waveform": waveform,
                "action_id": action_id,
                "action_label_en": ACTION_EN[action_id],
                "action_label": ACTION_LABEL[action_id],
                "metric": metric,
                "reference_estimate": component_summary(comp, weights)[metric],
                "n_cluster_bootstrap": len(values),
                "finite_replicates": len(finite),
                "percentile_2_5": q025,
                "percentile_50": q50,
                "percentile_97_5": q975,
            }
        )
    return rows


def deduplication_mask(
    manifest: pd.DataFrame, duplicates: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    keep = np.ones((40, 6, 50), dtype=bool)
    removal_rows = []
    for group_id, group in duplicates.groupby("duplicate_group", sort=True):
        if len(group) < 2:
            continue
        quarantined = group[group["participant_code"] > 30]
        if len(quarantined) == 1:
            chosen = quarantined.iloc[0]
            rule = "remove_codes_31_40_member_preserve_primary_block"
        else:
            chosen = group.sort_values(
                ["participant_code", "global_index"], ascending=[False, False]
            ).iloc[0]
            rule = "remove_lexicographically_last_record"
        matched = manifest[manifest["global_index"] == int(chosen["global_index"])]
        if len(matched) != 1:
            raise RuntimeError(f"Could not map duplicate global index {chosen['global_index']}")
        row = matched.iloc[0]
        p = int(row["participant_code"]) - 1
        a = int(row["action_id"]) - 1
        t = int(row["trial_in_cell"]) - 1
        keep[p, a, t] = False
        removal_rows.append(
            {
                "duplicate_group": int(group_id),
                "removed_global_index": int(row["global_index"]),
                "removed_participant_code": int(row["participant_code"]),
                "removed_action_id": int(row["action_id"]),
                "removed_trial_in_cell": int(row["trial_in_cell"]),
                "effective_pair_sha256": row["effective_pair_sha256"],
                "deterministic_removal_rule": rule,
            }
        )
    return keep, pd.DataFrame(removal_rows)


def joint_mapping_frame() -> pd.DataFrame:
    rows = []
    for index, label, chinese in JOINT_LABELS:
        first = 3 * (index - 1) + 1
        rows.append(
            {
                "joint_index": index,
                "published_label": label,
                "chinese_mapping": chinese,
                "archive_coordinate_columns_1based": f"{first}-{first + 2}",
                "coordinate_header_status": "archive_csv_has_no_axis_header",
                "mapping_source": "Zhang_et_al_2024_Figure_1",
                "interpretation_limit": "published_segment_label_not_an_exact_marker-centre_definition",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    duplicates = pd.read_csv(args.duplicates)
    required = {"global_index", "participant_code", "action_id", "trial_in_cell", "effective_pair_sha256"}
    if len(manifest) != 12_000 or not required.issubset(manifest.columns):
        raise ValueError("Expected the complete 12,000-record archive manifest")

    racket, body = derive_all_code_arrays(args.ttmd6_root, args.n_phase)
    arrays = {"racket": racket, "body_configuration": body}
    weights = integration_weights(args.n_phase)
    point_rows: list[dict] = []
    bootstrap_rows: list[dict] = []

    for scenario_index, spec in enumerate(SCENARIOS):
        code_idx = np.asarray(spec["codes"], int) - 1
        n_groups = len(code_idx)
        rng = np.random.default_rng(args.seed + scenario_index * 1009)
        counts = rng.multinomial(
            n_groups, np.full(n_groups, 1.0 / n_groups), size=args.n_bootstrap
        )
        for waveform, full in arrays.items():
            for action_id in range(1, 7):
                data = full[code_idx, action_id - 1, :, :]
                comp, group_means, within_by_group = balanced_reml(data)
                point_rows.append(
                    {
                        "scenario": spec["scenario"],
                        "scenario_interpretation": spec["interpretation"],
                        "analysis_method": "balanced_constrained_REML",
                        "waveform": waveform,
                        "action_id": action_id,
                        "action_label_en": ACTION_EN[action_id],
                        "action_label": ACTION_LABEL[action_id],
                        "code_definition": f"{min(spec['codes'])}-{max(spec['codes'])}",
                        "n_archive_codes": n_groups,
                        "n_trials_total": n_groups * 50,
                        "min_trials_per_code": 50,
                        "max_trials_per_code": 50,
                        "exact_duplicates_removed": 0,
                        "participant_identity_author_confirmed": False,
                        **component_summary(comp, weights),
                    }
                )
                bootstrap_rows.extend(
                    bootstrap_threshold_summary(
                        comp,
                        weights,
                        counts,
                        group_means,
                        within_by_group,
                        50,
                        spec["scenario"],
                        waveform,
                        action_id,
                    )
                )

    keep, removal_audit = deduplication_mask(manifest, duplicates)
    for waveform, full in arrays.items():
        for action_id in range(1, 7):
            action_keep = keep[:, action_id - 1, :]
            n, sums, sums_sq = unbalanced_sufficient(
                full[:, action_id - 1, :, :], action_keep
            )
            fit = unbalanced_profile_reml(n, sums, sums_sq)
            comp = fit["components"]
            point_rows.append(
                {
                    "scenario": "codes_1_40_exact_pair_deduplicated",
                    "scenario_interpretation": "all_archive_codes_after_one_record_per_exact_pair_group_removed",
                    "analysis_method": "unbalanced_profile_REML",
                    "waveform": waveform,
                    "action_id": action_id,
                    "action_label_en": ACTION_EN[action_id],
                    "action_label": ACTION_LABEL[action_id],
                    "code_definition": "1-40",
                    "n_archive_codes": 40,
                    "n_trials_total": int(n.sum()),
                    "min_trials_per_code": int(n.min()),
                    "max_trials_per_code": int(n.max()),
                    "exact_duplicates_removed": int((~action_keep).sum()),
                    "participant_identity_author_confirmed": False,
                    **component_summary(comp, weights),
                }
            )

    point = pd.DataFrame(point_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    primary = point[point["scenario"] == "codes_1_30_primary"].copy()
    aggregation_columns = [
        "waveform",
        "action_id",
        "action_label_en",
        "action_label",
        "required_n_R_L2_80",
        "required_n_mean_pointwise_R_80",
        "required_n_R_L2_90",
        "required_n_mean_pointwise_R_90",
    ]
    aggregation = primary[aggregation_columns].copy()
    aggregation["pointwise_minus_trace_n80"] = (
        aggregation["required_n_mean_pointwise_R_80"] - aggregation["required_n_R_L2_80"]
    )
    aggregation["pointwise_minus_trace_n90"] = (
        aggregation["required_n_mean_pointwise_R_90"] - aggregation["required_n_R_L2_90"]
    )
    aggregation["estimand_interpretation"] = (
        "trace_ratio_and_phase_mean_pointwise_ratio_are_distinct_functional_summaries"
    )

    write_csv(point, args.out / "Table_S_CohortAndDedupSensitivity.csv")
    write_csv(bootstrap, args.out / "Table_S_CohortBootstrapThresholds.csv")
    write_csv(aggregation, args.out / "Table_S_FunctionalAggregationComparison.csv")
    write_csv(removal_audit, args.out / "Table_S_ExactDuplicateRemovalAudit.csv")
    write_csv(joint_mapping_frame(), args.out / "Table_S_14JointAnatomicalMapping.csv")

    bd = aggregation[
        (aggregation["waveform"] == "racket") & (aggregation["action_id"] == 5)
    ].iloc[0]
    summary = {
        "status": "complete",
        "seed": args.seed,
        "n_cluster_bootstrap": args.n_bootstrap,
        "participant_code_crosswalk_available": False,
        "cohort_scenarios": [x["scenario"] for x in SCENARIOS]
        + ["codes_1_40_exact_pair_deduplicated"],
        "exact_pair_records_removed": int(len(removal_audit)),
        "backhand_drive_racket_n90": {
            "L2_trace_ratio": int(bd["required_n_R_L2_90"]),
            "phase_mean_pointwise_ratio": int(bd["required_n_mean_pointwise_R_90"]),
        },
        "inference_boundary": (
            "Archive-code scenarios do not establish the number or identity of independent participants."
        ),
    }
    (args.out / "COHORT_ESTIMAND_SENSITIVITY_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
