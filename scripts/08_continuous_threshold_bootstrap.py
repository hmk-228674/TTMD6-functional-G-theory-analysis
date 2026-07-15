#!/usr/bin/env python3
"""Export continuous and integer G-threshold bootstrap summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=Path, required=True)
    p.add_argument("--out-draws", type=Path, required=True)
    p.add_argument("--out-summary", type=Path, required=True)
    a = p.parse_args()
    d = pd.read_csv(a.draws)
    rows = []
    for g in (.80, .90):
        continuous = g * d.integrated_sigma2_trial / ((1.0 - g) * d.integrated_sigma2_athlete)
        x = d[["waveform", "action_id", "action_label", "bootstrap_replicate"]].copy()
        x["target_R_L2"] = g
        x["continuous_required_n"] = continuous
        x["integer_required_n"] = np.maximum(1, np.ceil(continuous))
        rows.append(x)
    out = pd.concat(rows, ignore_index=True)
    a.out_draws.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_draws, index=False, encoding="utf-8-sig", float_format="%.12g")
    summary = out.groupby(["waveform", "action_id", "action_label", "target_R_L2"]).agg(
        replicates=("continuous_required_n", "size"),
        continuous_median=("continuous_required_n", "median"),
        integer_median=("integer_required_n", "median"),
    ).reset_index()
    qs = out.groupby(["waveform", "action_id", "action_label", "target_R_L2"])[
        ["continuous_required_n", "integer_required_n"]
    ].quantile([.025, .975]).unstack()
    qs.columns = [f"{v}_{'p025' if q == .025 else 'p975'}" for v, q in qs.columns]
    summary = summary.merge(qs.reset_index(), on=["waveform", "action_id", "action_label", "target_R_L2"])
    summary.to_csv(a.out_summary, index=False, encoding="utf-8-sig", float_format="%.12g")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
