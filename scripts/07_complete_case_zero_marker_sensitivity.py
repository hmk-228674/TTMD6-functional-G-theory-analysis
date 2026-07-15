#!/usr/bin/env python3
"""Sensitivity: exclude whole trials containing any structural zero marker.

The primary body waveform retains trials and averages each frame interval over
the 10--14 joints observed at both endpoints.  This deliberately more severe
complete-case analysis drops every trial with at least one exact [0,0,0]
human-joint triplet and fits the same nodewise unbalanced random-intercept REML.
It is a sensitivity analysis, not the preferred handling of partial marker
missingness.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arrays", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--reml-script", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_reml(path: Path):
    spec = importlib.util.spec_from_file_location("ttmd6_reml", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    a = parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    mod = load_reml(a.reml_script.resolve())
    x = np.load(a.arrays)
    audit = pd.read_csv(a.audit)
    keep = np.ones((30, 6, 50), dtype=bool)
    for r in audit.itertuples(index=False):
        if r.body_zero_triplets > 0:
            keep[r.participant_code - 1, r.action_id - 1, r.trial_in_cell - 1] = False
    w = mod.integration_weights(x.shape[-1])
    rows = []
    for action in range(6):
        n, sums, sums_sq = mod.unbalanced_sufficient(x[:, action], keep[:, action])
        fit = mod.unbalanced_profile_reml(n, sums, sums_sq)
        comp = fit["components"]
        bi = float(mod.integrate(comp.between, w))
        wi = float(mod.integrate(comp.within, w))
        row = {
            "action_id": action + 1,
            "action_label": mod.ACTION_LABEL[action + 1],
            "retained_trials": int(n.sum()),
            "excluded_zero_marker_trials": int((~keep[:, action]).sum()),
            "min_trials_per_athlete": int(n.min()),
            "max_trials_per_athlete": int(n.max()),
            "integrated_sigma2_athlete": bi,
            "integrated_sigma2_trial": wi,
            "boundary_nodes": int((~comp.interior).sum()),
        }
        for m in (1, 5, 10, 50):
            row[f"R_L2_m{m}"] = bi / (bi + wi / m)
        row["required_n_R_L2_80"] = int(mod.required_trials(bi, wi, .80))
        row["required_n_R_L2_90"] = int(mod.required_trials(bi, wi, .90))
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig", float_format="%.12g")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
