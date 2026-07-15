#!/usr/bin/env python3
"""Whole-athlete pointwise bootstrap bands for the fixed six-label decomposition.

The bands are descriptive marginal uncertainty bands at each normalized time
node.  They are not simultaneous confidence bands and do not support local
significance testing.  Integrated quantities continue to use the unsmoothed
point estimates from the primary analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = ("action", "athlete", "action_x_athlete", "trial_residual")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--batch", type=int, default=100)
    return p.parse_args()


def bootstrap_bands(
    X: np.ndarray,
    family: str,
    n_bootstrap: int,
    seed: int,
    batch_size: int,
) -> pd.DataFrame:
    p, a, r, q = X.shape
    rng = np.random.default_rng(seed)
    cell_mean = X.mean(axis=2)
    within_ss = ((X - cell_mean[:, :, None]) ** 2).sum(axis=2)
    draws = {component: np.empty((n_bootstrap, q), dtype=np.float32) for component in COMPONENTS}
    done = 0
    while done < n_bootstrap:
        b = min(batch_size, n_bootstrap - done)
        idx = rng.integers(0, p, size=(b, p))
        C = cell_mean[idx]
        grand = C.mean(axis=(1, 2))
        athlete_mean = C.mean(axis=2)
        action_mean = C.mean(axis=1)
        ss_p = a * r * ((athlete_mean - grand[:, None]) ** 2).sum(axis=1)
        ss_a = p * r * ((action_mean - grand[:, None]) ** 2).sum(axis=1)
        interaction = C - athlete_mean[:, :, None] - action_mean[:, None] + grand[:, None, None]
        ss_pa = r * (interaction**2).sum(axis=(1, 2))
        ss_e = within_ss[idx].sum(axis=(1, 2))
        ms_p = ss_p / (p - 1)
        ms_a = ss_a / (a - 1)
        ms_pa = ss_pa / ((p - 1) * (a - 1))
        ms_e = ss_e / (p * a * (r - 1))
        components = {
            "action": np.maximum((ms_a - ms_pa) / (p * r), 0.0),
            "athlete": np.maximum((ms_p - ms_pa) / (a * r), 0.0),
            "action_x_athlete": np.maximum((ms_pa - ms_e) / r, 0.0),
            "trial_residual": np.maximum(ms_e, 0.0),
        }
        total = sum(components.values())
        for component in COMPONENTS:
            draws[component][done : done + b] = np.divide(
                components[component],
                total,
                out=np.zeros_like(total),
                where=total > 0,
            )
        done += b

    rows = []
    time = np.linspace(0.0, 100.0, q)
    for component in COMPONENTS:
        qv = np.quantile(draws[component], [0.025, 0.5, 0.975], axis=0)
        for node in range(q):
            rows.append(
                {
                    "curve_family": family,
                    "component": component,
                    "time_node_0based": node,
                    "normalised_time_pct": time[node],
                    "bootstrap_replicates": n_bootstrap,
                    "pointwise_p025": qv[0, node],
                    "pointwise_median": qv[1, node],
                    "pointwise_p975": qv[2, node],
                    "interval_type": "marginal_whole_athlete_percentile_not_simultaneous",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    inputs = {
        "racket": args.cache_dir / "racket_displacement_9000.npy",
        "body14mean": args.cache_dir / "body14mean_displacement_9000.npy",
    }
    tables = []
    for offset, (family, path) in enumerate(inputs.items()):
        X = np.load(path)
        if X.shape != (30, 6, 50, 200):
            raise ValueError(f"Unexpected array shape for {family}: {X.shape}")
        tables.append(
            bootstrap_bands(
                X,
                family,
                args.n_bootstrap,
                args.seed + offset * 1009,
                args.batch,
            )
        )
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig", float_format="%.12g")
    print(out.groupby(["curve_family", "component"]).size().to_string())


if __name__ == "__main__":
    main()
