#!/usr/bin/env python3
"""Cross-action context analysis for the TTMD6 reliability story.

This script reproduces the balanced pointwise decomposition for the full
9000-trial primary cohort and adds a 5000-replicate participant-cluster
bootstrap.  Its purpose is to establish the global tension (small athlete main
effect, large athlete-by-action heterogeneity).  Action-specific models are
implemented in the next-stage script and carry the trial-design conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRIAL_COUNTS = (1, 2, 5, 10, 15, 20, 30, 40, 50)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--batch", type=int, default=100)
    return p.parse_args()


def integrate(y: np.ndarray) -> np.ndarray:
    return np.trapezoid(y, np.linspace(0.0, 1.0, y.shape[-1]), axis=-1)


def balanced_components(X: np.ndarray) -> dict[str, np.ndarray]:
    p, a, r, _ = X.shape
    gm = X.mean(axis=(0, 1, 2))
    pm = X.mean(axis=(1, 2))
    am = X.mean(axis=(0, 2))
    cm = X.mean(axis=2)
    ss_p = a * r * ((pm - gm) ** 2).sum(axis=0)
    ss_a = p * r * ((am - gm) ** 2).sum(axis=0)
    dev = cm - pm[:, None] - am[None] + gm
    ss_pa = r * (dev**2).sum(axis=(0, 1))
    ss_e = ((X - cm[:, :, None]) ** 2).sum(axis=(0, 1, 2))
    ms_p = ss_p / (p - 1)
    ms_a = ss_a / (a - 1)
    ms_pa = ss_pa / ((p - 1) * (a - 1))
    ms_e = ss_e / (p * a * (r - 1))
    raw = {
        "action": (ms_a - ms_pa) / (p * r),
        "athlete": (ms_p - ms_pa) / (a * r),
        "action_x_athlete": (ms_pa - ms_e) / r,
        "trial_residual": ms_e,
    }
    return {**{k: np.maximum(v, 0.0) for k, v in raw.items()}, **{f"raw_{k}": v for k, v in raw.items()}}


def scalar_summary(vc: dict[str, np.ndarray], family: str) -> dict:
    c = {k: float(integrate(vc[k])) for k in ("action", "athlete", "action_x_athlete", "trial_residual")}
    total = sum(c.values())
    row = {"curve_family": family, **{f"integrated_sigma2_{k}": v for k, v in c.items()}}
    row.update({f"omega_{k}": v / total for k, v in c.items()})
    pa = c["athlete"] + c["action_x_athlete"]
    signal = c["action"] + pa
    row["context_relative_reliability_athlete_across_actions_n50"] = c["athlete"] / (
        c["athlete"] + c["action_x_athlete"] / 6 + c["trial_residual"] / (6 * 50)
    )
    row["context_relative_reliability_all_cells_n50"] = signal / (signal + c["trial_residual"] / 50)
    row["context_relative_reliability_pooled_fixed_actions_n50"] = pa / (pa + c["trial_residual"] / 50)
    return row


def pointwise_table(vc: dict[str, np.ndarray], family: str) -> pd.DataFrame:
    q = len(vc["action"])
    total = sum(vc[k] for k in ("action", "athlete", "action_x_athlete", "trial_residual"))
    out = pd.DataFrame({"curve_family": family, "normalised_time_pct": np.linspace(0, 100, q)})
    for k in ("action", "athlete", "action_x_athlete", "trial_residual"):
        out[f"sigma2_{k}"] = vc[k]
        out[f"raw_sigma2_{k}"] = vc[f"raw_{k}"]
        out[f"omega_{k}"] = np.divide(vc[k], total, out=np.zeros(q), where=total > 0)
    return out


def d_study(summary: dict) -> pd.DataFrame:
    a = summary["integrated_sigma2_action"]
    p = summary["integrated_sigma2_athlete"]
    pa = summary["integrated_sigma2_action_x_athlete"]
    e = summary["integrated_sigma2_trial_residual"]
    rows = []
    for n in TRIAL_COUNTS:
        rows.append({
            "curve_family": summary["curve_family"],
            "trials_per_cell": n,
            "context_relative_reliability_all_cells": (a + p + pa) / (a + p + pa + e / n),
            "context_relative_reliability_pooled_fixed_actions": (p + pa) / (p + pa + e / n),
            "context_relative_reliability_athlete_across_actions": p / (p + pa / 6 + e / (6 * n)),
        })
    return pd.DataFrame(rows)


def bootstrap_global(X: np.ndarray, family: str, B: int, seed: int, batch: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p, a, r, q = X.shape
    cm = X.mean(axis=2)
    within_ss = ((X - cm[:, :, None]) ** 2).sum(axis=2)
    rows = []
    done = 0
    while done < B:
        b = min(batch, B - done)
        idx = rng.integers(0, p, size=(b, p))
        C = cm[idx]
        gm = C.mean(axis=(1, 2))
        pm = C.mean(axis=2)
        am = C.mean(axis=1)
        ss_p = a * r * ((pm - gm[:, None]) ** 2).sum(axis=1)
        ss_a = p * r * ((am - gm[:, None]) ** 2).sum(axis=1)
        dev = C - pm[:, :, None] - am[:, None] + gm[:, None, None]
        ss_pa = r * (dev**2).sum(axis=(1, 2))
        ss_e = within_ss[idx].sum(axis=(1, 2))
        ms_p = ss_p / (p - 1)
        ms_a = ss_a / (a - 1)
        ms_pa = ss_pa / ((p - 1) * (a - 1))
        ms_e = ss_e / (p * a * (r - 1))
        comps = {
            "action": np.maximum((ms_a - ms_pa) / (p * r), 0),
            "athlete": np.maximum((ms_p - ms_pa) / (a * r), 0),
            "action_x_athlete": np.maximum((ms_pa - ms_e) / r, 0),
            "trial_residual": np.maximum(ms_e, 0),
        }
        I = {k: integrate(v) for k, v in comps.items()}
        total = sum(I.values())
        signal = I["action"] + I["athlete"] + I["action_x_athlete"]
        pooled = I["athlete"] + I["action_x_athlete"]
        for j in range(b):
            row = {
                "curve_family": family,
                "bootstrap": done + j + 1,
                **{f"omega_{k}": float(I[k][j] / total[j]) for k in I},
                "context_relative_reliability_all_cells_n10": float(signal[j] / (signal[j] + I["trial_residual"][j] / 10)),
                "context_relative_reliability_pooled_fixed_actions_n10": float(
                    pooled[j] / (pooled[j] + I["trial_residual"][j] / 10)
                ),
                "context_relative_reliability_athlete_across_actions_n50": float(
                    I["athlete"][j]
                    / (
                        I["athlete"][j]
                        + I["action_x_athlete"][j] / 6
                        + I["trial_residual"][j] / (6 * 50)
                    )
                ) if I["athlete"][j] > 0 else 0.0,
            }
            rows.append(row)
        done += b
    return pd.DataFrame(rows)


def summarise_boot(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in {"curve_family", "bootstrap"}]
    rows = []
    for fam, g in df.groupby("curve_family"):
        for metric in metrics:
            x = g[metric].to_numpy(float)
            rows.append({
                "curve_family": fam,
                "metric": metric,
                "bootstrap_replicates": len(x),
                "median": float(np.median(x)),
                "ci_low_2_5pct": float(np.quantile(x, 0.025)),
                "ci_high_97_5pct": float(np.quantile(x, 0.975)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "tables").mkdir(exist_ok=True)
    inputs = {
        "racket": cfg.cache_dir / "racket_displacement_9000.npy",
        "body14mean": cfg.cache_dir / "body14mean_displacement_9000.npy",
    }
    scalar, point, dstudy, boots = [], [], [], []
    for i, (family, path) in enumerate(inputs.items()):
        X = np.load(path)
        vc = balanced_components(X)
        s = scalar_summary(vc, family)
        scalar.append(s)
        point.append(pointwise_table(vc, family))
        dstudy.append(d_study(s))
        boots.append(bootstrap_global(X, family, cfg.n_bootstrap, cfg.seed + i * 1009, cfg.batch))
    pd.DataFrame(scalar).to_csv(cfg.out / "tables" / "Table_04_GlobalVarianceSummary.csv", index=False)
    pd.concat(point, ignore_index=True).to_csv(cfg.out / "tables" / "Table_05_GlobalPointwiseComponents.csv", index=False)
    pd.concat(dstudy, ignore_index=True).to_csv(cfg.out / "tables" / "Table_06_GlobalDStudy_ContextOnly.csv", index=False)
    boot = pd.concat(boots, ignore_index=True)
    boot.to_csv(cfg.out / "tables" / "Table_07_GlobalClusterBootstrap5000_Replicates.csv", index=False)
    summarise_boot(boot).to_csv(cfg.out / "tables" / "Table_08_GlobalClusterBootstrap5000_Summary.csv", index=False)
    log = {
        "n_bootstrap": cfg.n_bootstrap,
        "seed": cfg.seed,
        "cluster_unit": "participant",
        "independent_population_clusters": 30,
        "note": "Global model is context; action-specific models carry trial-design conclusions.",
    }
    (cfg.out / "GLOBAL_RUN.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(pd.DataFrame(scalar).to_string(index=False))


if __name__ == "__main__":
    main()
