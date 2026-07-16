#!/usr/bin/env python3
"""Build all Scientific Reports figures and their machine-readable source data.

Backend: Python/matplotlib only.  Figures use a fixed 162 mm canvas width,
lower-case panel labels, editable SVG/PDF text, 300 dpi PNG previews, and
600 dpi LZW-compressed TIFF files.  The script also emits a QA manifest that
checks canvas dimensions, TIFF compression/DPI, SVG text preservation, and
text-object bounds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from PIL import Image


FIG_WIDTH_MM = 162.0
MM_PER_INCH = 25.4
MIN_FONT = 7.2

ACTION_CODES = ["FA", "FD", "FP", "BA", "BD", "BP"]
ACTION_EN = {
    "FA": "forehand attack",
    "FD": "forehand drive",
    "FP": "forehand push",
    "BA": "backhand attack",
    "BD": "backhand drive",
    "BP": "backhand push",
}
ACTION_COLORS = {
    "FA": "#0072B2",
    "FD": "#009E73",
    "FP": "#56B4E9",
    "BA": "#E69F00",
    "BD": "#D55E00",
    "BP": "#CC79A7",
}
ACTION_LINESTYLES = {
    "FA": "-",
    "FD": "--",
    "FP": "-.",
    "BA": (0, (5, 1.5)),
    "BD": (0, (2, 1.3)),
    "BP": (0, (1, 1.2)),
}
COMPONENTS = [
    ("omega_action", "fixed-action contrast dispersion", "#0072B2", "-"),
    ("omega_athlete", "athlete", "#B58900", "--"),
    ("omega_action_x_athlete", "athlete × fixed action", "#D55E00", "-."),
    ("omega_trial_residual", "within-cell unexplained", "#777777", ":"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def configure_style() -> str:
    installed = {font.name for font in mpl.font_manager.fontManager.ttflist}
    family = next(
        (candidate for candidate in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if candidate in installed),
        "DejaVu Sans",
    )
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.8,
            "xtick.labelsize": MIN_FONT,
            "ytick.labelsize": MIN_FONT,
            "legend.fontsize": MIN_FONT,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "patch.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )
    return family


def make_figure(height_mm: float) -> plt.Figure:
    # This Matplotlib build quantizes requested inches to two decimals.
    return plt.figure(
        figsize=(round(FIG_WIDTH_MM / MM_PER_INCH, 2), round(height_mm / MM_PER_INCH, 2))
    )


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.6,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def clean_axes(ax: mpl.axes.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E1E1E1", linewidth=0.55, zorder=0)
    ax.tick_params(direction="out")


def text_boundary_violations(fig: plt.Figure, tolerance_px: float = 3.0) -> list[dict]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    violations = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if (
            bbox.x0 < fb.x0 - tolerance_px
            or bbox.y0 < fb.y0 - tolerance_px
            or bbox.x1 > fb.x1 + tolerance_px
            or bbox.y1 > fb.y1 + tolerance_px
        ):
            violations.append(
                {
                    "text": artist.get_text()[:80],
                    "bbox_px": [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)],
                    "figure_bbox_px": [float(fb.x0), float(fb.y0), float(fb.x1), float(fb.y1)],
                }
            )
    return violations


def save_figure(fig: plt.Figure, out: Path, stem: str, qa_rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    violations = text_boundary_violations(fig)
    png = out / f"{stem}.png"
    tif = out / f"{stem}.tif"
    pdf = out / f"{stem}.pdf"
    svg = out / f"{stem}.svg"
    fig.savefig(png, dpi=300, format="png", facecolor="white")
    fig.savefig(
        tif,
        dpi=600,
        format="tiff",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(pdf, format="pdf", facecolor="white")
    fig.savefig(svg, format="svg", facecolor="white")
    width_mm, height_mm = fig.get_size_inches() * MM_PER_INCH
    with Image.open(tif) as image:
        compression_tag = int(image.tag_v2.get(259, -1))
        dpi = image.info.get("dpi", (math.nan, math.nan))
        tiff_size = image.size
    svg_text = svg.read_text(encoding="utf-8")
    expected_width_600 = int(round(fig.get_size_inches()[0] * 600))
    qa_rows.append(
        {
            "figure": stem,
            "width_mm": float(width_mm),
            "height_mm": float(height_mm),
            "fixed_width_pass": bool(abs(width_mm - FIG_WIDTH_MM) < 0.10),
            "tiff_width_px": int(tiff_size[0]),
            "tiff_height_px": int(tiff_size[1]),
            "tiff_expected_width_px_600dpi": expected_width_600,
            "tiff_dimension_pass": bool(abs(tiff_size[0] - expected_width_600) <= 1),
            "tiff_compression_tag": compression_tag,
            "tiff_lzw_pass": bool(compression_tag == 5),
            "tiff_dpi_x": float(dpi[0]),
            "tiff_dpi_y": float(dpi[1]),
            "tiff_600dpi_pass": bool(abs(float(dpi[0]) - 600) < 1 and abs(float(dpi[1]) - 600) < 1),
            "svg_editable_text_pass": bool("<text" in svg_text),
            "text_boundary_violation_count": len(violations),
            "text_boundary_violations": json.dumps(violations, ensure_ascii=False),
        }
    )
    plt.close(fig)


def add_box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    face: str,
    title_size: float = 7.8,
    body_size: float = 7.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        edgecolor="#3F3F3F",
        facecolor=face,
        linewidth=0.85,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + height - 0.030, title, fontsize=title_size, fontweight="bold", va="top")
    ax.text(
        x + 0.018,
        y + height - 0.090,
        body,
        fontsize=body_size,
        va="top",
        linespacing=1.22,
    )


def figure1(out: Path, qa: list[dict]) -> None:
    fig = make_figure(112)
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    add_box(
        ax,
        0.02,
        0.70,
        0.225,
        0.255,
        "Data hierarchy",
        "30 athlete codes (inference units)\n6 fixed English action labels\n50 archived trials per cell\n9,000 paired coordinate files",
        "#EAF3F8",
    )
    add_box(
        ax,
        0.275,
        0.70,
        0.285,
        0.255,
        "Archive audit",
        "Collapsed 50 duplicated 400-row blocks\n190 body trials contain structural zeros\n1,180 nominal lengths exceed 200 frames\nCodes 31–40 quarantined",
        "#F5F0E4",
    )
    add_box(
        ax,
        0.590,
        0.70,
        0.390,
        0.255,
        "Operational waveforms without event registration",
        "Racket: framewise 3-D centroid displacement\nBody: mean displacement across available joints\nObserved segment normalized to 200 nodes\nNot face angle, centre of mass, or coordination",
        "#EAF5EC",
    )

    add_box(
        ax,
        0.06,
        0.385,
        0.39,
        0.205,
        "Contextual decomposition of six fixed labels",
        "Finite-set action contrast dispersion\n+ athlete + athlete × fixed action\n+ within-cell unexplained variation (descriptive)",
        "#EEF0F8",
    )
    add_box(
        ax,
        0.55,
        0.385,
        0.39,
        0.205,
        "Twelve action-specific design models",
        "6 labels × 2 representations estimate B and W\n$L^2$-trace relative reliability R(n)\nTrial counts summarized at 0.80 and 0.90",
        "#FAEEE8",
    )

    add_box(
        ax,
        0.05,
        0.055,
        0.90,
        0.205,
        "Uncertainty, assumptions, and inferential boundary",
        "Left: 5,000 whole-athlete bootstrap replicates;\ncontinuous thresholds, interval diagnostics, and leave-one-out influence\nRight: archive-order trace correlation and AR(1) scenarios;\npeak registration, 200-row boundary, balanced resampling, and Hampel-type rule\nBoundary: within-session relative distinguishability; no extrapolation\nto all techniques, populations, days, or racket sports",
        "#F3F3F3",
        body_size=6.85,
    )

    arrow = dict(arrowstyle="-|>", linewidth=1.0, color="#555555", shrinkA=2, shrinkB=2)
    ax.annotate("", xy=(0.275, 0.827), xytext=(0.245, 0.827), arrowprops=arrow)
    ax.annotate("", xy=(0.590, 0.827), xytext=(0.560, 0.827), arrowprops=arrow)
    ax.plot([0.785, 0.785], [0.70, 0.64], color="#555555", linewidth=1.0)
    ax.plot([0.255, 0.785], [0.64, 0.64], color="#555555", linewidth=1.0)
    ax.annotate("", xy=(0.255, 0.590), xytext=(0.255, 0.64), arrowprops=arrow)
    ax.annotate("", xy=(0.745, 0.590), xytext=(0.745, 0.64), arrowprops=arrow)
    ax.annotate("", xy=(0.255, 0.260), xytext=(0.255, 0.385), arrowprops=arrow)
    ax.annotate("", xy=(0.745, 0.260), xytext=(0.745, 0.385), arrowprops=arrow)

    source = pd.DataFrame(
        [
            ["athletes", 30],
            ["fixed_action_labels", 6],
            ["trials_per_cell", 50],
            ["primary_coordinate_pairs", 9000],
            ["structural_zero_affected_body_trials", 190],
            ["nominal_length_gt_200_trials", 1180],
            ["whole_athlete_bootstrap_replicates", 5000],
            ["balanced_subsamples", 1000],
        ],
        columns=["item", "count"],
    )
    source.to_csv(out / "Figure1_source_data.csv", index=False, encoding="utf-8-sig")
    save_figure(fig, out, "Figure1_study_design_and_inference_boundary", qa)


def figure2(root: Path, out: Path, qa: list[dict]) -> None:
    table_dir = root / "work/reanalysis_structural/global/tables"
    point = pd.read_csv(table_dir / "Table_05_GlobalPointwiseComponents.csv")
    scalar = pd.read_csv(table_dir / "Table_04_GlobalVarianceSummary.csv")
    bootstrap = pd.read_csv(table_dir / "Table_08_GlobalClusterBootstrap5000_Summary.csv")
    display_rows = []
    for family, group in point.groupby("curve_family"):
        group = group.sort_values("normalised_time_pct").copy()
        for column, label, _, _ in COMPONENTS:
            smooth = group[column].rolling(11, center=True, min_periods=1).mean()
            for time, value in zip(group.normalised_time_pct, smooth):
                display_rows.append(
                    {
                        "curve_family": family,
                        "normalised_time_pct": time,
                        "component": label,
                        "display_rolling_11": value,
                    }
                )
    display = pd.DataFrame(display_rows)
    point.to_csv(out / "Figure2_source_data_pointwise_raw.csv", index=False, encoding="utf-8-sig")
    display.to_csv(out / "Figure2_source_data_pointwise_display_rolling11.csv", index=False, encoding="utf-8-sig")
    scalar.to_csv(out / "Figure2_source_data_integrated.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(out / "Figure2_source_data_integrated_bootstrap.csv", index=False, encoding="utf-8-sig")

    fig = make_figure(125)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.13, 0.50, 0.37],
        left=0.14,
        right=0.985,
        bottom=0.10,
        top=0.97,
        hspace=0.46,
        wspace=0.28,
    )
    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis("off")
    handles = [
        mpl.lines.Line2D([], [], color=color, linestyle=linestyle, linewidth=1.6, label=label)
        for _, label, color, linestyle in COMPONENTS
    ]
    legend_ax.legend(handles=handles, ncol=4, frameon=False, loc="center")

    for col, (family, title) in enumerate(
        [("racket", "Racket displacement magnitude"), ("body14mean", "Aggregated body displacement magnitude")]
    ):
        ax = fig.add_subplot(gs[1, col])
        d = point[point.curve_family == family].sort_values("normalised_time_pct")
        for column, label, color, linestyle in COMPONENTS:
            y = d[column].rolling(11, center=True, min_periods=1).mean()
            ax.plot(d.normalised_time_pct, y, color=color, linestyle=linestyle, linewidth=1.45, label=label)
        ax.set(xlim=(0, 100), ylim=(0, 1), xlabel="Normalized time (%)", ylabel="Local proportion", title=title)
        ax.text(0.99, 0.96, "11-node smoothing for display only", transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color="#555555")
        clean_axes(ax, "y")
        panel_label(ax, "a" if col == 0 else "b")

    ax = fig.add_subplot(gs[2, :])
    y = np.arange(2)
    left = np.zeros(2)
    families = ["racket", "body14mean"]
    for column, label, color, _ in COMPONENTS:
        values = np.array([float(scalar.loc[scalar.curve_family == family, column].iloc[0]) for family in families])
        ax.barh(y, values, left=left, height=0.52, color=color, edgecolor="white", linewidth=0.7, label=label, zorder=2)
        for row, (start, value) in enumerate(zip(left, values)):
            pct = f"{100 * value:.1f}%"
            if value >= 0.07:
                ax.text(
                    start + value / 2,
                    row,
                    pct,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color="white" if color in {"#0072B2", "#D55E00", "#777777"} else "black",
                    zorder=3,
                )
            else:
                ax.annotate(
                    pct,
                    xy=(start + value / 2, row),
                    xytext=(start + value / 2, row - 0.34),
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.8),
                )
        left += values
    ax.set(
        xlim=(0, 1),
        xticks=np.linspace(0, 1, 6),
        xticklabels=[f"{int(x * 100)}" for x in np.linspace(0, 1, 6)],
        yticks=y,
        yticklabels=["Racket", "Aggregated body"],
        xlabel="Proportion of integrated total dispersion (%)",
    )
    ax.invert_yaxis()
    clean_axes(ax, "x")
    panel_label(ax, "c", x=-0.055)
    save_figure(fig, out, "Figure2_pointwise_dispersion_for_six_fixed_labels", qa)


def figure3(root: Path, out: Path, qa: list[dict]) -> None:
    result_dir = root / "work/reanalysis_structural/results"
    main = pd.read_csv(result_dir / "Table_R2_FullBalanced_Integrated.csv")
    ci = pd.read_csv(root / "work/reanalysis_structural/Table_S_ContinuousThresholdBootstrap5000_Summary.csv")
    ci = ci[ci.target_R_L2 == 0.90].copy()
    main["action_code"] = main.action_id.map(dict(enumerate(ACTION_CODES, start=1)))
    ci["action_code"] = ci.action_id.map(dict(enumerate(ACTION_CODES, start=1)))
    curve_rows = []
    for row in main.itertuples(index=False):
        for n in range(1, 51):
            value = row.integrated_sigma2_athlete / (
                row.integrated_sigma2_athlete + row.integrated_sigma2_trial / n
            )
            curve_rows.append(
                {
                    "waveform": row.waveform,
                    "action_id": row.action_id,
                    "action_code": row.action_code,
                    "trials": n,
                    "R_L2_independence": value,
                }
            )
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(out / "Figure3_source_data_RL2_curves.csv", index=False, encoding="utf-8-sig")
    main.to_csv(out / "Figure3_source_data_integrated_components.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(out / "Figure3_source_data_continuous_n90_bootstrap.csv", index=False, encoding="utf-8-sig")

    fig = make_figure(140)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.13, 0.43, 0.44],
        left=0.10,
        right=0.965,
        bottom=0.10,
        top=0.97,
        hspace=0.48,
        wspace=0.28,
    )
    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis("off")
    handles = [
        mpl.lines.Line2D([], [], color=ACTION_COLORS[code], linestyle=ACTION_LINESTYLES[code], linewidth=1.55, label=code)
        for code in ACTION_CODES
    ]
    legend_ax.legend(handles=handles, ncol=6, frameon=False, loc="center")

    for col, (waveform, title) in enumerate(
        [("racket", "Racket displacement magnitude"), ("body_configuration", "Aggregated body displacement magnitude")]
    ):
        ax = fig.add_subplot(gs[1, col])
        d = curves[curves.waveform == waveform]
        for code in ACTION_CODES:
            x = d[d.action_code == code]
            ax.plot(
                x.trials,
                x.R_L2_independence,
                color=ACTION_COLORS[code],
                linestyle=ACTION_LINESTYLES[code],
                linewidth=1.45,
            )
        ax.axvline(10, color="#888888", linestyle="--", linewidth=0.9)
        ax.axhline(0.80, color="#777777", linestyle="--", linewidth=0.9)
        ax.axhline(0.90, color="#222222", linestyle=":", linewidth=1.0)
        ax.text(49, 0.805, "0.80", ha="right", va="bottom", fontsize=7.2, color="#666666")
        ax.text(49, 0.905, "0.90", ha="right", va="bottom", fontsize=7.2)
        ax.text(10.7, 0.24, "n=10", rotation=90, va="bottom", fontsize=7.2, color="#666666")
        ax.set(
            # Keep the terminal tick label inside the fixed-width figure canvas.
            xlim=(1, 51.5),
            xticks=[10, 20, 30, 40, 50],
            ylim=(0.2, 1.0),
            xlabel="Included trial count n",
            ylabel=r"$R_{L^2}^{(0)}(n)$",
            title=title,
        )
        clean_axes(ax, "both")
        panel_label(ax, "a" if col == 0 else "b")

    ax = fig.add_subplot(gs[2, :])
    y = np.arange(6)
    for waveform, marker, color, offset, label in [
        ("racket", "o", "#0072B2", -0.15, "Racket"),
        ("body_configuration", "s", "#D55E00", 0.15, "Aggregated body"),
    ]:
        point = main[main.waveform == waveform].set_index("action_code").loc[ACTION_CODES]
        boot = ci[ci.waveform == waveform].set_index("action_code").loc[ACTION_CODES]
        estimate = 9.0 * point.integrated_sigma2_trial.to_numpy() / point.integrated_sigma2_athlete.to_numpy()
        low = boot.continuous_required_n_p025.to_numpy()
        high = boot.continuous_required_n_p975.to_numpy()
        ax.errorbar(
            estimate,
            y + offset,
            xerr=np.vstack([estimate - low, high - estimate]),
            fmt=marker,
            markersize=4.8,
            color=color,
            ecolor=color,
            elinewidth=1.15,
            capsize=2.6,
            label=label,
            zorder=3,
        )
        for x, yy in zip(estimate, y + offset):
            ax.annotate(
                f"{x:.1f}",
                (x, yy),
                xytext=(4, 5 if offset < 0 else -5),
                textcoords="offset points",
                va="bottom" if offset < 0 else "top",
                ha="left",
                fontsize=7.2,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.15, alpha=0.92),
            )
    ax.axvspan(0, 10, color="#009E73", alpha=0.055, zorder=0)
    ax.axvline(10, color="#777777", linestyle="--", linewidth=0.9)
    ax.set(
        xlim=(0, 47),
        xticks=[0, 10, 20, 30, 40],
        yticks=y,
        yticklabels=ACTION_CODES,
        xlabel=r"Continuous threshold $n^*$ for $R_{L^2}^{(0)}(n)\geq0.90$ (95% percentile interval)",
    )
    ax.invert_yaxis()
    clean_axes(ax, "x")
    panel_label(ax, "c", x=-0.055)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    save_figure(fig, out, "Figure3_action_specific_relative_reliability", qa)


def figure4(root: Path, out: Path, qa: list[dict]) -> None:
    base = root / "work/reanalysis_structural/assumption_influence"
    acf = pd.read_csv(base / "Table_S_ArchiveOrderResidualACF.csv")
    ar = pd.read_csv(base / "Table_S_AR1CorrelationScenarios.csv")
    registration = pd.read_csv(base / "Table_S_PeakRegistrationSensitivity.csv")
    influence = pd.read_csv(base / "Table_S_BackhandDriveInfluenceAndFlags.csv")
    acf.to_csv(out / "Figure4_source_data_archive_order_ACF.csv", index=False, encoding="utf-8-sig")
    ar.to_csv(out / "Figure4_source_data_AR1_scenarios.csv", index=False, encoding="utf-8-sig")
    registration.to_csv(out / "Figure4_source_data_registration.csv", index=False, encoding="utf-8-sig")
    influence.to_csv(out / "Figure4_source_data_BD_influence.csv", index=False, encoding="utf-8-sig")

    fig = make_figure(151)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.115,
        right=0.925,
        bottom=0.085,
        top=0.925,
        hspace=0.44,
        wspace=0.48,
        height_ratios=[0.88, 1.12],
    )

    ax = fig.add_subplot(gs[0, 0])
    lag1 = acf[acf.archive_order_lag == 1]
    null_low = lag1.lag1_permutation_null_p025.min()
    null_high = lag1.lag1_permutation_null_p975.max()
    ax.axhspan(null_low, null_high, color="#D9D9D9", alpha=0.8, label="Random-order reference range")
    x = np.arange(6)
    for waveform, offset, marker, color, label in [
        ("racket", -0.10, "o", "#0072B2", "Racket"),
        ("body_configuration", 0.10, "s", "#D55E00", "Aggregated body"),
    ]:
        d = lag1[lag1.waveform == waveform].set_index("action_code").loc[ACTION_CODES]
        ax.plot(x + offset, d.rho_L2_trace, marker=marker, color=color, linewidth=1.05, markersize=4.3, label=label)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set(
        xticks=x,
        xticklabels=ACTION_CODES,
        ylim=(-0.08, 0.37),
        ylabel=r"Lag-1 $L^2$ trace correlation in archive order",
        title="File-index proxy (not verified acquisition order)",
    )
    ax.set_yticks([-0.05, 0.00, 0.10, 0.20, 0.30])
    clean_axes(ax, "y")
    panel_label(ax, "a", x=-0.18)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), handlelength=1.6, columnspacing=0.9)

    ax = fig.add_subplot(gs[0, 1])
    fixed = ar[ar.scenario == "fixed"].copy()
    row_labels = [f"R–{code}" for code in ACTION_CODES] + [f"H–{code}" for code in ACTION_CODES]
    matrix = np.empty((12, 4))
    for i, (waveform, prefix) in enumerate([("racket", "R"), ("body_configuration", "H")]):
        for j, code in enumerate(ACTION_CODES):
            d = fixed[(fixed.waveform == waveform) & (fixed.action_code == code)].sort_values("AR1_phi")
            matrix[i * 6 + j] = d.required_n_R_L2_90.to_numpy()
    cmap = LinearSegmentedColormap.from_list("trial_counts", ["#F3F8FC", "#86B6D8", "#D55E00"])
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=4, vmax=49)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            ax.text(col, row, str(value), ha="center", va="center", fontsize=7.2, color="white" if value >= 28 else "black")
    ax.set(
        xticks=np.arange(4),
        xticklabels=["0", "0.10", "0.20", "0.30"],
        yticks=np.arange(12),
        yticklabels=row_labels,
        xlabel=r"AR(1) working scenario $\phi$",
        title=r"Integer trials for $R_{L^2}^{(\phi)}(n)\geq0.90$",
    )
    ax.axhline(5.5, color="white", linewidth=1.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    panel_label(ax, "b", x=-0.23)
    cbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.025)
    cbar.set_label("Trials", fontsize=7.2)
    cbar.ax.tick_params(labelsize=7.2)

    ax = fig.add_subplot(gs[1, 0])
    rows = []
    for waveform, prefix in [("racket", "R"), ("body_configuration", "H")]:
        for code in ACTION_CODES:
            d = registration[(registration.waveform == waveform) & (registration.action_code == code)]
            unreg = float(d.loc[d.registration_rule == "unregistered", "continuous_n90"].iloc[0])
            reg = float(d.loc[d.registration_rule == "robust_racket_peak_registered", "continuous_n90"].iloc[0])
            rows.append((f"{prefix}–{code}", unreg, reg))
    y = np.arange(12)
    for yy, (_, unreg, reg) in zip(y, rows):
        ax.plot([unreg, reg], [yy, yy], color="#B5B5B5", linewidth=1.1, zorder=1)
    ax.scatter([x[1] for x in rows], y, marker="o", s=23, color="#222222", label="Unregistered", zorder=3)
    ax.scatter([x[2] for x in rows], y, marker="D", s=22, facecolor="white", edgecolor="#009E73", linewidth=1.1, label="Robust racket-peak registration", zorder=3)
    ax.set(
        yticks=y,
        yticklabels=[x[0] for x in rows],
        xlim=(0, 35),
        xlabel=r"Continuous threshold $n^*$",
        title="Estimand sensitivity to one proxy peak registration",
    )
    ax.invert_yaxis()
    clean_axes(ax, "x")
    panel_label(ax, "c", x=-0.18)
    ax.legend(frameon=False, ncol=1, loc="lower right", handlelength=1.2)

    ax = fig.add_subplot(gs[1, 1])
    influence = influence.sort_values("participant_code")
    participant = influence.participant_code.to_numpy()
    main_y = influence.continuous_n90_unaltered_nonzero_primary.to_numpy()
    hampel_y = influence.continuous_n90_hampel_type_local_high_value.to_numpy()
    sizes = 17 + 1.1 * influence.displacement_values_flagged.to_numpy()
    for xx, y1, y2 in zip(participant, main_y, hampel_y):
        ax.plot([xx, xx], [y1, y2], color="#C7C7C7", linewidth=0.75, zorder=1)
    ax.scatter(participant, main_y, s=sizes, color="#222222", marker="o", label="Primary rule", zorder=3)
    ax.scatter(participant, hampel_y, s=sizes, facecolor="white", edgecolor="#D55E00", linewidth=1.0, marker="D", label="Hampel-type rule", zorder=3)
    ax.axhline(26.322566, color="#222222", linestyle=":", linewidth=0.9)
    ax.axhline(17.961616, color="#D55E00", linestyle=":", linewidth=0.9)
    ax.set(
        xlim=(0.3, 30.7),
        xticks=[1, 5, 10, 15, 20, 25, 30],
        ylim=(14.5, 31.5),
        xlabel="Athlete code omitted in turn",
        ylabel=r"BD racket continuous threshold $n^*$",
        title="Quality-rule difference is distributed",
    )
    clean_axes(ax, "y")
    panel_label(ax, "d")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.text(0.02, 0.04, "Marker area scales with flagged values for that athlete", transform=ax.transAxes, fontsize=7.2, color="#555555")

    save_figure(fig, out, "Figure4_dependence_registration_and_quality_sensitivity", qa)


def supplementary_figure_s1(root: Path, out: Path, qa: list[dict]) -> None:
    result_dir = root / "work/reanalysis_structural/results"
    main = pd.read_csv(result_dir / "Table_R2_FullBalanced_Integrated.csv")
    excluded = pd.read_csv(result_dir / "Table_R8_Excluded_UnbalancedREML_Integrated.csv")
    balanced = pd.read_csv(result_dir / "Table_R11_BalancedSubsample1000_Percentiles.csv")
    hampel = pd.read_csv(root / "work/reanalysis_hampel/results/Table_R2_FullBalanced_Integrated.csv")
    for frame in (main, excluded, hampel):
        frame["action_code"] = frame.action_id.map(dict(enumerate(ACTION_CODES, start=1)))
    source = main[["waveform", "action_id", "action_code", "R_L2_m50", "required_n_R_L2_90"]].merge(
        excluded[["waveform", "action_id", "R_L2_m50", "required_n_R_L2_90"]],
        on=["waveform", "action_id"],
        suffixes=("_main", "_exclude_gt200"),
    ).merge(
        hampel[["waveform", "action_id", "R_L2_m50", "required_n_R_L2_90"]],
        on=["waveform", "action_id"],
        suffixes=("", "_hampel"),
    )
    balanced50 = balanced[balanced.metric == "R_L2_m50"][[
        "waveform", "action_id", "percentile_2_5_finite", "percentile_50_finite", "percentile_97_5_finite"
    ]]
    source = source.merge(balanced50, on=["waveform", "action_id"])
    source.to_csv(out / "FigureS1_source_data_archive_robustness.csv", index=False, encoding="utf-8-sig")

    fig = make_figure(126)
    gs = fig.add_gridspec(2, 2, left=0.11, right=0.985, bottom=0.10, top=0.96, hspace=0.38, wspace=0.30)
    for row, (waveform, row_label) in enumerate([("racket", "Racket"), ("body_configuration", "Aggregated body")]):
        d = source[source.waveform == waveform].set_index("action_code").loc[ACTION_CODES]
        y = np.arange(6)
        ax = fig.add_subplot(gs[row, 0])
        ax.scatter(d.R_L2_m50_main, y - 0.16, marker="o", color="#222222", s=20, label="Complete design")
        ax.scatter(d.R_L2_m50_exclude_gt200, y, marker="s", color="#0072B2", s=20, label="Exclude nominal length >200")
        low = d.percentile_50_finite - d.percentile_2_5_finite
        high = d.percentile_97_5_finite - d.percentile_50_finite
        ax.errorbar(
            d.percentile_50_finite,
            y + 0.16,
            xerr=np.vstack([low, high]),
            fmt="^",
            markersize=3.8,
            color="#009E73",
            capsize=2.2,
            linewidth=1.0,
            label="Balanced subsampling",
        )
        ax.set(yticks=y, yticklabels=ACTION_CODES, xlim=(0.92, 1.001), xlabel=r"$R_{L^2}^{(0)}(50)$", title=row_label)
        ax.invert_yaxis()
        clean_axes(ax, "x")
        panel_label(ax, "a" if row == 0 else "c", x=-0.15)
        if row == 0:
            ax.legend(frameon=False, ncol=1, loc="lower left")

        ax = fig.add_subplot(gs[row, 1])
        primary = d.required_n_R_L2_90_main.to_numpy()
        robust = d.required_n_R_L2_90.to_numpy()
        for yy, x1, x2 in zip(y, primary, robust):
            ax.plot([x1, x2], [yy - 0.10, yy + 0.10], color="#B5B5B5", linewidth=1.0)
        ax.scatter(primary, y - 0.10, marker="o", color="#222222", s=21, label="Primary rule")
        ax.scatter(robust, y + 0.10, marker="D", facecolor="white", edgecolor="#D55E00", linewidth=1.0, s=21, label="Hampel-type rule")
        ax.set(yticks=y, yticklabels=ACTION_CODES, xlim=(0, 30), xlabel="Integer trials required for 0.90", title=row_label)
        ax.invert_yaxis()
        clean_axes(ax, "x")
        panel_label(ax, "b" if row == 0 else "d", x=-0.15)
        if row == 0:
            ax.legend(frameon=False, ncol=2, loc="lower right")
    save_figure(fig, out, "FigureS1_archive_boundary_and_resampling_robustness", qa)


def explode_hampel_flags(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in audit.itertuples(index=False):
        steps = str(record.racket_hampel_flag_steps_0based).split(";") if pd.notna(record.racket_hampel_flag_steps_0based) else []
        times = str(record.racket_hampel_flag_time_pct).split(";") if pd.notna(record.racket_hampel_flag_time_pct) else []
        steps = [x for x in steps if x and x.lower() != "nan"]
        times = [x for x in times if x and x.lower() != "nan"]
        if len(steps) != len(times):
            raise ValueError("Hampel flag step/time fields are inconsistent")
        for step, time in zip(steps, times):
            rows.append(
                {
                    "participant_code": int(record.participant_code),
                    "action_id": int(record.action_id),
                    "action_code": ACTION_CODES[int(record.action_id) - 1],
                    "trial_in_cell": int(record.trial_in_cell),
                    "raw_step_0based": int(step),
                    "raw_step_time_pct": float(time),
                    "global_index": int(record.global_index),
                }
            )
    return pd.DataFrame(rows)


def supplementary_figure_s2(root: Path, out: Path, qa: list[dict]) -> None:
    audit_candidates = [
        root / "work/hampel_arrays/structural_missingness_audit_9000.csv",
        # Backwards-compatible name used by the development workspace.
        root / "work/hampel_sensitivity_arrays/structural_missingness_audit_9000.csv",
    ]
    audit_path = next((path for path in audit_candidates if path.is_file()), None)
    if audit_path is None:
        raise FileNotFoundError(
            "Hampel-type audit not found; checked: "
            + ", ".join(str(path) for path in audit_candidates)
        )
    audit = pd.read_csv(audit_path)
    flags = explode_hampel_flags(audit)
    flags.to_csv(out / "FigureS2_source_data_racket_hampel_flags.csv", index=False, encoding="utf-8-sig")
    fig = make_figure(126)
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.985, bottom=0.10, top=0.96, hspace=0.42, wspace=0.32)

    player_action = flags.groupby(["participant_code", "action_code"]).size().unstack(fill_value=0).reindex(columns=ACTION_CODES)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(player_action.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
    ax.set(xticks=np.arange(6), xticklabels=ACTION_CODES, yticks=[0, 4, 9, 14, 19, 24, 29], yticklabels=[1, 5, 10, 15, 20, 25, 30], xlabel="Fixed label", ylabel="Athlete code", title="Flagged racket values by athlete × action")
    panel_label(ax, "a", x=-0.17)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=7.2)

    bd = flags[flags.action_code == "BD"]
    trial_matrix = bd.groupby(["participant_code", "trial_in_cell"]).size().unstack(fill_value=0).reindex(index=np.arange(1, 31), columns=np.arange(1, 51), fill_value=0)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(trial_matrix.to_numpy(), aspect="auto", cmap="Oranges", vmin=0, vmax=max(1, trial_matrix.to_numpy().max()))
    ax.set(xticks=[0, 9, 19, 29, 39, 49], xticklabels=[1, 10, 20, 30, 40, 50], yticks=[0, 4, 9, 14, 19, 24, 29], yticklabels=[1, 5, 10, 15, 20, 25, 30], xlabel="BD archive trial rank", ylabel="Athlete code", title="BD flags occur in 292/1,500 trials")
    panel_label(ax, "b", x=-0.17)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(bd.raw_step_time_pct, bins=np.linspace(0, 100, 21), color="#D55E00", edgecolor="white", linewidth=0.7)
    ax.axvspan(40, 60, color="#0072B2", alpha=0.08)
    ax.set(xlim=(0, 100), xlabel="Position in observed segment (%)", ylabel="Flagged values", title="Temporal location of BD flags")
    clean_axes(ax, "y")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    counts = flags.groupby("action_code").size().reindex(ACTION_CODES, fill_value=0)
    bars = ax.bar(ACTION_CODES, counts.to_numpy(), color=[ACTION_COLORS[x] for x in ACTION_CODES], width=0.65)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 5, str(int(value)), ha="center", va="bottom", fontsize=7.2)
    ax.set(ylim=(0, max(counts) * 1.18), xlabel="Fixed label", ylabel="Flagged values", title="BD accounts for 73.0% of racket flags")
    clean_axes(ax, "y")
    panel_label(ax, "d")
    save_figure(fig, out, "FigureS2_distribution_of_Hampel_type_flags", qa)


def supplementary_figure_s3(root: Path, out: Path, qa: list[dict]) -> None:
    table_dir = root / "work/reanalysis_structural/global/tables"
    point = pd.read_csv(table_dir / "Table_05_GlobalPointwiseComponents.csv")
    bands = pd.read_csv(table_dir / "Table_S_GlobalPointwiseBootstrapBands.csv")
    bands.to_csv(out / "FigureS3_source_data_pointwise_bootstrap_bands.csv", index=False, encoding="utf-8-sig")
    fig = make_figure(91)
    gs = fig.add_gridspec(2, 4, left=0.08, right=0.965, bottom=0.13, top=0.94, hspace=0.35, wspace=0.28)
    component_lookup = {
        "action": ("omega_action", "Fixed-action contrast dispersion", "#0072B2"),
        "athlete": ("omega_athlete", "Athlete", "#B58900"),
        "action_x_athlete": ("omega_action_x_athlete", "Athlete × fixed action", "#D55E00"),
        "trial_residual": ("omega_trial_residual", "Within-cell unexplained", "#777777"),
    }
    for row, (family, row_label) in enumerate([("racket", "Racket"), ("body14mean", "Aggregated body")]):
        for col, component in enumerate(["action", "athlete", "action_x_athlete", "trial_residual"]):
            column, title, color = component_lookup[component]
            ax = fig.add_subplot(gs[row, col])
            p = point[point.curve_family == family].sort_values("normalised_time_pct")
            b = bands[(bands.curve_family == family) & (bands.component == component)].sort_values("normalised_time_pct")
            ax.fill_between(b.normalised_time_pct, b.pointwise_p025, b.pointwise_p975, color=color, alpha=0.18, linewidth=0)
            ax.plot(p.normalised_time_pct, p[column], color=color, linewidth=1.05)
            ax.set(xlim=(0, 100), ylim=(0, 1), xlabel="Time (%)" if row == 1 else "", ylabel="Proportion" if col == 0 else "", title=title if row == 0 else "")
            clean_axes(ax, "y")
            if col == 0:
                ax.text(-0.38, 0.5, row_label, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=7.4, fontweight="bold")
            if row == 0:
                ax.text(
                    0.02,
                    0.96,
                    chr(ord("a") + col),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9.0,
                    fontweight="bold",
                )
    save_figure(fig, out, "FigureS3_pointwise_marginal_bootstrap_intervals", qa)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    family = configure_style()
    qa: list[dict] = []
    figure1(out, qa)
    figure2(root, out, qa)
    figure3(root, out, qa)
    figure4(root, out, qa)
    supplementary_figure_s1(root, out, qa)
    supplementary_figure_s2(root, out, qa)
    supplementary_figure_s3(root, out, qa)
    qa_df = pd.DataFrame(qa)
    qa_df.to_csv(out / "Figure_output_manifest.csv", index=False, encoding="utf-8-sig")
    pass_columns = [
        "fixed_width_pass",
        "tiff_dimension_pass",
        "tiff_lzw_pass",
        "tiff_600dpi_pass",
        "svg_editable_text_pass",
    ]
    summary = {
        "backend": "Python/matplotlib",
        "font_family": family,
        "figure_width_mm": FIG_WIDTH_MM,
        "figures": len(qa_df),
        "hard_checks_pass": bool(qa_df[pass_columns].all().all()),
        "text_boundary_violations_total": int(qa_df.text_boundary_violation_count.sum()),
        "manifest": "Figure_output_manifest.csv",
    }
    (out / "Figure_QA.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not summary["hard_checks_pass"]:
        raise RuntimeError(f"Figure hard QA failed: {summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
