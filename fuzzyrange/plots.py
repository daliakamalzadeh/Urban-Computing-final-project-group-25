"""Figures for the report (matplotlib, saved as PNG).

Every metric keeps the same colour and legend name in every figure:
    point-wise                       grey, dotted (classical baseline)
    range-based                      blue          (original metric of the paper)
    uncertainty-aware, linear        orange
    uncertainty-aware, slow exp.     aqua
    uncertainty-aware, fast exp.     yellow
    uncertainty-aware, step          magenta
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import WEIGHT_FUNCTIONS, membership, tolerance  # noqa: E402

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY = "#8a8984"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SIGMA_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # light -> dark = small -> large sigma

WEIGHT_STYLE = {   # weight function -> (legend name, colour)
    "linear": ("linear decay", ORANGE),
    "exponential": ("slow exponential decay", AQUA),
    "exp-fast": ("fast exponential decay", YELLOW),
    "step": ("step (no decay)", MAGENTA),
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 10.5,
    "legend.frameon": False,
})


def _metric_style(name: str) -> tuple[str, str, str]:
    """Metric name -> (legend label, colour, line style)."""
    if name == "point-wise":
        return "point-wise (classical)", GREY, ":"
    if name == "range-based":
        return "range-based (original)", BLUE, "-"
    m = re.match(r"UA-(\S+) \(tau=(.+) of L\)", name)
    if m:
        label, colour = WEIGHT_STYLE[m.group(1)]
        return f"uncertainty-aware, {label}", colour, "-"
    return name, INK_2, "-"


def _finish(fig, path: Path, title: str, subtitle: str | None = None, legend_handles=None,
            ncol: int = 3) -> None:
    """Title + one-line explanation at the top, legend underneath, plots below that."""
    h = fig.get_figheight()
    y = 1 - 0.08 / h                                   # header layout measured in inches
    fig.text(0.01, y, title, fontsize=12.5, fontweight="bold", ha="left", va="top")
    y -= 0.32 / h
    if subtitle:
        fig.text(0.01, y, subtitle, ha="left", va="top", color=INK_2, fontsize=10)
        y -= 0.28 / h
    if legend_handles:
        rows = int(np.ceil(len(legend_handles) / ncol))
        fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.005, y),
                   ncol=ncol, fontsize=9, handlelength=2.5)
        y -= (0.1 + 0.22 * rows) / h
    fig.tight_layout(rect=(0, 0, 1, y))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metric_lines(ax, df: pd.DataFrame, x: str, y: str, names: list[str]):
    handles = []
    for nm in names:
        g = df[df["metric"] == nm].sort_values(x)
        label, colour, ls = _metric_style(nm)
        (h,) = ax.plot(g[x], g[y], color=colour, ls=ls, label=label)
        handles.append(h)
    return handles


# --------------------------------------------------------------------------- #
# Fig. 1 - weight functions
# --------------------------------------------------------------------------- #
def plot_weight_functions(path: Path, r: tuple[int, int], tau_frac: float = 0.05) -> None:
    """Credit a predicted point receives at distance d outside a labelled anomaly."""
    L = r[1] - r[0] + 1
    tau = tolerance(r, tau_frac)
    d = np.arange(0, tau + 4)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    handles = []
    for w, (label, colour) in WEIGHT_STYLE.items():
        if w not in WEIGHT_FUNCTIONS:
            continue
        y = np.where(d == 0, 1.0,
                     np.where(d <= tau, WEIGHT_FUNCTIONS[w](d.astype(float).clip(1), tau), 0.0))
        (h,) = ax.step(d, y, where="mid", color=colour, label=label)
        handles.append(h)
    ax.axvline(tau + 0.5, color=INK_2, ls="--", lw=1)
    ax.text(tau + 0.8, 0.55, f"end of tolerance zone\nτ = {tau} steps\n({tau_frac * 100:g}% of {L} steps)",
            color=INK_2, fontsize=9, va="center")
    ax.set_xlabel("distance of the predicted point from the labelled start or end (time steps)")
    ax.set_ylabel("credit for this point (1 = full)")
    ax.set_ylim(-0.03, 1.08)
    _finish(fig, path, "Partial credit for predicted points just outside a labelled anomaly",
            handles, ncol=4)


# --------------------------------------------------------------------------- #
# Fig. 2 - example anomaly with the prediction scenarios
# --------------------------------------------------------------------------- #
def plot_example(path: Path, dataset, predictions: dict, anomaly_idx: int = 4,
                 tau_frac: float = 0.05, weight: str = "linear", pad: int = 150) -> None:
    """One labelled anomaly of the series, its tolerance zones and the six simulated predictions."""
    s, e = dataset.anomalies[anomaly_idx]
    lo, hi = max(0, s - pad), min(dataset.n - 1, e + pad)
    t = np.arange(lo, hi + 1)
    tau = tolerance((s, e), tau_frac)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1.5]})
    ax.plot(t, dataset.values[lo:hi + 1], color=INK_2, lw=1.2)
    h1 = ax.axvspan(s, e, color=BLUE, alpha=0.15, lw=0, label="labelled anomaly")
    h2 = ax.axvspan(s - tau, s, color=ORANGE, alpha=0.35, lw=0,
                    label=f"tolerance zones (τ = {tau} steps = {tau / 2:g} h): partial credit")
    ax.axvspan(e, e + tau, color=ORANGE, alpha=0.35, lw=0)
    ax.set_ylabel("taxi passengers per 30 min")
    ax.set_title(f"Taxi demand around anomaly {anomaly_idx + 1}", loc="left")

    names = list(predictions)
    for k, nm in enumerate(names):
        for ps, pe in predictions[nm]:
            if pe >= lo and ps <= hi:
                ax2.plot([max(ps, lo), min(pe, hi)], [k, k], color=INK_2, lw=7,
                         solid_capstyle="butt")
    ax2.axvspan(s, e, color=BLUE, alpha=0.15, lw=0)
    ax2.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax2.invert_yaxis()
    ax2.set_xlabel("time step (one step = 30 min)")
    ax2.set_title("Simulated predictions (grey bars = predicted anomaly)", loc="left")
    ax2.grid(axis="y", visible=False)
    _finish(fig, path,
            f"Example: labelled anomaly {anomaly_idx + 1} "
            f"({dataset.timestamps[s]:%d %b} – {dataset.timestamps[e]:%d %b %Y}) and the six test predictions",
            None, [h1, h2], ncol=2)


# --------------------------------------------------------------------------- #
# Fig. 3 - robustness to shifted labels
# --------------------------------------------------------------------------- #
def plot_robustness(path: Path, per_mag: pd.DataFrame) -> None:
    """Average |score on original labels - score on shifted labels| per shift size."""
    names = list(dict.fromkeys(per_mag["metric"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, col, title in zip(axes, ["diff_precision", "diff_recall", "diff_f"],
                              ["Precision", "Recall", "F1 score"]):
        handles = _metric_lines(ax, per_mag, "magnitude", col, names)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(sorted(per_mag["magnitude"].unique()))
        ax.set_xlabel("labelled start/end moved by (time steps)")
    axes[0].set_ylabel("difference in score\n(original vs moved labels)")
    _finish(fig, path,
            "How much do the scores change when the labelled anomaly boundaries are moved?",
            handles, ncol=3)


# --------------------------------------------------------------------------- #
# Fig. 4 - discrimination on the original labels
# --------------------------------------------------------------------------- #
def plot_discrimination(path: Path, disc: pd.DataFrame) -> None:
    names = list(dict.fromkeys(disc["metric"]))
    scen = list(dict.fromkeys(disc["scenario"]))
    x = np.arange(len(scen))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(12, 5))
    handles = []
    for i, nm in enumerate(names):
        g = disc[disc["metric"] == nm].set_index("scenario").reindex(scen)
        label, colour, _ = _metric_style(nm)
        handles.append(ax.bar(x + (i - (len(names) - 1) / 2) * width, g["f"], width * 0.9,
                              color=colour, label=label))
    ax.set_xticks(x, [s.replace("_", "\n") for s in scen])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    _finish(fig, path, "F1 score of each simulated prediction (scored on the original labels)",
            handles, ncol=3)


# --------------------------------------------------------------------------- #
# Fig. 5 - choosing tau: stability vs discrimination
# --------------------------------------------------------------------------- #
def plot_tradeoff(path: Path, sweep: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """Left: stability, right: discrimination, both against the tolerance size tau."""
    ua = sweep[sweep["family"] == "uncertainty-aware"]
    fracs = sorted(ua["tau_frac"].unique())
    x = np.arange(len(fracs))
    panels = [("mean_dF_small_shifts",
               "Stability: change in F1 when labels move 2–5 steps",
               "difference in F1"),
              ("gap_small_vs_large",
               "Discrimination: F1(small error) − F1(large error)",
               "difference in F1")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    handles = []
    for ax, (col, title, ylab) in zip(axes, panels):
        handles = []
        for w, (label, colour) in WEIGHT_STYLE.items():
            g = ua[ua["weight"] == w].set_index("tau_frac").reindex(fracs)
            if g[col].isna().all():
                continue
            (h,) = ax.plot(x, g[col], color=colour, label=f"uncertainty-aware, {label}")
            handles.append(h)
        for _, r in baselines.iterrows():
            label, colour, ls = _metric_style(r["metric"])
            handles.append(ax.axhline(r[col], color=colour, ls="--" if ls == "-" else ls, lw=1.5,
                                      label=label))
        ax.set_xticks(x, [f"{f * 100:g}%" for f in fracs])
        ax.set_xlabel("tolerance τ (% of anomaly length)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
    _finish(fig, path, "Comparing different tolerances τ",
            handles, ncol=3)


# --------------------------------------------------------------------------- #
# Fig. 6 - ranking stability
# --------------------------------------------------------------------------- #
def plot_ranking_stability(path: Path, stab: pd.DataFrame) -> None:
    agg = stab.groupby(["metric", "magnitude"], sort=False)["kendall_tau"].mean().reset_index()
    names = list(dict.fromkeys(agg["metric"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    handles = _metric_lines(ax, agg, "magnitude", "kendall_tau", names)
    ax.set_xticks(sorted(agg["magnitude"].unique()))
    ax.set_xlabel("labelled start/end moved by (time steps)")
    ax.set_ylabel("Kendall rank correlation")
    _finish(fig, path, "Effect of slightly moving the labels on the ranking of the six predictions",
            handles, ncol=2)


# --------------------------------------------------------------------------- #
# Fig. 7 - simulated annotators
# --------------------------------------------------------------------------- #
def plot_annotators(path: Path, summary: pd.DataFrame, names: list[str]) -> None:
    """Spread, bias, RMSE and ordering accuracy against annotator disagreement sigma."""
    df = summary.assign(ordering_pct=summary["ordering"] * 100)
    panels = [("spread", "Spread across annotators", "std. dev. of F1"),
              ("bias", "Bias: average F1 − deserved F1", "> 0 too generous, < 0 too harsh"),
              ("rmse", "Overall error (RMSE)", "distance to deserved F1 "),
              ("ordering_pct", "Correctly ordered prediction pairs", "% of pairs")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (col, title, ylab) in zip(axes, panels):
        handles = _metric_lines(ax, df, "sigma", col, names)
        if col == "bias":
            ax.axhline(0, color=INK_2, lw=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_xticks(sorted(df["sigma"].unique()))
        ax.set_xlabel("annotator disagreement σ (time steps)")
    _finish(fig, path, "Scoring against 20 simulated annotators",
            handles, ncol=2)


# --------------------------------------------------------------------------- #
# Fig. 8 - best tau per annotator disagreement
# --------------------------------------------------------------------------- #
def plot_tau_choice(path: Path, summary: pd.DataFrame) -> None:
    """RMSE relative to the original metric for every tau; one line per sigma."""
    base = summary[summary["family"] == "range-based"].set_index("sigma")["rmse"]
    ua = summary[
        (summary["family"] == "uncertainty-aware")
        & (summary["tau_frac"] <= 0.05)
    ]
    weights = [w for w in WEIGHT_STYLE if w in set(ua["weight"])]
    fracs = [0.0] + sorted(ua["tau_frac"].unique())
    x = np.arange(len(fracs))
    sigmas = sorted(summary["sigma"].unique())
    fig, axes = plt.subplots(1, len(weights), figsize=(4.2 * len(weights), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, weights):
        handles = []
        for i, sg in enumerate(sigmas):
            g = ua[(ua["weight"] == w) & (ua["sigma"] == sg)].set_index("tau_frac")["rmse"]
            rel = np.array([1.0] + [g[f] / base[sg] for f in fracs[1:]])
            colour = SIGMA_RAMP[i % len(SIGMA_RAMP)]
            (h,) = ax.plot(x, rel, color=colour, label=f"σ = {sg:g} steps")
            handles.append(h)
            k = int(np.argmin(rel))
            ax.scatter(x[k], rel[k], s=90, facecolors="white", edgecolors=colour, lw=2, zorder=3)
        ax.axhline(1, color=INK_2, lw=1, ls="--")
        ax.set_yscale("log")
        ticks = [0.8, 1, 1.5, 2, 5, 10, 20]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xticks(x, ["0" if f == 0 else f"{f * 100:g}%" for f in fracs], fontsize=8.5)
        ax.set_xlabel("tolerance τ (% of anomaly length)")
        ax.set_title(WEIGHT_STYLE[w][0], loc="left", fontweight="bold")
    axes[0].set_ylabel("error relative to original metric\n(below 1 = more accurate)")
    _finish(fig, path, "Which tolerance τ gives the most accurate score with annotator disagreement?",
            handles, ncol=4)
