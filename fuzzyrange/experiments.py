"""
The experiments:

RQ1/RQ2  Robustness:
how much does each metric change when the ground-truth boundaries are shifted?
RQ3 Discrimination:
does the metric still rank good predictions above bad ones?
Trade-off:            
Sweep tau and the weight function to see how much robustness is bought at what cost in discrimination.
Annotators: 
Score against 20 simulated annotators and compare with the TRUE labels: spread, bias,RMSE and ordering accuracy.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .metrics import Metric, Range, build_metrics
from .scenarios import (PERTURBATION_MODES, SCENARIO_QUALITY_RANK, perturb_labels,
                        simulate_annotators)

MAGNITUDES = (0, 2, 5, 10, 20) #to shift anomaly ranges
TAU_FRACS = (0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2)   # tolerance as a fraction of L
SIGMAS = (2, 5, 10, 20)                               # annotator disagreement (steps)


def run_robustness(real: list[Range], n: int, predictions: dict[str, list[Range]],
                   metrics: Sequence[Metric], magnitudes: Sequence[int] = MAGNITUDES,
                   modes: Sequence[str] = PERTURBATION_MODES) -> pd.DataFrame:
    """Score predictions against fixed and deterministically perturbed labels."""

    #create perturbed labels
    label_sets = []
    for m in magnitudes:
        for mode in modes:
            label_sets.append((m, mode, perturb_labels(real, n, m, mode)))


    #score
    rows = []
    for metric in metrics:
        for scen, pred in predictions.items():
            base = metric(real, pred, n)
            for m, mode, labels in label_sets:
                s = metric(labels, pred, n)
                rows.append({
                    "metric": metric.name, "family": metric.family,
                    "tau_frac": metric.tau_frac, "weight": metric.weight,
                    "scenario": scen, "magnitude": m, "mode": mode,
                    "precision": s.precision, "recall": s.recall, "f": s.f,
                    "diff_precision": abs(base.precision - s.precision),
                    "diff_recall": abs(base.recall - s.recall),
                    "diff_f": abs(base.f - s.f),
                })
    return pd.DataFrame(rows)


def run_discrimination(real: list[Range], n: int, predictions: dict[str, list[Range]],
                       metrics: Sequence[Metric]) -> pd.DataFrame:
    """Scores of every scenario on the original labels."""
    rows = []
    for metric in metrics:
        for scen, pred in predictions.items():
            s = metric(real, pred, n)
            rows.append({"metric": metric.name, "family": metric.family,
                         "tau_frac": metric.tau_frac, "weight": metric.weight,
                         "scenario": scen, **s.as_dict()})
    return pd.DataFrame(rows)


def ranking_stability(robust: pd.DataFrame) -> pd.DataFrame:
    """Check whether the ranking of scenarios remain the same when the labels are perturbed using Kendall tau score.
        - 1.0 means the shifted labels do not change the ranking
        - 0.0 means the shifted labels completely change the ranking.
    """
    scen = [s for s in SCENARIO_QUALITY_RANK if s in set(robust["scenario"])]
    df = robust[robust["scenario"].isin(scen)]
    ref = (df[df["magnitude"] == 0].groupby(["metric", "scenario"])["f"].first())
    rows = []
    for (metric, m, mode), g in df.groupby(["metric", "magnitude", "mode"], sort=False):
        f_pert = g.set_index("scenario")["f"].reindex(scen)
        f_ref = ref.loc[metric].reindex(scen)
        kt = kendalltau(f_ref.values, f_pert.values).statistic
        rows.append({"metric": metric, "magnitude": m, "mode": mode,
                     "kendall_tau": 1.0 if np.isnan(kt) else kt})
    return pd.DataFrame(rows)


def summarize_discrimination(disc: pd.DataFrame) -> pd.DataFrame:
    """Per metric calculate Spearman correlation of the rankings."""
    rows = []
    for (metric,), g in disc.groupby(["metric"], sort=False):
        g = g.set_index("scenario")
        scen = [s for s in SCENARIO_QUALITY_RANK if s in g.index]
        f = g.loc[scen, "f"].values
        expected_quality = [-SCENARIO_QUALITY_RANK[s] for s in scen]   # higher = better
        rho = spearmanr(f, expected_quality).statistic
        rows.append({
            "metric": metric,
            "spearman_rho": rho,
            "gap_exact_vs_missed": g.loc["exact", "f"] - g.loc["mostly_missed", "f"],
            "gap_small_vs_large": (g.loc["small_boundary_error", "f"]
                                   - g.loc["large_boundary_error", "f"]),
            "f_exact": g.loc["exact", "f"],
            "f_mostly_missed": g.loc["mostly_missed", "f"],
        })
    return pd.DataFrame(rows)


def summarize_robustness(robust: pd.DataFrame, small: Sequence[int] = (2, 5)) -> pd.DataFrame:
    """How much do the metric scores change with small label perturbations"""
    scen = list(SCENARIO_QUALITY_RANK)
    df = robust[robust["scenario"].isin(scen)]
    per_mag = (df.groupby(["metric", "magnitude"], sort=False)[
        ["diff_precision", "diff_recall", "diff_f"]
    ]
               .mean().reset_index())
    small_df = (df[df["magnitude"].isin(small)].groupby("metric", sort=False)["diff_f"]
                .mean().rename("mean_dF_small_shifts").reset_index())
    large_df = (df[df["magnitude"] == max(df["magnitude"])].groupby("metric", sort=False)["diff_f"]
                .mean().rename("mean_dF_largest_shift").reset_index())
    return per_mag, small_df.merge(large_df, on="metric")


def run_tau_sweep(real: list[Range], n: int, predictions: dict[str, list[Range]],
                  tau_fracs: Sequence[float] = TAU_FRACS,
                  weights: Sequence[str] = ("linear", "exponential", "exp-fast", "step"),
                  metric_kwargs: dict | None = None, **robust_kwargs) -> pd.DataFrame:
    """Robustness vs discrimination for every (weight, tau_frac) combination."""
    metric_kwargs = metric_kwargs or {}
    synth = {k: v for k, v in predictions.items() if k in SCENARIO_QUALITY_RANK}
    metrics = build_metrics(tau_fracs=[f for f in tau_fracs if f > 0], weights=weights, **metric_kwargs)
    robust = run_robustness(real, n, synth, metrics, **robust_kwargs)
    disc = run_discrimination(real, n, synth, metrics)
    _, rob = summarize_robustness(robust)
    dis = summarize_discrimination(disc)
    meta = pd.DataFrame([{"metric": m.name, "family": m.family, "tau_frac": m.tau_frac,
                          "weight": m.weight} for m in metrics])
    return meta.merge(rob, on="metric").merge(dis, on="metric")


def run_annotator_experiment(real: list[Range], n: int, predictions: dict[str, list[Range]],
                             metrics: Sequence[Metric], sigmas: Sequence[float] = SIGMAS,
                             n_annotators: int = 20, seed: int = 0) -> pd.DataFrame:
    """Simulate and score against multiple annotators with Gaussian noise on the boundaries of the ground-truth ranges."""
    preds = {k: v for k, v in predictions.items() if k in SCENARIO_QUALITY_RANK}
    original = next(m for m in metrics if m.name == "range-based")
    gold = {scen: original(real, pred, n).f for scen, pred in preds.items()}
    panels = {sg: simulate_annotators(real, n, sg, n_annotators,
                                      np.random.default_rng(seed * 10_000 + int(sg * 100)))
              for sg in sigmas}
    rows = []
    for metric in metrics:
        for scen, pred in preds.items():
            for sg, labels in panels.items():
                for a, lab in enumerate(labels):
                    rows.append({"metric": metric.name, "family": metric.family,
                                 "tau_frac": metric.tau_frac, "weight": metric.weight,
                                 "sigma": sg, "annotator": a, "scenario": scen,
                                 "f": metric(lab, pred, n).f, "f_gold": gold[scen]})
    return pd.DataFrame(rows)


def summarize_annotators(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarizes the different anootator experiment results:
    - spread:  mean F1
    - bias:    mean F1 minus the reference F1 (from true labels)
    - rmse:    error from the expected value
    - ordering:  correctness of ranking scenarios
    - inflation_mostly_missed:  bias for the mostly_missed scenario
    """
    keys = ["metric", "sigma"]
    per_scen = (raw.groupby(keys + ["scenario"], sort=False)
                .agg(mean_f=("f", "mean"), std_f=("f", "std"), gold=("f_gold", "first"))
                .reset_index())
    per_scen["bias"] = per_scen["mean_f"] - per_scen["gold"]
    out = (per_scen.groupby(keys, sort=False)
           .agg(spread=("std_f", "mean"), bias=("bias", "mean"),
                abs_bias=("bias", lambda b: b.abs().mean()))
           .reset_index())

    sq = raw.assign(se=(raw["f"] - raw["f_gold"]) ** 2)
    rmse = sq.groupby(keys, sort=False)["se"].mean().pow(0.5).rename("rmse").reset_index()

    scen = [s for s in SCENARIO_QUALITY_RANK if s in set(raw["scenario"])]
    pairs = [(a, b) for a in scen for b in scen
             if SCENARIO_QUALITY_RANK[a] < SCENARIO_QUALITY_RANK[b]]
    wide = raw.pivot_table(index=keys + ["annotator"], columns="scenario", values="f")
    correct = sum(np.where(wide[a] > wide[b], 1.0, np.where(wide[a] == wide[b], 0.5, 0.0))
                  for a, b in pairs) / len(pairs)
    ordering = (pd.Series(correct, index=wide.index).groupby(level=keys).mean()
                .rename("ordering").reset_index())

    missed = (per_scen[per_scen["scenario"] == "mostly_missed"][keys + ["bias"]]
              .rename(columns={"bias": "inflation_mostly_missed"}))
    meta = raw.drop_duplicates("metric")[["metric", "family", "tau_frac", "weight"]]
    return (meta.merge(out, on="metric").merge(rmse, on=keys)
            .merge(ordering, on=keys).merge(missed, on=keys))


def best_tau_per_sigma(summary: pd.DataFrame) -> pd.DataFrame:
    """For every weight function and sigma: the tau_frac with the lowest RMSE."""
    base = summary[summary["family"] == "range-based"]
    ua = summary[summary["family"] == "uncertainty-aware"]
    rows = []
    for w in dict.fromkeys(ua["weight"]):
        cand = pd.concat([ua[ua["weight"] == w], base.assign(weight=w, tau_frac=0.0)])
        for sg, g in cand.groupby("sigma"):
            best = g.loc[g["rmse"].idxmin()]
            rows.append({"weight": w, "sigma": sg, "best_tau_frac": best["tau_frac"],
                         "rmse": best["rmse"], "bias": best["bias"], "spread": best["spread"],
                         "ordering": best["ordering"],
                         "rmse_original": base[base["sigma"] == sg]["rmse"].iloc[0]})
    return pd.DataFrame(rows)
