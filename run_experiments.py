"""
Run all experiments write tables + figures to ''results/''.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fuzzyrange import plots
from fuzzyrange.data import load_nyc_taxi
from fuzzyrange.experiments import (
    MAGNITUDES, TAU_FRACS, best_tau_per_sigma, ranking_stability, run_annotator_experiment,
    run_discrimination, run_robustness, run_tau_sweep, summarize_annotators,
    summarize_discrimination, summarize_robustness,
)
from fuzzyrange.metrics import build_metrics, tolerance
from fuzzyrange.scenarios import make_predictions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tau-frac", type=float, default=0.05,
                    help="tolerance as a fraction of each anomaly's length (main comparison)")
    ap.add_argument("--alpha", type=float, default=0.0, help="existence-reward weight (recall)")
    ap.add_argument("--annotators", type=int, default=20, help="simulated annotators per sigma")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160, "display.max_columns", 20, "display.precision", 3)

    ds = load_nyc_taxi(args.data_dir)
    print(f"Dataset {ds.name}: n = {ds.n} points, {len(ds.anomalies)} labelled anomalies")
    for i, (s, e) in enumerate(ds.anomalies):
        print(f"  R{i + 1}: [{s}, {e}]  length {e - s + 1}  tau {tolerance((s, e), args.tau_frac)}  "
              f"({ds.timestamps[s]} -> {ds.timestamps[e]})")

    predictions = make_predictions(ds.anomalies, ds.n)

    metric_kwargs = dict(alpha=args.alpha)
    metrics = build_metrics(tau_fracs=[args.tau_frac], weights=["linear", "exponential", "exp-fast"], **metric_kwargs)

    # ---- RQ1 + RQ2: robustness to perturbed labels ------------------------
    robust = run_robustness(ds.anomalies, ds.n, predictions, metrics,
                            magnitudes=MAGNITUDES)
    robust.to_csv(out / "robustness_raw.csv", index=False)
    per_mag, rob_summary = summarize_robustness(robust)
    per_mag.to_csv(out / "robustness_by_magnitude.csv", index=False)
    stab = ranking_stability(robust)
    stab.to_csv(out / "ranking_stability.csv", index=False)

    # ---- RQ3: discrimination on original labels ---------------------------
    disc = run_discrimination(ds.anomalies, ds.n, predictions, metrics)
    disc.to_csv(out / "discrimination_scores.csv", index=False)
    disc_summary = summarize_discrimination(disc)
    summary = rob_summary.merge(disc_summary, on="metric")
    summary.to_csv(out / "summary.csv", index=False)

    # ---- tau / weight-function sweep --------------------------------------
    sweep = run_tau_sweep(ds.anomalies, ds.n, predictions, metric_kwargs=metric_kwargs)
    sweep.to_csv(out / "tau_sweep.csv", index=False)

    # ---- simulated Gaussian annotators ------------------------------------
    ann_metrics = build_metrics(tau_fracs=[f for f in TAU_FRACS if f > 0],
                                weights=["linear", "exponential", "exp-fast", "step"], **metric_kwargs)
    ann_raw = run_annotator_experiment(ds.anomalies, ds.n, predictions, ann_metrics,
                                       n_annotators=args.annotators)
    ann_raw.to_csv(out / "annotators_raw.csv", index=False)
    ann = summarize_annotators(ann_raw)
    ann.to_csv(out / "annotators_summary.csv", index=False)
    best_tau = best_tau_per_sigma(ann)
    best_tau.to_csv(out / "annotators_best_tau.csv", index=False)

    # ---- figures ----------------------------------------------------------
    plots.plot_weight_functions(out / "fig1_weight_functions.png", ds.anomalies[0], args.tau_frac)
    plots.plot_example(out / "fig2_example_anomaly.png", ds, predictions, tau_frac=args.tau_frac)
    plots.plot_robustness(out / "fig3_robustness.png", per_mag)
    plots.plot_discrimination(out / "fig4_discrimination.png", disc)
    plots.plot_tradeoff(out / "fig5_tradeoff.png", sweep,
                        summary[summary["metric"].isin(["point-wise", "range-based"])])
    plots.plot_ranking_stability(out / "fig6_ranking_stability.png", stab)
    plots.plot_annotators(out / "fig7_annotators.png", ann,
                          [m.name for m in metrics if m.family != "point-wise"])
    plots.plot_tau_choice(out / "fig8_tau_choice.png", ann)

    # ---- console report ---------------------------------------------------
    print("\nF1 per scenario on ORIGINAL labels:")
    print(disc.pivot(index="scenario", columns="metric", values="f")
          .reindex(list(predictions))[[m.name for m in metrics]])
    print("\nMean |Delta F1| per boundary shift (synthetic scenarios, all modes):")
    print(per_mag.pivot(index="magnitude", columns="metric", values="diff_f")[[m.name for m in metrics]])
    print("\nSummary (robustness + discrimination):")
    print(summary.set_index("metric"))
    print("\nTau sweep:")
    print(sweep[["metric", "mean_dF_small_shifts", "mean_dF_largest_shift",
                 "spearman_rho", "gap_exact_vs_missed", "gap_small_vs_large"]].to_string(index=False))
    print(f"\nSimulated annotators ({args.annotators} per sigma), compared with the TRUE labels:")
    main_names = [m.name for m in metrics]
    print(ann[ann["metric"].isin(main_names)]
          .pivot(index="sigma", columns="metric", values=["rmse", "bias", "ordering"])
          .reindex(columns=main_names, level=1).to_string())
    print("  (point-wise is compared with the range-based reference, so its bias also")
    print("   contains the difference between the two definitions)")
    print("\nBest tau per annotator disagreement (lowest RMSE):")
    print(best_tau.to_string(index=False))
    print(f"\nTables and figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
