# Uncertainty-aware range-based precision & recall (Group 25)

Implementation of Tatbul et al. (NeurIPS 2018) range-based precision/recall plus our
fuzzy-boundary extension, and the experiments from the proposal.

```
fuzzyrange/
  metrics.py      point-wise, range-based (Tatbul) and uncertainty-aware metrics
  data.py         NAB NYC-Taxi loader (downloads once to data/)
  scenarios.py    prediction scenarios, label perturbations, simulated annotators
  experiments.py  robustness (RQ1/2), discrimination (RQ3), tau sweep, annotator experiment
  plots.py        figures
run_experiments.py  runs everything -> results/
tests/test_metrics.py
```

## Run

```bash
pip install -r requirements.txt
python run_experiments.py            # NYC-Taxi
python -m pytest -q                  # validation tests
```

Options: `--tau-frac`, `--alpha`, `--annotators`, `--data-dir`, `--out`.

The tolerance tau of every anomaly is a fraction of its length (`--tau-frac`, default
0.05): tau = round(0.05 x 207) = 10 steps (5 h) for the NYC-Taxi windows.

The range-based metrics always use a flat positional bias and cardinality
gamma(x) = 1/x, so fragmented detections are penalised.

## Simulated annotators

Each of 20 simulated annotators places every anomaly boundary with independent
Gaussian noise N(0, sigma), sigma = 2, 5, 10, 20 steps. Every prediction is scored
against each annotator and compared with the score it deserves: the original
range-based F1 on the TRUE labels. Per metric and sigma we report

* spread - std of F1 across annotators (a too strict metric scores high)
* bias - mean F1 minus deserved F1 (a too tolerant metric scores positive)
* RMSE - combines both; lowest = most accurate metric / best tau
* ordering - how often a better prediction still gets the higher score

so a metric cannot look better just by being more tolerant
(fig7, fig8, `annotators_*.csv`).

`pip install prts` enables an extra test that compares our range-based metric with the
PRTS reference implementation (it is skipped otherwise).
