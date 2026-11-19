"""Validation of the metric implementations.  Run with:  python -m pytest -q"""
import numpy as np
import pytest

from fuzzyrange.metrics import (
    normalize_ranges, pointwise_scores, range_precision, range_recall,
    tolerance, uncertainty_precision, uncertainty_recall,
)
from fuzzyrange.experiments import run_annotator_experiment, summarize_annotators
from fuzzyrange.metrics import build_metrics, range_scores
from fuzzyrange.scenarios import (SCENARIO_QUALITY_RANK, make_predictions, perturb_labels,
                                  simulate_annotators)

WEIGHTS = ["linear", "exponential", "exp-fast", "step"]


def random_ranges(rng, n, k, max_len=30):
    starts = np.sort(rng.choice(n - max_len, size=k, replace=False))
    return normalize_ranges([(s, s + rng.integers(0, max_len)) for s in starts], n)


def test_normalize_clips_sorts_and_merges():
    assert normalize_ranges([(5, 8), (0, 2), (3, 4), (20, 30)], n=25) == [(0, 8), (20, 24)]
    assert normalize_ranges([(30, 40)], n=25) == []


def test_hand_worked_example():
    # R = [10, 19] (10 points), P = [12, 21] -> 8 points overlap
    R, P, n = [(10, 19)], [(12, 21)], 100
    assert range_recall(R, P, n) == pytest.approx(0.8)
    assert range_precision(R, P, n) == pytest.approx(0.8)
    # tau = 40% of L = 4, linear: w(1) = 1 - 1/5 = 0.8, w(2) = 0.6 for points 20, 21
    assert uncertainty_recall(R, P, n, tau_frac=0.4, weight="linear") == pytest.approx((8 + 1.4) / 10)
    assert uncertainty_precision(R, P, n, tau_frac=0.4, weight="linear") == pytest.approx((8 + 1.4) / 10)
    # exponential: w(d) = exp(-d / tau)
    w = np.exp(-1 / 4) + np.exp(-2 / 4)
    assert uncertainty_precision(R, P, n, tau_frac=0.4, weight="exponential") == pytest.approx((8 + w) / 10)
    # fast exponential: w(d) = exp(-3 d / tau), stricter than linear
    w_fast = np.exp(-3 / 4) + np.exp(-6 / 4)
    assert uncertainty_precision(R, P, n, tau_frac=0.4, weight="exp-fast") == pytest.approx((8 + w_fast) / 10)
    assert w_fast < 1.4                        # less credit than linear (0.8 + 0.6)


def test_cardinality_and_existence():
    R, n = [(0, 9)], 50
    P = [(0, 1), (4, 5)]                       # 2 fragments, 4/10 points
    assert range_recall(R, P, n) == pytest.approx(0.4 / 2)          # gamma = 1/x
    assert range_recall(R, P, n, alpha=1.0) == pytest.approx(1.0)
    # one predicted range spanning two real ranges -> half credit in precision
    assert range_precision([(0, 4), (6, 9)], [(0, 9)], n) == pytest.approx(0.9 / 2)


def test_subsumes_classical_pointwise():
    """Sec. 4.3 of the paper: with unit-size ranges, alpha=0 and flat bias,
    Recall_T == Recall and Precision_T == Precision.  Points are drawn on even
    indices only so that no two unit ranges touch (touching ranges would merge);
    every range then overlaps at most one other, so gamma has no effect."""
    rng = np.random.default_rng(1)
    n = 300
    for _ in range(20):
        r_mask, p_mask = np.zeros(n, bool), np.zeros(n, bool)
        r_mask[::2] = rng.random(n // 2) < 0.3
        p_mask[::2] = rng.random(n // 2) < 0.3
        R = [(i, i) for i in np.flatnonzero(r_mask)]
        P = [(i, i) for i in np.flatnonzero(p_mask)]
        classical = pointwise_scores(R, P, n)
        assert range_recall(R, P, n) == pytest.approx(classical.recall)
        assert range_precision(R, P, n) == pytest.approx(classical.precision)


@pytest.mark.parametrize("weight", WEIGHTS)
def test_tau_zero_equals_original(weight):
    rng = np.random.default_rng(2)
    n = 500
    for _ in range(15):
        R, P = random_ranges(rng, n, 5), random_ranges(rng, n, 8)
        for alpha in (0.0, 0.3):
            assert uncertainty_recall(R, P, n, tau_frac=0, weight=weight, alpha=alpha) \
                == pytest.approx(range_recall(R, P, n, alpha=alpha))
        assert uncertainty_precision(R, P, n, tau_frac=0, weight=weight) \
            == pytest.approx(range_precision(R, P, n))


@pytest.mark.parametrize("weight", WEIGHTS)
def test_bounds_and_perfect_score(weight):
    rng = np.random.default_rng(3)
    n = 500
    for _ in range(20):
        R, P = random_ranges(rng, n, 4), random_ranges(rng, n, 6)
        for frac in (0.1, 0.5, 1.0):
            ur = uncertainty_recall(R, P, n, tau_frac=frac, weight=weight)
            up = uncertainty_precision(R, P, n, tau_frac=frac, weight=weight)
            assert 0 <= ur <= 1 and 0 <= up <= 1
    # Perfect prediction -> 1.  The ranges (L = 21, tau = 8) are more than tau apart: closer ranges
    # would fall into each other's tolerance window, giving x = 2 and gamma = 1/2.
    R = [(s, s + 20) for s in range(0, 480, 60)]
    assert uncertainty_recall(R, R, n, tau_frac=0.4, weight=weight) == pytest.approx(1.0)
    assert uncertainty_precision(R, R, n, tau_frac=0.4, weight=weight) == pytest.approx(1.0)


def test_tau_scales_with_anomaly_length():
    assert tolerance((0, 9), 0.2) == 2           # L = 10
    assert tolerance((0, 99), 0.2) == 20         # L = 100
    assert tolerance((8423, 8629), 0.05) == 10   # NYC-Taxi window, L = 207
    # the same prediction 3 steps past the end: inside the tolerance of a long
    # anomaly (tau = 10), outside the tolerance of a short one (tau = 2)
    n = 400
    assert uncertainty_precision([(100, 199)], [(202, 202)], n, tau_frac=0.1) > 0
    assert uncertainty_precision([(100, 119)], [(122, 122)], n, tau_frac=0.1) == 0


def test_partial_credit_for_shifted_label():
    """A perfect prediction scored against labels shifted by 3 steps keeps more
    credit under the uncertainty-aware metric than under the original one."""
    R, n = [(100, 199)], 400
    shifted = perturb_labels(R, n, 3, "shift_right")
    assert uncertainty_recall(shifted, R, n, tau_frac=0.1) > range_recall(shifted, R, n)
    assert uncertainty_precision(shifted, R, n, tau_frac=0.1) > range_precision(shifted, R, n)


def test_against_prts_reference():
    """Optional cross-check against the third-party PRTS package (pip install prts)."""
    prts = pytest.importorskip("prts")
    rng = np.random.default_rng(4)
    n = 400
    for _ in range(10):
        R, P = random_ranges(rng, n, 4), random_ranges(rng, n, 6)
        real = np.zeros(n, int)
        pred = np.zeros(n, int)
        for s, e in R:
            real[s:e + 1] = 1
        for s, e in P:
            pred[s:e + 1] = 1
        # API checked against github.com/CompML/PRTS: ts_recall/ts_precision(real, pred,
        # alpha, cardinality, bias) on 0/1 arrays. Inputs are never all-zero here
        # (PRTS raises an AssertionError on that, see its issue #76).
        for alpha in (0.0, 0.5):
            assert prts.ts_recall(real, pred, alpha=alpha, cardinality="reciprocal", bias="flat") \
                == pytest.approx(range_recall(R, P, n, alpha=alpha), abs=1e-9)
        assert prts.ts_precision(real, pred, alpha=0.0, cardinality="reciprocal", bias="flat") \
            == pytest.approx(range_precision(R, P, n), abs=1e-9)


def test_simulated_annotators():
    R, n = [(100, 199), (300, 349)], 500
    rng = np.random.default_rng(5)
    assert simulate_annotators(R, n, 0, 5, rng) == [R] * 5          # sigma 0: everyone agrees
    panel = simulate_annotators(R, n, 5, 20, rng)
    assert len(panel) == 20 and all(len(a) == 2 for a in panel)
    starts = np.array([a[0][0] for a in panel]) - 100
    assert abs(starts.mean()) < 3 and 2 < starts.std() < 8         # roughly N(0, 5)


def test_annotator_summary_without_noise():
    """With sigma = 0 every annotator has the true labels: the original metric has
    no spread, no bias and no error, and its ordering accuracy is just how well its
    scores on the true labels follow SCENARIO_QUALITY_RANK."""
    R, n = [(100, 199), (300, 399)], 600
    preds = make_predictions(R, n)
    raw = run_annotator_experiment(R, n, preds, build_metrics(), sigmas=[0], n_annotators=3)
    s = summarize_annotators(raw).set_index("metric")
    assert s.loc["range-based", ["spread", "bias", "rmse"]].abs().max() == pytest.approx(0)
    f = {k: range_scores(R, v, n).f for k, v in preds.items()}
    pairs = [(x, y) for x in f for y in f if SCENARIO_QUALITY_RANK[x] < SCENARIO_QUALITY_RANK[y]]
    expected = np.mean([1.0 if f[x] > f[y] else 0.5 if f[x] == f[y] else 0.0 for x, y in pairs])
    assert s.loc["range-based", "ordering"] == pytest.approx(expected)
    # the uncertainty-aware metric is more generous than deserved when labels are exact
    assert s.loc["UA-linear (tau=5% of L)", "bias"] > 0
