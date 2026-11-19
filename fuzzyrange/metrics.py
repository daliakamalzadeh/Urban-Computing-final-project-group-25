"""
Evaluation metrics for time-series anomaly detection:

1. Classical point-wise precision / recall
2. Range-based precision / recall of Tatbul et al. (2018) with flat positional
   bias and cardinality gamma(x) = 1/x (fragmented detections are penalised)
3. Our uncertainty-aware extension of (2)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

Range = tuple[int, int]


def normalize_ranges(ranges: Iterable[Sequence[int]], n: int) -> list[Range]:
    """Clip to [0, n-1], sort and merge overlapping/adjacent ranges."""
    merged: list[Range] = []
    for s, e in sorted((max(int(s), 0), min(int(e), n - 1)) for s, e in ranges):
        if s > e:
            continue
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def ranges_to_mask(ranges: Iterable[Range], n: int) -> np.ndarray:
    """Boolean point mask made from the given ranges."""
    mask = np.zeros(n, dtype=bool)
    for s, e in ranges:
        mask[max(s, 0): min(e, n - 1) + 1] = True
    return mask


def _overlaps(a: Range, b: Range) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def cardinality_factor(x: int) -> float:
    """Penalise fragmented predictions as in the paper"""
    return 1.0 if x <= 1 else 1.0 / x


# Results Calculation
def f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    """Combine precision and recall into an F-beta score."""
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


@dataclass(frozen=True)
class Scores:
    precision: float
    recall: float
    f: float

    def as_dict(self) -> dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f": self.f}


# 1. Classical point-wise metric
def pointwise_scores(real: Iterable[Range], pred: Iterable[Range], n: int,
                     beta: float = 1.0) -> Scores:
    """Classical precision = TP/(TP+FP), recall = TP/(TP+FN) over individual points."""
    r = ranges_to_mask(normalize_ranges(real, n), n)
    p = ranges_to_mask(normalize_ranges(pred, n), n)
    tp = int(np.sum(r & p))
    fp = int(np.sum(~r & p))
    fn = int(np.sum(r & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return Scores(precision, recall, f_beta(precision, recall, beta))


# 2. Original range-based metric (Tatbul et al. 2018)
def range_recall(real: Iterable[Range], pred: Iterable[Range], n: int,
                 alpha: float = 0.0) -> float:
    """range-based recall according to the paper"""
    R = normalize_ranges(real, n)
    P = normalize_ranges(pred, n)
    if not R:
        return 0.0
    pred_mask = ranges_to_mask(P, n)
    total = 0.0
    for r in R:
        covered = pred_mask[r[0]:r[1] + 1]
        existence = float(covered.any())                           # Eq. 5
        x = sum(_overlaps(r, p) for p in P)                        # number of overlapping P_j
        overlap = cardinality_factor(x) * float(covered.mean())    # Eq. 6
        total += alpha * existence + (1 - alpha) * overlap         # Eq. 4
    return total / len(R)                                          # Eq. 3


def range_precision(real: Iterable[Range], pred: Iterable[Range], n: int) -> float:
    """range-based precision according to the paper"""
    R = normalize_ranges(real, n)
    P = normalize_ranges(pred, n)
    if not P:
        return 0.0
    real_mask = ranges_to_mask(R, n)
    total = 0.0
    for p in P:
        x = sum(_overlaps(p, r) for r in R)
        total += cardinality_factor(x) * float(real_mask[p[0]:p[1] + 1].mean())   # Eq. 9
    return total / len(P)                                                          # Eq. 8


def range_scores(real: Iterable[Range], pred: Iterable[Range], n: int,
                 alpha: float = 0.0, beta: float = 1.0) -> Scores:
    p = range_precision(real, pred, n)
    r = range_recall(real, pred, n, alpha=alpha)
    return Scores(p, r, f_beta(p, r, beta))


# 3. Uncertainty-aware extension

def _w_linear(d: np.ndarray, tau: int) -> np.ndarray:
    """linear decreasing tolerance region"""
    # 1 at the boundary, decreasing linearly, 0 after going over tau points away from the boundary
    return 1.0 - d / (tau + 1.0)


def _w_exponential(d: np.ndarray, tau: int) -> np.ndarray:
    """exponentially decreasing tolerance region"""
    # 1 at the boundary, decreasing exponentially, 0 after going over tau points away from the boundary
    return np.exp(-d / float(tau))


def _w_exp_fast(d: np.ndarray, tau: int) -> np.ndarray:
    """fast exponential decay: w(d) = exp(-3 d / tau)"""
    # stricter than linear: already ~0.05 at d = tau, so it fades out smoothly (no jump at tau)
    return np.exp(-3.0 * d / float(tau))


def _w_step(d: np.ndarray, tau: int) -> np.ndarray:
    """comparison baseline: tolorence region is flat, 1 inside the tolerance region, 0 outside"""
    return np.ones_like(d, dtype=float)


WEIGHT_FUNCTIONS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "linear": _w_linear,
    "exponential": _w_exponential,
    "exp-fast": _w_exp_fast,
    "step": _w_step,
}


def tolerance(r: Range, tau_frac: float) -> int:
    """Tolerance tau of range r. Longer anomalies get a wider tolerance region."""
    return int(round(tau_frac * (r[1] - r[0] + 1)))


def membership(r: Range, n: int, tau: int, weight: str) -> tuple[int, int, np.ndarray]:
    """Assign scores to a range and its nearby tolerance region.

    - Points inside 'r' receive a score of 1. 
    - Points within 'tau' steps outside the range receive a score from the selected weighting function
    """
    range_start, range_end = r
    window_start = max(0, range_start - tau)
    window_end = min(n - 1, range_end + tau)
    points = np.arange(window_start, window_end + 1)
    distances = np.where(
        points < range_start,
        range_start - points,
        np.where(points > range_end, points - range_end, 0),
    )
    scores = np.ones(len(points), dtype=float)
    outside_range = distances > 0
    if outside_range.any():
        scores[outside_range] = WEIGHT_FUNCTIONS[weight](
            distances[outside_range].astype(float), tau
        )
    return window_start, window_end, scores


def uncertainty_recall(real: Iterable[Range], pred: Iterable[Range], n: int,
                       tau_frac: float = 0.05, weight: str = "linear", alpha: float = 0.0) -> float:
    """Measures recall while giving partial credit to nearby boundary predictions.
    The tolerance of every real range is tau_frac times its length."""
    R = normalize_ranges(real, n)
    P = normalize_ranges(pred, n)
    if not R:
        return 0.0
    pred_mask = ranges_to_mask(P, n)
    total = 0.0
    for s, e in R:
        window_start, window_end, membership_scores = membership(
            (s, e), n, tolerance((s, e), tau_frac), weight
        )
        covered = pred_mask[window_start:window_end + 1]
        existence = (
            float(membership_scores[covered].max())
            if covered.any()
            else 0.0
        )
        omega = min(
            1.0,
            float(np.sum(membership_scores * covered)) / (e - s + 1),
        )
        x = sum(_overlaps((window_start, window_end), p) for p in P)
        total += (
            alpha * existence
            + (1 - alpha) * cardinality_factor(x) * omega
        )
    return total / len(R)


def uncertainty_precision(real: Iterable[Range], pred: Iterable[Range], n: int,
                          tau_frac: float = 0.05, weight: str = "linear") -> float:
    """Measures precision while giving partial credit to nearby boundaries.
    The tolerance of every real range is tau_frac times its length."""
    R = normalize_ranges(real, n)
    P = normalize_ranges(pred, n)
    if not P:
        return 0.0
    membership_mask = np.zeros(n, dtype=float)
    tolerance_ranges = []
    for r in R:
        window_start, window_end, membership_scores = membership(
            r, n, tolerance(r, tau_frac), weight
        )
        membership_mask[window_start:window_end + 1] = np.maximum(
            membership_mask[window_start:window_end + 1],
            membership_scores,
        )
        tolerance_ranges.append((window_start, window_end))
    total = 0.0
    for p in P:
        x = sum(_overlaps(p, tolerance_range) for tolerance_range in tolerance_ranges)
        total += cardinality_factor(x) * float(
            membership_mask[p[0]:p[1] + 1].mean()
        )
    return total / len(P)


def uncertainty_scores(real: Iterable[Range], pred: Iterable[Range], n: int,
                       tau_frac: float = 0.05, weight: str = "linear",
                       alpha: float = 0.0, beta: float = 1.0) -> Scores:
    p = uncertainty_precision(real, pred, n, tau_frac=tau_frac, weight=weight)
    r = uncertainty_recall(real, pred, n, tau_frac=tau_frac, weight=weight, alpha=alpha)
    return Scores(p, r, f_beta(p, r, beta))

# Convenience: a named metric that maps (real, pred, n) -> Scores
@dataclass(frozen=True)
class Metric:
    name: str
    family: str                      # "point-wise" | "range-based" | "uncertainty-aware"
    fn: Callable[[list[Range], list[Range], int], Scores]
    tau_frac: float = 0.0            # tolerance as a fraction of the anomaly length
    weight: str = ""

    def __call__(self, real: list[Range], pred: list[Range], n: int) -> Scores:
        return self.fn(real, pred, n)


def build_metrics(tau_fracs: Sequence[float] = (0.05,),
                  weights: Sequence[str] = ("linear", "exponential", "exp-fast"),
                  alpha: float = 0.0, beta: float = 1.0) -> list[Metric]:
    """The set of metrics compared in the experiments."""
    metrics = [
        Metric("point-wise", "point-wise",
               lambda R, P, n: pointwise_scores(R, P, n, beta=beta)),
        Metric("range-based", "range-based",
               lambda R, P, n: range_scores(R, P, n, alpha=alpha, beta=beta)),
    ]
    for w in weights:
        for frac in tau_fracs:
            metrics.append(Metric(
                f"UA-{w} (tau={frac * 100:g}% of L)", "uncertainty-aware",
                (lambda R, P, n, frac=frac, w=w: uncertainty_scores(
                    R, P, n, tau_frac=frac, weight=w, alpha=alpha, beta=beta)),
                tau_frac=frac, weight=w))
    return metrics
