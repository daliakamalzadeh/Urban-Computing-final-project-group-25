"""
Synthetic prediction scenarios including: exact, small boundary error, large boundary error, too long, fragmented, and mostly missed.

* ``perturb_labels`` shifts the start/end of every ground-truth range to
  simulate annotation uncertainty (0, +-2, +-5, +-10, +-20 time steps).
"""
from __future__ import annotations

import numpy as np

from .metrics import Range, normalize_ranges

# Scenario name -> expected quality rank (1 = best). Ties share a rank.
# Used to measure how well a metric *discriminates* good from bad predictions.
SCENARIO_QUALITY_RANK: dict[str, float] = {
    "exact": 1,
    "small_boundary_error": 2,
    "large_boundary_error": 3,
    "too_long": 4,
    "fragmented": 5,
    "mostly_missed": 6,
}

PERTURBATION_MODES = ("shift_left", "shift_right", "expand", "shrink")


def make_predictions(real: list[Range], n: int, seed: int = 0,
                     small_error: int = 3, large_error: int = 25,
                     too_long_frac: float = 0.5, n_fragments: int = 5,
                     missed_frac: float = 0.9) -> dict[str, list[Range]]:
    """Creates prediction ranges of different quality for evaluation: 
    - exact
    - small_boundary_error:  start and end each moved by +-small_error
    - large_boundary_error: start and end each moved by +-large_error (random sign)
    - too_long: each range extended on both sides by too_long_frac * L
    - fragmented:  range cut into n_fragments pieces, every other one kept
    - mostly_missed: range shifted right by missed_frac * L 
    """
    rng = np.random.default_rng(seed)

    def jitter(mag: int) -> list[Range]:
        out = []
        for s, e in real:
            ds, de = rng.choice([-1, 1], size=2) * mag
            out.append((s + ds, max(s + ds, e + de)))
        return out

    #extend on both sides by a fraction of the range length
    too_long = [(s - int(too_long_frac * (e - s + 1)), e + int(too_long_frac * (e - s + 1)))
                for s, e in real]

    # split into fragments
    fragmented = []
    for s, e in real:
        edges = np.linspace(s, e + 1, n_fragments + 1).astype(int)
        for k in range(0, n_fragments, 2): # keep every other fragment
            fragmented.append((edges[k], edges[k + 1] - 1))

    # shift right by a fraction of the range length
    mostly_missed = []
    for s, e in real:
        shift = int(missed_frac * (e - s + 1))
        mostly_missed.append((s + shift, e + shift))

    scenarios = {
        "exact": list(real),
        "small_boundary_error": jitter(small_error),
        "large_boundary_error": jitter(large_error),
        "too_long": too_long,
        "fragmented": fragmented,
        "mostly_missed": mostly_missed,
    }
    return {k: normalize_ranges(v, n) for k, v in scenarios.items()}


def perturb_labels(real: list[Range], n: int, magnitude: int, mode: str) -> list[Range]:
    """Simulate annotation uncertainty with small differences accross annotators by perturbing the start and end of every ground-truth range. """
    out = []
    for s, e in real:
        if mode == "shift_left":
            ds, de = -magnitude, -magnitude
        elif mode == "shift_right":
            ds, de = magnitude, magnitude
        elif mode == "expand":
            ds, de = -magnitude, magnitude
        elif mode == "shrink":
            ds, de = magnitude, -magnitude
        else:
            raise ValueError(f"unknown perturbation mode {mode!r}")
        ns, ne = s + ds, e + de
        if ns > ne:                       # a very short range shrank to nothing
            ns = ne = (s + e) // 2
        out.append((ns, ne))
    return normalize_ranges(out, n)


def simulate_annotators(real: list[Range], n: int, sigma: float, n_annotators: int = 20,
                        rng: np.random.Generator | None = None) -> list[list[Range]]:
    """Label sets of ``n_annotators`` simulated annotators of the same data.

    Each annotator places every start and every end boundary with its own
    independent Gaussian error, round(N(0, sigma)) time steps: most boundaries
    land close to the true one, a few further away.
    """
    if rng is None:
        rng = np.random.default_rng()
    annotators = []
    for _ in range(n_annotators):
        out = []
        for s, e in real:
            ds, de = np.rint(rng.normal(0, sigma, size=2)).astype(int)
            ns, ne = s + ds, e + de
            if ns > ne:                   # a very short range shrank to nothing
                ns = ne = (s + e) // 2
            out.append((ns, ne))
        annotators.append(normalize_ranges(out, n))
    return annotators
