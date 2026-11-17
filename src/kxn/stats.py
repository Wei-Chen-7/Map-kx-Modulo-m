"""Exact statistics of X_p = log L_p / log p for f(x) = k x^n mod p.

L_p is the eventual cycle length of a uniformly random unit modulo the prime p.
Everything is computed exactly from the closed forms in ``affine`` (all p - 1
units, no sampling, no orbit iteration).

The distribution depends on k only through the valuations v_q(ind_g(k)) at the
primes q | n - 1. Those valuations do not depend on the choice of primitive
root g, so ``shift`` below may be any integer c with the right valuations.
"""
from __future__ import annotations

import random
from math import log
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sympy import discrete_log, legendre_symbol, primerange, primitive_root

from . import affine

Dist = Tuple[np.ndarray, np.ndarray]  # sorted support, weights summing to 1


def shift_of(k: int, p: int) -> int:
    """c = ind_g(k) modulo p - 1 for the least primitive root g."""
    return discrete_log(p, k % p, primitive_root(p))


def length_distribution(p: int, n: int, c: int) -> Dict[int, int]:
    """{L: number of units of Z/p with eventual cycle length L}."""
    return affine.length_distribution(n, c, p - 1)


def x_distribution(p: int, n: int, c: int) -> Dist:
    d = length_distribution(p, n, c)
    xs = np.array([log(L) / log(p) for L in d])
    ws = np.array([cnt / (p - 1) for cnt in d.values()])
    o = np.argsort(xs)
    return xs[o], ws[o]


def pooled(primes: Sequence[int], n: int, shift: Callable[[int], int]) -> Dist:
    """Pool X_p over the primes, each prime weighted equally."""
    xs: List[float] = []
    ws: List[float] = []
    for p in primes:
        x, w = x_distribution(p, n, shift(p))
        xs.extend(x)
        ws.extend(w / len(primes))
    xs_a, ws_a = np.array(xs), np.array(ws)
    o = np.argsort(xs_a, kind="stable")
    return xs_a[o], ws_a[o]


def moments(d: Dist) -> Dict[str, float]:
    xs, ws = d
    mean = float((xs * ws).sum())
    sd = float(np.sqrt(((xs - mean) ** 2 * ws).sum()))
    cw = np.cumsum(ws)
    med = float(xs[min(np.searchsorted(cw, 0.5), len(xs) - 1)])
    return {"mean": mean, "sd": sd, "median": med}


def cdf(d: Dist, grid: np.ndarray) -> np.ndarray:
    xs, ws = d
    cum = np.concatenate([[0.0], np.cumsum(ws)])
    return cum[np.searchsorted(xs, grid, side="right")]


def tail(d: Dist, lo: float = -1.0, hi: float = 2.0) -> float:
    """P[lo < X < hi]."""
    xs, ws = d
    return float(ws[(xs > lo) & (xs < hi)].sum())


def ks_distance(a: Dist, b: Dist) -> float:
    grid = np.union1d(a[0], b[0])
    return float(np.abs(cdf(a, grid) - cdf(b, grid)).max())


def histogram(d: Dist, bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    h = np.histogram(d[0], bins=edges, weights=d[1])[0] * bins
    return edges, h


def label_permutation_test(
    primes_a: Sequence[int],
    primes_b: Sequence[int],
    n: int,
    shift: Callable[[int], int],
    trials: int = 2000,
    seed: int = 7,
) -> Dict[str, float]:
    """Is the KS distance between two groups of primes larger than chance?

    Each prime keeps its own exact distribution; only the group labels are
    shuffled. This respects the fact that vertices of one prime are not
    independent samples.
    """
    per = {p: x_distribution(p, n, shift(p)) for p in list(primes_a) + list(primes_b)}

    def pool(ps):
        xs = np.concatenate([per[p][0] for p in ps])
        ws = np.concatenate([per[p][1] / len(ps) for p in ps])
        o = np.argsort(xs, kind="stable")
        return xs[o], ws[o]

    observed = ks_distance(pool(primes_a), pool(primes_b))
    rng = random.Random(seed)
    allp = list(primes_a) + list(primes_b)
    na = len(primes_a)
    vals = []
    for _ in range(trials):
        rng.shuffle(allp)
        vals.append(ks_distance(pool(allp[:na]), pool(allp[na:])))
    vals.sort()
    return {
        "observed": observed,
        "null_median": vals[trials // 2],
        "null_95": vals[int(0.95 * trials)],
        "p_value": sum(v >= observed for v in vals) / trials,
    }


def coupling_bound(p: int, n: int) -> float:
    """eta_n(p) = log B_n(p) / log p with B_n(p) = prod_{q | n-1} q^{v_q(p-1)}."""
    return log(affine.smooth_part(n, p - 1)) / log(p)


def monotone_ratio(p: int, n: int, c1: int, c2: int) -> float:
    """Largest ratio between the sorted cycle lengths for two shifts c1, c2."""
    d1 = length_distribution(p, n, c1)
    d2 = length_distribution(p, n, c2)
    a = np.repeat(np.array(list(d1.keys()), dtype=float), list(d1.values()))
    b = np.repeat(np.array(list(d2.keys()), dtype=float), list(d2.values()))
    a.sort()
    b.sort()
    return float(np.maximum(a / b, b / a).max())


def window(lo: int, hi: int) -> List[int]:
    return [int(p) for p in primerange(lo, hi)]


def qr_shift(k: int) -> Callable[[int], int]:
    """Shift with the right parity for n = 3: even iff k is a square mod p."""
    return lambda p: 0 if legendre_symbol(k, p) == 1 else 1


def correction_term(p: float) -> float:
    """(log log p)^2 log log log p / log p, the Martin-Pomerance scale."""
    lp = log(p)
    return log(lp) ** 2 * log(log(lp)) / lp
