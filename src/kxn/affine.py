"""Closed forms for the affine map T(v) = n v + c on Z/MZ.

No orbit is iterated anywhere in this module. Each function names the
statement of the paper it implements.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from math import ceil, gcd, lcm
from typing import Dict, List, Tuple

from sympy import factorint, n_order

INF = 10**9


def vq(x: int, q: int) -> int:
    """q-adic valuation of a nonzero integer (INF for 0)."""
    if x == 0:
        return INF
    k = 0
    while x % q == 0:
        x //= q
        k += 1
    return k


def sigma(n: int, t: int) -> int:
    """sigma_t = 1 + n + ... + n^(t-1)."""
    return t if n == 1 else (n**t - 1) // (n - 1)


def fixed_point_count(n: int, c: int, M: int) -> int:
    """Number of fixed points of T on Z/MZ (Theorem: fixed points)."""
    d = gcd(n - 1, M)
    return d if c % d == 0 else 0


def delta(q: int, beta: int, n: int, c: int, v: int) -> int:
    """delta(v) = min(beta, v_q((n-1) v + c)) on Z/q^beta."""
    val = ((n - 1) * v + c) % q**beta
    return beta if val == 0 else min(beta, vq(val, q))


def local_length_from_gap(q: int, i: int, n: int) -> int:
    """Cycle length on Z/q^beta of a vertex with beta - delta(v) = i.

    Requires gcd(n, q) = 1. (Theorem: local cycle lengths.)
    """
    if n % q == 0:
        raise ValueError("local theorem needs gcd(n, q) = 1")
    if i == 0:
        return 1
    if (n - 1) % q:
        return n_order(n, q**i)
    if q != 2:
        return q**i
    v2 = 1 if n == 1 else vq(n + 1, 2)
    return 2 ** max(1, i + 1 - v2)


def local_cycle_length(q: int, beta: int, n: int, c: int, v: int) -> int:
    return local_length_from_gap(q, beta - delta(q, beta, n, c, v), n)


def stabilization_index(n: int, M: int) -> int:
    """t0 = max over q | gcd(n, M) of ceil(v_q(M) / v_q(n)); 0 if gcd = 1."""
    t0 = 0
    for q, b in factorint(M).items():
        if n % q == 0:
            t0 = max(t0, ceil(b / vq(n, q)))
    return t0


def reduction(n: int, c: int, M: int) -> Tuple[int, int, int, int]:
    """(t0, g, M1, c1): the recurrent part of T is conjugate to j -> n j + c1 on Z/M1.

    (Theorem: reduction to the bijective case.)
    """
    t0 = stabilization_index(n, M)
    g = gcd(n**t0, M)
    M1 = M // g
    c1 = (c * (n**t0 // g)) % M1
    return t0, g, M1, c1


def eventual_cycle_length(n: int, c: int, M: int, v: int) -> int:
    """Length of the cycle that v falls into under T, by formula only."""
    t0, g, M1, c1 = reduction(n, c, M)
    if M1 == 1:
        return 1
    # T^t0(v) = c sigma_t0 + n^t0 v = iota(j) with j = (n^t0 / g) v mod M1
    j = ((n**t0 // g) * v) % M1
    L = 1
    for q, b in factorint(M1).items():
        L = lcm(L, local_cycle_length(q, b, n, c1 % q**b, j % q**b))
    return L


def _local_delta_counts(q: int, beta: int, n: int, c: int) -> Dict[int, int]:
    """{i = beta - delta: number of v in Z/q^beta with that gap}, gcd(n, q) = 1."""
    Q = q**beta
    if (n - 1) % q:  # v -> (n-1) v + c is a bijection of Z/q^beta
        d = {beta - i: (q - 1) * q ** (beta - i - 1) for i in range(beta)}
        d[0] = 1
        return d
    e = INF if n == 1 else vq(n - 1, q)
    kappa = beta if c % Q == 0 else vq(c % Q, q)
    if kappa < min(e, beta):  # fixed-point-free: delta = kappa everywhere
        return {beta - kappa: Q}
    if e >= beta:  # T is the identity on this factor
        return {0: Q}
    d = {beta - e - i: (q - 1) * q ** (beta - i - 1) for i in range(beta - e)}
    d[0] = q**e
    return d


def length_distribution(n: int, c: int, M: int) -> Dict[int, int]:
    """{L: number of v in Z/MZ whose eventual cycle length is L}, by formula only."""
    t0, g, M1, c1 = reduction(n, c, M)
    if M1 == 1:
        return {1: M}
    locs: List[List[Tuple[int, int]]] = []
    for q, b in factorint(M1).items():
        counts = _local_delta_counts(q, b, n, c1 % q**b)
        locs.append([(local_length_from_gap(q, i, n), cnt) for i, cnt in counts.items()])
    dist: Dict[int, int] = defaultdict(int)
    for combo in product(*locs):
        L, cnt = 1, g  # each j in Z/M1 has g preimages v in Z/M
        for length, k in combo:
            L = lcm(L, length)
            cnt *= k
        dist[L] += cnt
    return dict(sorted(dist.items()))


def cycle_multiset(n: int, c: int, M: int) -> Dict[int, int]:
    """{cycle length: number of cycles} of T on Z/MZ, by formula only."""
    t0, g, M1, c1 = reduction(n, c, M)
    dist = length_distribution(n, c1, M1) if M1 > 1 else {1: 1}
    return {L: cnt // L for L, cnt in dist.items()}


def level_profile(n: int, M: int) -> Dict[int, int]:
    """{s: g_s - g_{s-1}}: vertices at depth s in every tree of T. Independent of c."""
    t0 = stabilization_index(n, M)
    return {s: gcd(n**s, M) - gcd(n ** (s - 1), M) for s in range(1, t0 + 1)}


def tree_height(n: int, M: int) -> int:
    return stabilization_index(n, M)


def recurrent_set(n: int, c: int, M: int) -> List[int]:
    t0, g, M1, _ = reduction(n, c, M)
    base = c * sigma(n, t0)
    return sorted((base + g * j) % M for j in range(M1))


def smooth_part(n: int, M: int) -> int:
    """B = product over primes q | n-1 of q^{v_q(M)}: the coupling constant."""
    if n <= 1:
        return M
    B = 1
    for q in factorint(n - 1):
        B *= q ** vq(M, q) if M % q == 0 else 1
    return B
