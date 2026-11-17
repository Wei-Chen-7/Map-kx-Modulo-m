"""The map f(x) = k x^n mod m: brute force and closed forms side by side."""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd, lcm
from typing import Dict, List, Tuple

from sympy import discrete_log, factorint, n_order, primitive_root, totient

from . import affine
from .affine import vq
from .graph import FunctionalGraph, functional_graph


def trunc_val(x: int, p: int, a: int) -> int:
    """Truncated valuation on Z/p^a: min(a, v_p(x)), so v(0) = a."""
    x %= p**a
    return a if x == 0 else vq(x, p)


def unit_group(p: int, a: int) -> List[int]:
    """Orders of the cyclic factors of U_{p^a} used for discrete-log coordinates."""
    if p == 2 and a >= 3:
        return [2, 2 ** (a - 2)]
    return [int(totient(p**a))]


def unit_coords(x: int, p: int, a: int) -> List[int]:
    """Discrete-log coordinates of a unit x of Z/p^a, matching ``unit_group``.

    Odd p, or 2^a with a <= 2: x = g^v for the least primitive root g.
    2^a with a >= 3: x = (-1)^s 5^t.
    """
    N = p**a
    x %= N
    if x % p == 0:
        raise ValueError(f"{x} is not a unit modulo {N}")
    if N == 2:
        return [0]
    if p == 2 and a >= 3:
        s = 0 if x % 4 == 1 else 1
        y = x if s == 0 else (N - x) % N
        return [s, discrete_log(N, y, 5)]
    return [discrete_log(N, x, primitive_root(N))]


def nth_roots_of_unity(n: int, p: int, b: int) -> int:
    """Number of u in U_{p^b} with u^n = 1 (b = 0 gives the trivial group)."""
    if b == 0:
        return 1
    return _prod(gcd(n, M) for M in unit_group(p, b))


def is_nth_power(w: int, n: int, p: int, b: int) -> bool:
    """Is the unit w an n-th power in U_{p^b}?"""
    if b == 0:
        return True
    return all(c % gcd(n, M) == 0 for c, M in zip(unit_coords(w, p, b), unit_group(p, b)))


def _prod(it) -> int:
    out = 1
    for v in it:
        out *= v
    return out


def absorption_depth(p: int, a: int, n: int, eps: int, j0: int) -> int:
    """Steps for a vertex of valuation j0 to reach 0 (Theorem: absorption depth).

    Needs n >= 2 and j0 >= 1, or eps >= 1.
    """
    if not ((n >= 2 and j0 >= 1) or eps >= 1):
        raise ValueError("this stratum is never absorbed")
    s = 1
    while n**s * j0 + eps * affine.sigma(n, s) < a:
        s += 1
    return s


@dataclass
class PowerMap:
    k: int
    n: int
    m: int

    def __post_init__(self):
        if self.m < 2 or self.n < 1:
            raise ValueError("need m >= 2 and n >= 1")
        self.factors: Dict[int, int] = {int(p): int(a) for p, a in factorint(self.m).items()}

    # ---- brute force -------------------------------------------------
    def f(self, x: int) -> int:
        return (self.k * pow(x, self.n, self.m)) % self.m

    def graph(self) -> FunctionalGraph:
        return functional_graph(self.f, range(self.m))

    def units(self) -> List[int]:
        return [x for x in range(self.m) if gcd(x, self.m) == 1]

    def unit_graph(self) -> FunctionalGraph:
        if gcd(self.k, self.m) != 1:
            raise ValueError("units are invariant only when gcd(k, m) = 1")
        return functional_graph(self.f, self.units())

    # ---- closed forms ------------------------------------------------
    def local_cycle_length(self, x: int, p: int) -> int:
        """Eventual cycle length of x mod p^a under k x^n mod p^a, no iteration."""
        a, n = self.factors[p], self.n
        N = p**a
        k = self.k % N
        if k == 0:
            return 1
        eps = vq(k, p)
        j = trunc_val(x, p, a)
        if j == a:
            return 1
        if n == 1 and eps == 0:
            # multiplication by the unit k permutes the stratum p^j U_{p^(a-j)}
            return n_order(k, p ** (a - j))
        if j >= 1 or eps >= 1:
            return 1  # absorbed into 0
        L = 1
        for M, c, v in zip(unit_group(p, a), unit_coords(k, p, a), unit_coords(x, p, a)):
            L = lcm(L, affine.eventual_cycle_length(n, c, M, v))
        return L

    def eventual_cycle_length(self, x: int) -> int:
        """Eventual cycle length of any vertex x of Z/m, by formula only."""
        return _lcm_all(self.local_cycle_length(x, p) for p in self.factors)

    def local_fixed_points(self, p: int) -> int:
        a, n = self.factors[p], self.n
        N = p**a
        k = self.k % N
        if k == 0:
            return 1
        eps = vq(k, p)
        total = 1  # the vertex 0
        if eps == 0:
            total += _prod(
                affine.fixed_point_count(n, c, M)
                for M, c in zip(unit_group(p, a), unit_coords(k, p, a))
            )
            if n == 1:
                for s in range(1, a):
                    if (k - 1) % p ** (a - s) == 0:
                        total += int(totient(p ** (a - s)))
        return total

    def fixed_point_count(self) -> int:
        """Number of fixed points of f on all of Z/m, by formula only."""
        return _prod(self.local_fixed_points(p) for p in self.factors)

    def local_indegree(self, y: int, p: int) -> int:
        """In-degree of y in the digraph of k x^n on Z/p^a, by formula only.

        (Proposition: in-degrees on the strata.)
        """
        a, n = self.factors[p], self.n
        N = p**a
        k = self.k % N
        if k == 0:
            return N if y % N == 0 else 0
        eps = vq(k, p)
        kap = k // p**eps
        s2 = trunc_val(y, p, a)
        if s2 == a:  # y = 0: everything of valuation j with eps + n j >= a
            jmin = max(0, -(-(a - eps) // n))
            return p ** (a - jmin)
        if s2 < eps or (s2 - eps) % n:
            return 0
        j = (s2 - eps) // n
        b = a - s2
        Q = p**b
        w = ((y % N) // p**s2) % Q
        t = (w * pow(kap, -1, Q)) % Q if Q > 1 else 0
        if not is_nth_power(t if Q > 1 else 1, n, p, b):
            return 0
        return p ** (s2 - j) * nth_roots_of_unity(n, p, b)

    def indegree(self, y: int) -> int:
        return _prod(self.local_indegree(y, p) for p in self.factors)

    def scaled_multiplier(self, lam: int) -> int:
        """k' with  lam * f_k(x) = f_k'(lam * x): the substitution y = lam x."""
        if gcd(lam, self.m) != 1:
            raise ValueError("lam must be a unit")
        return (self.k * pow(pow(lam, -1, self.m), self.n - 1, self.m)) % self.m

    def is_conjugate_to_pure_power(self) -> bool:
        """True when k is an (n-1)-th power of a unit, so G_{k,n,m} = G_{1,n,m}."""
        if gcd(self.k, self.m) != 1:
            return False
        if self.n == 1:
            return self.k % self.m == 1
        return all(is_nth_power(self.k, self.n - 1, p, a) for p, a in self.factors.items())

    # ---- report ------------------------------------------------------
    def describe(self, brute_limit: int = 200_000) -> str:
        k, n, m = self.k, self.n, self.m
        out = [f"f(x) = {k} x^{n} mod {m}    m = " + " * ".join(f"{p}^{a}" for p, a in self.factors.items())]
        for p, a in self.factors.items():
            N = p**a
            kk = k % N
            if kk == 0:
                out.append(f"  mod {N}: k = 0, every vertex maps to 0")
                continue
            eps = vq(kk, p)
            out.append(f"  mod {N}: eps = v_{p}(k) = {eps}")
            if eps == 0:
                for M, c in zip(unit_group(p, a), unit_coords(kk, p, a)):
                    t0, g, M1, c1 = affine.reduction(n, c, M)
                    d = gcd(n - 1, M)
                    regime = "fixed-point" if c % d == 0 else "fixed-point-free"
                    out.append(
                        f"    units: T(v) = {n} v + {c} on Z/{M}   [{regime}; d = {d}, gamma = {gcd(n, M)}, t0 = {t0}]"
                    )
                    out.append(f"      cycles {{length: count}} = {affine.cycle_multiset(n, c, M)}")
                    if t0:
                        out.append(f"      tree levels {{depth: vertices}} = {affine.level_profile(n, M)}")
            if n >= 2 or eps >= 1:
                lo = 0 if eps >= 1 else 1
                depths = {j: absorption_depth(p, a, n, eps, j) for j in range(lo, a)}
                if depths:
                    out.append(f"    absorption depth by stratum = {depths}")
        out.append(f"  fixed points on Z/{m} (formula): {self.fixed_point_count()}")
        out.append(f"  conjugate to the pure power map x^{n}: {self.is_conjugate_to_pure_power()}")
        if m <= brute_limit:
            G = self.graph()
            out.append(f"  brute force: cycles {{length: count}} = {G.cycle_multiset()}")
            out.append(f"  brute force: distinct tree shapes = {len(G.tree_shapes())}, max depth = {max(G.depth.values())}")
            ok = all(self.eventual_cycle_length(x) == G.cycle_len[x] for x in range(m))
            out.append(f"  formula vs brute force on every vertex: {'agree' if ok else 'MISMATCH'}")
        return "\n".join(out)


def _lcm_all(it) -> int:
    out = 1
    for v in it:
        out = lcm(out, v)
    return out
