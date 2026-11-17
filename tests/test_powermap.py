"""f(x) = k x^n mod m: strata, absorption, in-degrees, fixed points, cycle lengths."""
from math import gcd

import pytest
from sympy import primefactors, reduced_totient

from kxn.affine import vq
from kxn.graph import functional_graph
from kxn.powermap import PowerMap, absorption_depth, trunc_val

PRIME_POWERS = [(2, a) for a in range(1, 7)] + [(3, a) for a in range(1, 5)] + [(5, 1), (5, 2), (5, 3), (7, 1), (7, 2)]


@pytest.mark.parametrize("p,a", PRIME_POWERS)
def test_prime_power_statements(p, a):
    N = p**a
    for n in range(1, 8):
        for k in range(1, N):
            pm = PowerMap(k, n, N)
            G = pm.graph()
            eps = vq(k, p)
            units = [x for x in range(N) if x % p]
            # stratum dynamics
            assert all(trunc_val(pm.f(x), p, a) == min(a, eps + n * trunc_val(x, p, a)) for x in range(1, N))
            # unit stratum invariant iff eps = 0, bijective iff gcd(n, lambda) = 1
            assert all(pm.f(x) % p for x in units) == (eps == 0)
            if eps == 0:
                bij = len({pm.f(x) for x in units}) == len(units)
                assert bij == (gcd(n, int(reduced_totient(N))) == 1)
            # singular fixed points
            for s in range(1, a):
                S = [x for x in range(N) if trunc_val(x, p, a) == s]
                fixed = [x for x in S if pm.f(x) == x]
                cond = n == 1 and eps == 0 and (k - 1) % p ** (a - s) == 0
                assert bool(fixed) == cond and (not cond or len(fixed) == len(S))
            # absorption depth
            if n >= 2 or eps >= 1:
                for x in range(1, N):
                    j0 = trunc_val(x, p, a)
                    if j0 == 0 and eps == 0:
                        continue
                    assert G.depth[x] == absorption_depth(p, a, n, eps, j0) and G.root[x] == 0
            # in-degree of every vertex, fixed-point count, cycle length of every vertex
            indeg = G.indegree()
            assert all(indeg.get(y, 0) == pm.indegree(y) for y in range(N))
            assert sum(pm.f(x) == x for x in range(N)) == pm.fixed_point_count()
            assert all(G.cycle_len[x] == pm.eventual_cycle_length(x) for x in range(N))


@pytest.mark.parametrize("m", [6, 12, 15, 24, 36, 45, 60, 63, 72, 75, 91, 96, 100, 105, 120, 144])
def test_composite_moduli(m):
    for n in range(1, 7):
        for k in range(1, m):
            pm = PowerMap(k, n, m)
            G = pm.graph()
            assert all(G.cycle_len[x] == pm.eventual_cycle_length(x) for x in range(m))
            assert sum(pm.f(x) == x for x in range(m)) == pm.fixed_point_count()
            indeg = G.indegree()
            assert all(indeg.get(y, 0) == pm.indegree(y) for y in range(m))


@pytest.mark.parametrize("m", range(2, 70))
def test_scaling_conjugacy_and_unit_trees(m):
    U = [x for x in range(m) if gcd(x, m) == 1]
    for n in range(1, 6):
        for k in U:
            pm = PowerMap(k, n, m)
            # y = lam x conjugates f_k to f_{k'}
            for lam in U[:5]:
                k2 = pm.scaled_multiplier(lam)
                assert all((lam * pm.f(x)) % m == (k2 * pow(lam * x, n, m)) % m for x in range(m))
            # all trees on the unit group are isomorphic, for every m (2-power and composite included)
            G = pm.unit_graph()
            assert len(G.tree_shapes()) == 1
            pure = functional_graph(lambda x: pow(x, n, m), U)
            assert set(G.tree_shapes()) == set(pure.tree_shapes())
            # k an (n-1)-th power  <=>  same graph as the pure power map
            if pm.is_conjugate_to_pure_power():
                assert pm.graph().cycle_multiset() == PowerMap(1, n, m).graph().cycle_multiset()


def test_n_equals_2_is_always_the_squaring_map():
    for m in range(2, 80):
        base = PowerMap(1, 2, m).graph()
        for k in range(1, m):
            if gcd(k, m) == 1:
                G = PowerMap(k, 2, m).graph()
                assert G.cycle_multiset() == base.cycle_multiset()
                assert sorted(G.tree_shapes().items()) == sorted(base.tree_shapes().items())


@pytest.mark.parametrize("m", range(2, 130))
def test_involutions(m):
    lam = int(reduced_totient(m))
    U = [x for x in range(m) if gcd(x, m) == 1]
    a2, om = vq(m, 2) if m % 2 == 0 else 0, len(primefactors(m))
    for n in {max(1, lam - 1), 2 * lam - 1}:
        for k in U:
            pm = PowerMap(k, n, m)
            assert all(pm.f(pm.f(x)) == x for x in U)
            nfix = sum(pm.f(x) == x for x in U)
            ok = all(
                (pow(k, (q - 1) // 2, q) == 1) if q != 2 else (b < 2 or (b == 2 and k % 4 == 1) or (b >= 3 and k % 8 == 1))
                for q, b in pm.factors.items()
            )
            pred = 2**om if a2 in (0, 2) else (2 ** (om - 1) if a2 == 1 else 2 ** (om + 1))
            assert nfix == (pred if ok else 0)
