"""Closed forms for T(v) = n v + c on Z/MZ against direct iteration."""
from math import gcd

import pytest

from kxn import affine
from kxn.graph import functional_graph

SMALL_M = range(1, 73)
SMALL_N = range(1, 20)


def brute(n, c, M):
    return functional_graph(lambda v: (n * v + c) % M, range(M))


def test_local_cycle_lengths_and_fixed_point_free_corollary():
    for q, bmax in [(2, 5), (3, 4), (5, 2), (7, 2)]:
        for beta in range(1, bmax + 1):
            Q = q**beta
            for n in range(1, 40):
                if n % q == 0:
                    continue
                for c in range(Q):
                    G = brute(n, c, Q)
                    for v in range(Q):
                        assert G.cycle_len[v] == affine.local_cycle_length(q, beta, n, c, v)
                    e = affine.INF if n == 1 else affine.vq(n - 1, q)
                    kappa = beta if c == 0 else affine.vq(c, q)
                    if kappa < min(e, beta):
                        v2 = 1 if n == 1 else affine.vq(n + 1, 2)
                        tstar = q ** (beta - kappa) if q != 2 else 2 ** max(1, beta - kappa + 1 - v2)
                        assert set(G.cycle_len.values()) == {tstar}


@pytest.mark.parametrize("M", SMALL_M)
def test_everything_on_general_modulus(M):
    for n in SMALL_N:
        gam = gcd(n, M)
        t0, g, M1, _ = affine.reduction(n, 0, M)
        kernel_tree = brute(n, 0, M).tree_shape(0)
        for c in range(M):
            G = brute(n, c, M)
            # fixed points
            assert sum(G.F(v) == v for v in range(M)) == affine.fixed_point_count(n, c, M)
            # in-degrees
            indeg = G.indegree()
            assert all(indeg.get(u, 0) == (gam if (u - c) % gam == 0 else 0) for u in range(M))
            # reduction: recurrent set, depth bound
            assert sorted(G.recurrent) == affine.recurrent_set(n, c, M)
            assert max(G.depth.values()) == (t0 if gam > 1 else 0)
            # cycle lengths of every vertex and the cycle multiset, formula only
            assert all(G.cycle_len[v] == affine.eventual_cycle_length(n, c, M, v) for v in range(M))
            assert G.cycle_multiset() == affine.cycle_multiset(n, c, M)
            assert G.length_distribution() == affine.length_distribution(n, c, M)
            # tree isomorphism and level profile
            assert set(G.tree_shapes()) == {kernel_tree}
            prof = affine.level_profile(n, M)
            for r in list(G.recurrent)[:3]:
                assert G.level_profile(r) == {s: k for s, k in prof.items() if k}


def test_remark_shift_within_fixed_point_free_regime_changes_lengths():
    # n = 10 on Z/27: c = 1 and c = 3 are both fixed-point free, lengths differ
    assert affine.cycle_multiset(10, 1, 27) == {27: 1}
    assert affine.cycle_multiset(10, 3, 27) == {9: 3}


def test_wrong_height_formula_example():
    # M = 16, n = 8: height 2, not 1
    assert affine.tree_height(8, 16) == 2
    assert affine.level_profile(8, 16) == {1: 7, 2: 8}
