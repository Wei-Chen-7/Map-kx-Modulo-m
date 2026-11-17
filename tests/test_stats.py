"""Exact statistics against brute force, and the coupling bound."""
from math import log

from sympy import primerange

from kxn import stats
from kxn.affine import smooth_part
from kxn.powermap import PowerMap


def test_exact_distribution_matches_brute_force():
    for p in list(primerange(3, 200)) + [1009, 2003, 4513]:
        for n, k in [(2, 1), (2, 3), (3, 3), (3, 2), (5, 2), (5, 3), (7, 3), (9, 2)]:
            if k % p == 0:
                continue
            G = PowerMap(k, n, p).unit_graph()
            assert G.length_distribution() == stats.length_distribution(p, n, stats.shift_of(k, p))


def test_coupling_bound_between_multipliers():
    # sorted cycle lengths for two multipliers differ by at most B_n(p)
    for n in [2, 3, 4, 5, 7, 9, 10, 13]:
        for p in primerange(5, 140):
            B = smooth_part(n, p - 1)
            for c in range(1, p - 1):
                assert stats.monotone_ratio(p, n, 0, c) <= B + 1e-9


def test_n_equals_2_distribution_does_not_depend_on_k():
    for p in primerange(5, 300):
        base = stats.length_distribution(p, 2, 0)
        assert all(stats.length_distribution(p, 2, c) == base for c in range(1, p - 1, 7))


def test_window_prime_counts():
    assert len(stats.window(10**3, 3 * 10**3)) == 262
    assert len(stats.window(10**5, 103 * 10**3)) == 261


def test_regime_gap_is_bounded_by_the_two_adic_factor():
    for p in stats.window(10**5, 10**5 + 400):
        a = stats.moments(stats.x_distribution(p, 3, 0))["mean"]
        b = stats.moments(stats.x_distribution(p, 3, 1))["mean"]
        assert -1e-12 <= b - a <= stats.coupling_bound(p, 3) + 1e-12
        assert abs(stats.coupling_bound(p, 3) - log(smooth_part(3, p - 1)) / log(p)) < 1e-12
