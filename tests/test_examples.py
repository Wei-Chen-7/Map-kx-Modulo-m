"""Every worked example and figure of the paper."""
from math import gcd

from kxn import affine
from kxn.graph import functional_graph
from kxn.powermap import PowerMap, absorption_depth


def units_cycles(k, n, m):
    return PowerMap(k, n, m).unit_graph().cycle_multiset()


def test_figure_G_2_2_27():
    pm = PowerMap(2, 2, 27)
    G = pm.graph()
    assert G.cycle_multiset() == {1: 2, 2: 1, 6: 1}
    orbit = [2]
    for _ in range(5):
        orbit.append(pm.f(orbit[-1]))
    assert orbit == [2, 8, 20, 17, 11, 26] and pm.f(26) == 2
    assert pm.f(5) == 23 and pm.f(23) == 5 and pm.f(14) == 14
    assert {x for x in range(27) if pm.f(x) == 18 and x != 18} == {3, 6, 12, 15, 21, 24}
    assert pm.indegree(18) == 6 and pm.indegree(9) == 0  # explained by the stratum in-degree formula
    assert {pm.f(9), pm.f(18)} == {0}
    # nine unit cycle vertices carry a one-edge tree, the tree at 0 is different
    shapes = G.tree_shapes()
    assert shapes["(())"] == 9 and len(shapes) == 2


def test_2x3_mod_25():
    assert units_cycles(2, 3, 25) == {2: 2, 4: 4}
    assert affine.fixed_point_count(3, 1, 20) == 0


def test_small_affine_examples():
    assert functional_graph(lambda v: (3 * v + 1) % 4, range(4)).cycle_multiset() == {2: 2}
    assert units_cycles(1, 3, 17) == {1: 2, 2: 3, 4: 2}
    assert affine.cycle_multiset(3, 0, 16) == {1: 2, 2: 3, 4: 2}
    assert affine.cycle_multiset(3, 1, 16) == {8: 2}
    assert units_cycles(3, 3, 17) == {8: 2}


def test_indegree_example_mod_7():
    pm = PowerMap(2, 3, 7)
    pre = {y: sorted(x for x in range(1, 7) if pm.f(x) == y) for y in range(1, 7)}
    assert pre[2] == [1, 2, 4] and pre[5] == [3, 5, 6]
    assert all(not pre[y] for y in (1, 3, 4, 6))


def test_2x7_mod_45():
    assert units_cycles(2, 7, 45) == {6: 4}
    assert units_cycles(2, 7, 9) == {6: 1} and units_cycles(2, 7, 5) == {2: 2}


def test_x8_mod_17_tree():
    f = lambda x: pow(x, 8, 17)
    assert sorted(x for x in range(2, 17) if f(x) == 1) == [2, 4, 8, 9, 13, 15, 16]
    assert sorted(x for x in range(1, 17) if f(x) == 16) == [3, 5, 6, 7, 10, 11, 12, 14]


def test_absorption_example():
    assert absorption_depth(3, 10, 2, 2, 1) == 2


def test_involution_example_mod_5():
    pm = PowerMap(2, 3, 5)
    assert {x: pm.f(x) for x in range(1, 5)} == {1: 2, 2: 1, 3: 4, 4: 3}


def test_full_graph_trees_are_not_all_isomorphic():
    assert len(PowerMap(2, 2, 45).graph().tree_shapes()) == 4


def test_global_fixed_points_odd_modulus():
    for m in range(3, 150, 2):
        for n in range(2, 6):
            for k in range(1, m):
                if gcd(k, m) == 1:
                    pm = PowerMap(k, n, m)
                    assert sum(pm.f(x) == x for x in range(m)) == pm.fixed_point_count()
