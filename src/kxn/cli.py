"""Command line: ``kxn describe 2 2 27``, ``kxn length 2 3 25 7``, ``kxn dot 2 2 27``, ``kxn xp 3 3 100003``."""
from __future__ import annotations

import argparse

from . import stats
from .graph import to_dot
from .powermap import PowerMap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kxn", description="Functional digraph of x -> k x^n mod m")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("describe", help="structure of G_{k,n,m}: formulas, then a brute-force cross-check")
    d.add_argument("k", type=int), d.add_argument("n", type=int), d.add_argument("m", type=int)

    ln = sub.add_parser("length", help="eventual cycle length of one vertex, by formula only")
    ln.add_argument("k", type=int), ln.add_argument("n", type=int), ln.add_argument("m", type=int)
    ln.add_argument("x", type=int)

    dt = sub.add_parser("dot", help="Graphviz DOT source of G_{k,n,m} (small m)")
    dt.add_argument("k", type=int), dt.add_argument("n", type=int), dt.add_argument("m", type=int)

    xp = sub.add_parser("xp", help="exact distribution of the cycle length of a random unit mod a prime p")
    xp.add_argument("k", type=int), xp.add_argument("n", type=int), xp.add_argument("p", type=int)

    a = ap.parse_args(argv)
    if a.cmd == "describe":
        print(PowerMap(a.k, a.n, a.m).describe())
    elif a.cmd == "length":
        print(PowerMap(a.k, a.n, a.m).eventual_cycle_length(a.x))
    elif a.cmd == "dot":
        print(to_dot(PowerMap(a.k, a.n, a.m).graph(), name=f"G_{a.k}_{a.n}_{a.m}"))
    elif a.cmd == "xp":
        c = stats.shift_of(a.k, a.p)
        dist = stats.length_distribution(a.p, a.n, c)
        m = stats.moments(stats.x_distribution(a.p, a.n, c))
        print(f"p = {a.p}, T(v) = {a.n} v + {c} on Z/{a.p - 1}")
        for L, cnt in dist.items():
            print(f"  L = {L:>12}   units = {cnt}")
        print(f"  X_p: mean {m['mean']:.4f}, median {m['median']:.4f}, sd {m['sd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
