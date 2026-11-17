"""Brute-force functional graphs of maps on finite sets.

Everything here iterates the map directly. It is the ground truth that the
closed forms in ``affine`` and ``powermap`` are tested against.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Set


@dataclass
class FunctionalGraph:
    """The functional digraph of ``F`` on the vertex list ``V``."""

    F: Callable
    V: List[Hashable]
    recurrent: Set[Hashable]
    cycle_len: Dict[Hashable, int]  # eventual cycle length of every vertex
    depth: Dict[Hashable, int]  # steps needed to reach the recurrent set
    root: Dict[Hashable, Hashable]  # first recurrent vertex on the orbit

    def cycle_multiset(self) -> Dict[int, int]:
        """{cycle length: number of cycles of that length}."""
        seen = Counter(self.cycle_len[r] for r in self.recurrent)
        return {length: count // length for length, count in sorted(seen.items())}

    def length_distribution(self) -> Dict[int, int]:
        """{L: number of vertices whose eventual cycle length is L}."""
        return dict(sorted(Counter(self.cycle_len.values()).items()))

    def children(self) -> Dict[Hashable, List[Hashable]]:
        """Tree children: non-recurrent preimages of each vertex."""
        ch: Dict[Hashable, List[Hashable]] = {}
        for x in self.V:
            if x not in self.recurrent:
                ch.setdefault(self.F(x), []).append(x)
        return ch

    def tree_shape(self, r: Hashable, _children=None) -> str:
        """Canonical string of the rooted tree hanging from the recurrent vertex r.

        Two trees are isomorphic as rooted trees exactly when the strings agree.
        """
        ch = _children if _children is not None else self.children()
        return _canon(ch, r)

    def tree_shapes(self) -> Counter:
        """Counter of tree shapes over all recurrent vertices."""
        ch = self.children()
        return Counter(_canon(ch, r) for r in self.recurrent)

    def level_profile(self, r: Hashable) -> Dict[int, int]:
        """{depth s >= 1: number of tree vertices above r at depth s}."""
        prof = Counter(self.depth[x] for x in self.V if self.root[x] == r and x != r)
        return dict(sorted(prof.items()))

    def indegree(self) -> Counter:
        return Counter(self.F(x) for x in self.V)


def _canon(children: Dict, node) -> str:
    # iterative post-order so that deep trees do not hit the recursion limit
    out: Dict = {}
    stack = [(node, False)]
    while stack:
        v, done = stack.pop()
        kids = children.get(v, [])
        if done:
            out[v] = "(" + "".join(sorted(out[c] for c in kids)) + ")"
        else:
            stack.append((v, True))
            stack.extend((c, False) for c in kids)
    return out[node]


def functional_graph(F: Callable, V: Iterable[Hashable]) -> FunctionalGraph:
    """Build the functional graph of F on V by direct iteration."""
    V = list(V)
    nxt = {x: F(x) for x in V}
    # 0 = unvisited, 1 = on the current path, 2 = finished
    state = dict.fromkeys(V, 0)
    cycle_len: Dict = {}
    depth: Dict = {}
    root: Dict = {}
    recurrent: Set = set()
    for start in V:
        if state[start]:
            continue
        path = []
        x = start
        while state[x] == 0:
            state[x] = 1
            path.append(x)
            x = nxt[x]
        if state[x] == 1:  # closed a new cycle at x
            i = path.index(x)
            cyc = path[i:]
            for y in cyc:
                recurrent.add(y)
                cycle_len[y] = len(cyc)
                depth[y] = 0
                root[y] = y
                state[y] = 2
            path = path[:i]
        for y in reversed(path):  # walk the tail back towards the start
            z = nxt[y]
            cycle_len[y] = cycle_len[z]
            depth[y] = depth[z] + 1
            root[y] = root[z]
            state[y] = 2
    return FunctionalGraph(nxt.__getitem__, V, recurrent, cycle_len, depth, root)


def to_dot(G: FunctionalGraph, name: str = "G") -> str:
    """Graphviz DOT source for a (small) functional graph."""
    lines = [f"digraph {name} {{", "  node [shape=circle, fontsize=10];"]
    for x in G.V:
        style = ' style=filled fillcolor="#dddddd"' if x in G.recurrent else ""
        lines.append(f'  "{x}" [label="{x}"{style}];')
    for x in G.V:
        lines.append(f'  "{x}" -> "{G.F(x)}";')
    lines.append("}")
    return "\n".join(lines)
