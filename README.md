# kxn-digraph

Code for the paper *The Functional Digraph of the Map x -> k x^n Modulo m* (Wei Chen).

It does three jobs:

1. **Checks the theorems.** Every numbered statement and every worked example of the paper is tested against direct iteration of the map.
2. **Computes the graph without iterating.** Cycle lengths, cycle counts, tree levels, in-degrees, fixed points and absorption depths come from closed forms.
3. **Produces the numbers and figures of the computational section.** They are exact (all units of each prime, no sampling), and the paper reads them from generated files.

## Install

```
pip install -e ".[dev]"
```

## Use

```
kxn describe 2 2 27      # structure of G_{2,2,27}: formulas, then a brute-force cross-check
kxn length 2 7 45 4      # eventual cycle length of the vertex 4, by formula only
kxn xp 3 3 100003        # exact cycle-length distribution of 3x^3 modulo a prime
kxn dot 2 2 27 > g.dot   # Graphviz source for a small graph
```

In Python:

```python
from kxn import PowerMap, cycle_multiset

pm = PowerMap(2, 3, 25)
pm.eventual_cycle_length(7)   # formula only
pm.graph().cycle_multiset()   # brute force
cycle_multiset(3, 1, 20)      # T(v) = 3v + 1 on Z/20: {2: 2, 4: 4}
```

## Reproduce the paper

```
pytest -q                        # all statements and examples (about 40 s)
python scripts/make_numbers.py   # generated/numbers.tex, generated/table_drift.tex (about 1 min)
python scripts/make_figures.py   # figures/fig_xp_*.pdf
python scripts/check_tex.py paper.tex   # labels, citations, hand-typed numbers
```

## Layout

| Path | What it holds |
|---|---|
| `src/kxn/graph.py` | brute-force functional graphs: cycles, depths, tree shapes |
| `src/kxn/affine.py` | closed forms for `T(v) = n v + c` on `Z/M` |
| `src/kxn/powermap.py` | `k x^n mod m`: strata, absorption, in-degrees, fixed points, cycle lengths |
| `src/kxn/stats.py` | exact distribution of `log L_p / log p`, KS distance, label-permutation test |
| `tests/` | one test per statement of the paper |
| `scripts/` | numbers, figures, LaTeX checker |
| `generated/`, `figures/` | outputs that the paper includes |

## Which test checks which statement

| Statement | Test |
|---|---|
| Stratum dynamics, unit stratum, singular fixed points, absorption depth | `test_powermap.py::test_prime_power_statements` |
| In-degrees on the strata, fixed points on all of `Z/m` | `test_powermap.py::test_prime_power_statements`, `::test_composite_moduli` |
| Scaling lemma, trees on the unit group of any `m` | `test_powermap.py::test_scaling_conjugacy_and_unit_trees` |
| Level sizes of the unit trees for any `m` | `test_powermap.py::test_unit_tree_level_sizes` |
| Recurrent vertices for composite `m` | `test_powermap.py::test_global_structure_recurrent_vertices` |
| Local cycle lengths, fixed-point-free corollary | `test_affine.py::test_local_cycle_lengths_and_fixed_point_free_corollary` |
| Fixed points, in-degrees, reduction, cycle multiset, tree isomorphism, level profile | `test_affine.py::test_everything_on_general_modulus` |
| Involutions | `test_powermap.py::test_involutions` |
| Coupling bound between multipliers | `test_stats.py::test_coupling_bound_between_multipliers` |
| Worked examples and figures | `test_examples.py` |

## License

MIT
