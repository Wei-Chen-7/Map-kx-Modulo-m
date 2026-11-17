"""kxn: the functional digraph of x -> k x^n modulo m.

``graph``     brute-force functional graphs (ground truth)
``affine``    closed forms for T(v) = n v + c on Z/MZ
``powermap``  the map k x^n mod m: strata, absorption, in-degrees, cycle lengths
``stats``     exact statistics of log L_p / log p
"""
from .affine import cycle_multiset, eventual_cycle_length, level_profile, reduction
from .graph import FunctionalGraph, functional_graph, to_dot
from .powermap import PowerMap

__all__ = [
    "FunctionalGraph",
    "PowerMap",
    "cycle_multiset",
    "eventual_cycle_length",
    "functional_graph",
    "level_profile",
    "reduction",
    "to_dot",
]
__version__ = "0.1.0"
