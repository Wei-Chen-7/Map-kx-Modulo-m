#!/usr/bin/env python3
"""Consistency checks on the LaTeX source of the paper.

  python scripts/check_tex.py path/to/paper.tex

Reports: labels that are referenced but not defined, defined but never
referenced, or defined twice; citations with no bibliography entry and
bibliography entries never cited; generated-number macros that are used but
not defined; em dashes; and decimal numbers typed by hand inside the
computational section (they should come from generated/numbers.tex).
Exit code 1 if anything in the first group (errors) is found.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def main(path: str) -> int:
    src = Path(path)
    tex = strip_comments(src.read_text())
    errors, warnings = [], []

    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = set(re.findall(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}", tex))
    for lab, cnt in Counter(labels).items():
        if cnt > 1:
            errors.append(f"label defined {cnt} times: {lab}")
    for r in sorted(refs - set(labels)):
        errors.append(f"reference to undefined label: {r}")
    for lab in sorted(set(labels) - refs):
        warnings.append(f"label never referenced: {lab}")

    cites = set()
    for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        cites.update(k.strip() for k in group.split(","))
    bib = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for k in sorted(cites - set(bib)):
        errors.append(f"citation with no bibliography entry: {k}")
    for k in sorted(set(bib) - cites):
        warnings.append(f"bibliography entry never cited: {k}")

    used = set(re.findall(r"\\(Num[A-Za-z]+)", tex))
    defined = set()
    for inp in re.findall(r"\\input\{([^}]+)\}", tex):
        f = (src.parent / inp).with_suffix(".tex") if not inp.endswith(".tex") else src.parent / inp
        if f.exists():
            defined.update(re.findall(r"\\newcommand\{\\(Num[A-Za-z]+)\}", f.read_text()))
        else:
            errors.append(f"\\input file not found: {inp}")
    for m in sorted(used - defined):
        errors.append(f"number macro used but not generated: \\{m}")

    for i, line in enumerate(tex.splitlines(), 1):
        if "\u2014" in line or re.search(r"(?<!-)---(?!-)", line):
            warnings.append(f"line {i}: em dash")

    comp = re.search(r"\\section\{Computations.*?(?=\\section|\\begin\{thebibliography\})", tex, re.S)
    if comp:
        body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", comp.group(0), flags=re.S)
        hand = sorted(set(re.findall(r"(?<![\w.])0\.\d{3,}", body)))
        if hand:
            warnings.append("decimals typed by hand in the computational section: " + ", ".join(hand))

    for e in errors:
        print("ERROR  ", e)
    for w in warnings:
        print("warning", w)
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
