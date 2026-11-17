#!/usr/bin/env python3
"""Copy the generated numbers and figures into a copy of the paper.

  python scripts/sync_paper.py path/to/paper_dir
  python scripts/sync_paper.py --check path/to/paper_dir

Copies
  generated/numbers.tex, generated/table_drift.tex  ->  <paper_dir>/generated/
  figures/fig_xp_*.pdf                              ->  <paper_dir>/figures/
and prints, for each file, whether it was new, updated or unchanged.
Files are compared byte by byte, so an unchanged file is never rewritten.

With --check nothing is copied; the exit code is 1 if the paper is out of date.
The paper directory must be outside this repository: the manuscript is not
part of it.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sources() -> list[Path]:
    """The files the paper includes, as paths relative to the repository root."""
    files = [Path("generated/numbers.tex"), Path("generated/table_drift.tex")]
    files += sorted(p.relative_to(ROOT) for p in (ROOT / "figures").glob("fig_xp_*.pdf"))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paper_dir", type=Path, help="directory that holds paper.tex")
    parser.add_argument("--check", action="store_true", help="report differences, copy nothing")
    args = parser.parse_args(argv)

    paper = args.paper_dir.resolve()
    if not paper.is_dir():
        print(f"error: not a directory: {args.paper_dir}", file=sys.stderr)
        return 2
    if paper == ROOT or ROOT in paper.parents:
        print(f"error: {args.paper_dir} is inside this repository; keep the paper outside it", file=sys.stderr)
        return 2

    files = sources()
    missing = [rel for rel in files if not (ROOT / rel).is_file()]
    if missing or len(files) == 2:
        for rel in missing:
            print(f"error: missing in the repository: {rel}", file=sys.stderr)
        if len(files) == 2:
            print("error: no figures/fig_xp_*.pdf in the repository", file=sys.stderr)
        return 2

    counts = {"new": 0, "updated": 0, "unchanged": 0}
    for rel in files:
        src, dst = ROOT / rel, paper / rel
        if not dst.exists():
            status = "new"
        elif filecmp.cmp(src, dst, shallow=False):
            status = "unchanged"
        else:
            status = "updated"
        counts[status] += 1
        if status != "unchanged" and not args.check:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        print(f"{status:<9} {rel}")

    changed = counts["new"] + counts["updated"]
    verb = "would copy" if args.check else "copied"
    print(f"{verb} {changed} of {len(files)} files into {paper} ({counts['unchanged']} unchanged)")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
