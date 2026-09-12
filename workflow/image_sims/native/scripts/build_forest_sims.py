#!/usr/bin/env python3
"""Build a tile's exposure symlink forest (the $SP_EXP view).

Forked from shapepipe/workflow/scripts/build_forest.py: identical mechanism
and rationale (see that file's docstring), except the tile->exposure lookup
reads the native workflow's plain-JSON index instead of a sqlite
run_index.sqlite (see the Snakefile's module docstring for why JSON is
enough here).
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tile", required=True)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--forest", required=True, type=Path)
    args = p.parse_args()

    index = json.loads(args.index.read_text())
    exps = index.get(args.tile, [])

    args.forest.mkdir(parents=True, exist_ok=True)
    for e in exps:
        src = args.run_dir / "exp" / e[:2] / e / "output"
        dst = args.forest / e[:2] / e / "output"   # sharded: the module glob's shape
        dst.parent.mkdir(parents=True, exist_ok=True)
        # A symlink (the normal case) is unlinked; a REAL directory left behind
        # by a hand-run or an older layout must be removed as a tree -- unlink()
        # raises IsADirectoryError on it and would kill the job.
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.symlink_to(src)
    print(f"[build_forest_sims] {args.tile}: {len(exps)} exposures -> {args.forest}")


if __name__ == "__main__":
    main()
