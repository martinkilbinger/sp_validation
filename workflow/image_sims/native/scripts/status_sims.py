#!/usr/bin/env python3
"""Disk-scan status report for the native image-sims workflow.

Not a DAG node (same reasoning as shapepipe/workflow/scripts/run_report.py:
a report that's part of the DAG would itself be poisoned by the failures it
must enumerate). Run any time, mid-run or after:

    python scripts/status_sims.py <run_dir>
    python scripts/status_sims.py -c my_run.yaml     # dirs from the run config

For each tile/exposure and each stage, a manifest means complete, a log
with no manifest means failed, and neither means not yet attempted -- the
same three-way read completeness.py's own contract establishes.
"""

import argparse
import json
import sys
from pathlib import Path

TILE_STAGES = [
    "tile_get_images", "tile_uncompress", "tile_find_exposures",
    "tile_merge_headers", "tile_detect", "tile_fake_psf", "tile_vignets",
    "tile_ngmix", "tile_make_cat",
]
EXP_STAGES = ["exp_get_images", "exp_split"]

# The Gen-2 RUN_NAME suffixes (run_sp_tile_Git, run_sp_exp_Gie, ...) --
# shorter and more familiar than the stage names for a wide table.
STAGE_ABBR = {
    "tile_get_images": "Git",
    "tile_uncompress": "Uz",
    "tile_find_exposures": "Fe",
    "tile_merge_headers": "Mh_exp",
    "tile_detect": "Sx",
    "tile_fake_psf": "fpsf",
    "tile_vignets": "ViVi",
    "tile_ngmix": "Ng",
    "tile_make_cat": "Mc",
    "exp_get_images": "Gie",
    "exp_split": "Sp",
}

SYMBOL = {"complete": ".", "failed": "X", "missing": " "}


def stage_status(unit_dir: Path, stage: str) -> str:
    if (unit_dir / "manifests" / f"{stage}.json").exists():
        return "complete"
    if (unit_dir / "logs" / f"{stage}.json").exists():
        return "failed"
    return "missing"


def report_finished(run_dir: Path, kind: str, stages: list) -> None:
    """--very-short: the abbreviated header, and ONE row below it -- how many
    units have COMPLETED each step -- instead of one row per unit."""
    root = run_dir / kind
    units = sorted(p for p in root.glob("*/*") if p.is_dir()) if root.is_dir() else []
    print(f"{len(units)} {kind}")
    if not units:
        return

    header = f"{'':<12}" + "".join(f"{STAGE_ABBR[s]:>8}" for s in stages)
    print(header)
    print("-" * len(header))

    counts = {s: 0 for s in stages}
    for unit_dir in units:
        for s in stages:
            if stage_status(unit_dir, s) == "complete":
                counts[s] += 1
    row = f"{'complete':<12}" + "".join(f"{counts[s]:>8}" for s in stages)
    print(row)


def report_units(run_dir: Path, kind: str, stages: list, *, short: bool = False) -> None:
    root = run_dir / kind
    units = sorted(p for p in root.glob("*/*") if p.is_dir()) if root.is_dir() else []
    if not units:
        print(f"No {kind} units found under {root}")
        return

    counts = {s: {"complete": 0, "failed": 0, "missing": 0} for s in stages}

    header = f"{'unit':<12}" + "".join(f"{STAGE_ABBR[s]:>8}" for s in stages)
    print(header)
    print("-" * len(header))

    for unit_dir in units:
        unit = unit_dir.name
        row = f"{unit:<12}"
        for s in stages:
            st = stage_status(unit_dir, s)
            counts[s][st] += 1
            row += f"{SYMBOL[st]:>8}"
        print(row)

    # --short: just the per-unit table above, skip the aggregate counts below.
    if short:
        print()
        return

    print()
    print(f"{len(units)} {kind} total")
    for s in stages:
        c = counts[s]
        print(f"  {STAGE_ABBR[s]:<8} complete={c['complete']:<4} failed={c['failed']:<4} "
              f"not-yet-run={c['missing']}")
    print()


def final_cats(products_dir: Path) -> list:
    """The published catalogues, on the products root (which is run_dir when
    the run config sets no products_dir)."""
    return sorted(products_dir.glob("tiles/*/*/final_cat-*.fits"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, nargs="?",
                   help="run directory; omit when passing -c")
    p.add_argument("-c", "--config", type=Path,
                   help="run config (as given to run.sh): read outputs.run_dir "
                        "and outputs.products_dir from it, $variables expanded")
    p.add_argument("-s", "--short", action="store_true",
                   help="per-tile/per-exposure table only, no aggregate "
                        "'N units total' / per-stage count summary")
    p.add_argument("-S", "--very-short", action="store_true",
                   help="no per-unit table at all -- just 'N/M finished' "
                        "for tiles and exposures, once each (overrides -s)")
    args = p.parse_args(argv)

    if args.config:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import run_config_sims
        try:
            run_dir, products_dir = run_config_sims.run_dirs(args.config)
        except run_config_sims.ConfigError as exc:
            p.error(f"{exc}")
    elif args.run_dir:
        run_dir = products_dir = args.run_dir
    else:
        p.error("give a run directory, or -c with a run config")

    print(f"=== {run_dir} ===")
    if products_dir != run_dir:
        print(f"=== products: {products_dir} ===")

    if args.very_short:
        report_finished(run_dir, "tiles", TILE_STAGES)
        report_finished(run_dir, "exp", EXP_STAGES)
        print(f"final_cat produced: {len(final_cats(products_dir))}")
        return 0

    print()
    print("--- tiles ---")
    report_units(run_dir, "tiles", TILE_STAGES, short=args.short)
    print("--- exposures ---")
    report_units(run_dir, "exp", EXP_STAGES, short=args.short)

    cats = final_cats(products_dir)
    print(f"final_cat produced: {len(cats)}")
    if not args.short:
        for f in cats:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
