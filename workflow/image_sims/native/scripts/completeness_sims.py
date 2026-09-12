#!/usr/bin/env python3
"""The count-based completeness table for the NATIVE image-sims workflow.

Forked from shapepipe/workflow/scripts/completeness.py (same CLI, same
manifest/log contract -- see that file's docstring for the full design
rationale, which is unchanged here). Only the two module-level tables below
differ: COMPLETENESS and STAGE_DIR are image-sims-specific, since the real
one is keyed to real-data stage names, counts and RUN_NAMEs.

Being built up stage by stage as the port proceeds -- now covers the full
chain through tile_make_cat. No exp_psf: image sims use a known injected PSF
(tile-level fake_psf, in tile.smk), not a fitted PSFEx model. No
tile_merge_cats: unlike the real design's N-chunk ngmix, image sims run
ngmix as one call (see config_tile_Ng_batch_psfex_sx.ini), so tile_make_cat
reads it directly.

The tile.smk counts below (tile_detect, tile_fake_psf, tile_vignets,
tile_ngmix, tile_make_cat) are PROVISIONAL -- carried over from the
real-data table where the chain is structurally similar, or a best guess
where it isn't (tile_fake_psf, tile_vignets' run_2/run_3). Calibrate them
against the first real run's actual output counts and correct here, the
same way the real table documents being "verified against" specific
campaigns.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# stage -> {runner_subdir: {expect, [warn], [subpath]}}
COMPLETENESS = {
    # --- tile prepare ---
    # get_images: symlink retrieve, tile image + weight = 2 files (matches the
    # real-data nibi-flavor count for the same RETRIEVE=symlink configs).
    "tile_get_images": {"get_images_runner": dict(expect=2)},
    # tile_uncompress is FAKED for image sims (the sim weight is already
    # uncompressed -- rules/prepare.smk symlinks it into a synthetic
    # uncompress_fits_runner/output/ dir rather than calling shapepipe_run),
    # but the manifest/completeness contract is kept identical so downstream
    # rules and tooling can't tell the difference.
    "tile_uncompress": {"uncompress_fits_runner": dict(expect=1)},
    "tile_find_exposures": {"find_exposures_runner": dict(expect=1)},

    # --- exposure chain ---
    # symlink retrieve, image+weight+flag = 3 files (Gen-2's own
    # run_job_sp_canfar_v2.0.bash used n_exp=3 for image_sims vs 6 for data).
    "exp_get_images": {"get_images_runner": dict(expect=3)},
    # Same threshold as real data's split_exp_runner (Gen-2's dispatch does
    # not branch on type for job 16): 40 CCDs x 3 outputs + 1.
    "exp_split": {"split_exp_runner": dict(expect=121)},

    # --- tile post (PROVISIONAL -- see module docstring) ---
    "tile_merge_headers": {"merge_headers_runner": dict(expect=1)},
    "tile_detect": {"sextractor_runner": dict(expect=2)},
    # fake_psf_runner: one aggregate output (a galaxy_psf sqlite) per tile.
    "tile_fake_psf": {"fake_psf_runner": dict(expect=1)},
    "tile_vignets": {
        "vignetmaker_runner_run_1": dict(expect=1),
        # 3, not the real design's 5: image/weight/flag only -- image sims
        # has no simulated background, so no background/background_rms
        # multi-epoch vignet (config_tile_Sx.ini: "no background subtraction
        # for image sims").
        "vignetmaker_runner_run_2": dict(expect=3),
        # RUN_3 (segmentation vignet) has no real-data counterpart at all.
        "vignetmaker_runner_run_3": dict(expect=1),
    },
    "tile_ngmix": {"ngmix_runner": dict(expect=1)},
    "tile_make_cat": {"make_cat_runner": dict(expect=1)},
}


def count_products(run_dir, runner, spec):
    out = run_dir / runner / "output"
    if "subpath" in spec:
        out = out / spec["subpath"]
    if not out.is_dir():
        return 0
    n = 0
    try:
        with os.scandir(out) as entries:
            for e in entries:
                if not e.is_symlink() or os.path.exists(e.path):
                    n += 1
    except OSError:
        return 0
    return n


def check_counts(stage, run_dir):
    table = COMPLETENESS[stage]
    details, ok = [], True
    for runner, spec in table.items():
        n = count_products(run_dir, runner, spec)
        warn = spec.get("warn", False)
        details.append((runner, n, spec["expect"], warn))
        if not warn and n < spec["expect"]:
            ok = False
    return ok, details


# stage -> (level, run_sp_<prefix> dir under $SP_RUN/output/). Every ini this
# table names is a fork under config/cfis_image_sims/ with RUN_DATETIME=False,
# so the path is fixed -- no run-log resolution needed.
STAGE_DIR = {
    "tile_get_images":     ("tile", "run_sp_tile_Git"),
    "tile_uncompress":     ("tile", "run_sp_tile_Uz"),
    "tile_find_exposures": ("tile", "run_sp_tile_Fe"),
    "exp_get_images":      ("exp",  "run_sp_exp_Gie"),
    "exp_split":           ("exp",  "run_sp_exp_Sp"),
    "tile_merge_headers":  ("tile", "run_sp_tile_Mh_exp"),
    "tile_detect":         ("tile", "run_sp_tile_Sx"),
    "tile_fake_psf":       ("tile", "run_sp_tile_fpsf"),
    "tile_vignets":        ("tile", "run_sp_tile_ViVi"),
    "tile_ngmix":          ("tile", "run_sp_tile_Ng"),
    "tile_make_cat":       ("tile", "run_sp_tile_Mc_psfex"),
}

_ERROR_RE = re.compile(
    r"traceback|exception|\berror\b|\bfailed\b|no such file|not found|"
    r"killed|out of memory|oom|segmentation fault|bad chi2",
    re.IGNORECASE)
_TS_RE = re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\s*")
_NOISE_RE = re.compile(r"A total of 0 errors were recorded")

MAX_LOG_FILES = 40
MAX_TAIL_LINES = 120
MAX_REASONS = 3


def _normalise(line: str) -> str:
    line = _TS_RE.sub("", line.strip())
    line = re.sub(r"/\S+", "<path>", line)
    line = re.sub(r"\d+", "N", line)
    return line[:200]


def scrape_reasons(stage_dir, runner):
    seen, reasons = {}, []
    candidates = []
    for d in (stage_dir / runner / "logs", stage_dir / "logs"):
        if d.is_dir():
            candidates += sorted(p for p in d.iterdir() if p.is_file())
    for path in candidates[:MAX_LOG_FILES]:
        try:
            lines = path.read_text(errors="replace").splitlines()[-MAX_TAIL_LINES:]
        except OSError:
            continue
        for raw in lines:
            if not _ERROR_RE.search(raw) or _NOISE_RE.search(raw):
                continue
            shape = _normalise(raw)
            if shape in seen:
                seen[shape] += 1
                continue
            seen[shape] = 1
            reasons.append([path.name, _TS_RE.sub("", raw.strip())[:300], shape])
    out = []
    for name, text, shape in reasons[:MAX_REASONS]:
        n = seen[shape]
        out.append(f"{name}: {text}" + (f"  [x{n}]" if n > 1 else ""))
    return out


def build_manifest(stage, run_dir, unit, stage_subdir=None):
    level, subdir = STAGE_DIR.get(stage, (None, None))
    subdir = stage_subdir or (os.path.expandvars(subdir) if subdir else None)
    stage_dir = run_dir / "output" / subdir if subdir else run_dir
    manifest = {
        "stage": stage, "level": level, "unit": unit,
        "run_dir": str(run_dir), "stage_dir": str(stage_dir),
        "runners": {}, "failures": [],
    }

    if stage not in COMPLETENESS:
        produced = list(stage_dir.glob("**/output/*")) if stage_dir.is_dir() else []
        ok = bool(produced)
        manifest["status"] = "complete" if ok else "failed"
        manifest["n_products"] = len(produced)
        if not ok:
            manifest["failures"].append(
                {"runner": None, "found": 0, "expect": 1, "warn": False,
                 "status": "failed", "reasons": [f"zero output under {stage_dir}"]})
        return manifest, ok

    ok, details = check_counts(stage, stage_dir)
    short = False
    for runner, n, expect, warn in details:
        below = n < expect
        if below:
            short = True
        status = "complete" if not below else "warn" if warn else "failed"
        manifest["runners"][runner] = {
            "found": n, "expect": expect, "warn": warn, "status": status,
        }
        if below:
            manifest["failures"].append({
                "runner": runner, "found": n, "expect": expect,
                "warn": warn, "status": status,
                "reasons": scrape_reasons(stage_dir, runner),
            })
    manifest["status"] = "failed" if not ok else ("warn" if short else "complete")
    return manifest, ok


def write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def _unit_from_run_dir(run_dir):
    return Path(str(run_dir)).name or "unknown"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Image-sims per-unit completeness check")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="count products, write the manifest")
    c.add_argument("stage")
    c.add_argument("manifest", type=Path)
    c.add_argument("--log", type=Path, required=True)
    c.add_argument("--run-dir", type=Path, default=None)
    c.add_argument("--unit", default=None)
    c.add_argument("--stage-dir", default=None)
    c.add_argument("--job-rc", type=int, default=0)
    args = p.parse_args(argv)

    run_dir = args.run_dir or Path(os.environ.get("SP_RUN", ""))
    if not str(run_dir):
        print("[completeness_sims] FATAL: $SP_RUN unset and --run-dir not given",
              file=sys.stderr)
        return 2
    unit = args.unit or _unit_from_run_dir(run_dir)

    manifest, ok = build_manifest(args.stage, Path(run_dir), unit, args.stage_dir)

    if args.job_rc != 0:
        ok = False
        manifest["status"] = "failed"
        manifest["job_rc"] = args.job_rc
        manifest["failures"].append({
            "runner": "shapepipe_run", "found": 0, "expect": 1,
            "warn": False, "status": "failed",
            "reasons": [f"shapepipe_run exited {args.job_rc} (counts met expect)"],
        })

    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    write_if_changed(args.log, text)
    if ok:
        write_if_changed(args.manifest, text)

    for runner, r in manifest["runners"].items():
        tag = {"complete": "OK", "warn": "warn", "failed": "<-- BELOW expect"}
        print(f"[completeness_sims]   {runner}: {r['found']}/{r['expect']} "
              f"{tag[r['status']]}", file=sys.stderr)
    print(f"[completeness_sims] {args.stage} {unit}: {manifest['status']} "
          f"-> {args.log}" + (f" + {args.manifest}" if ok else ""), file=sys.stderr)
    for f in manifest["failures"]:
        for reason in f["reasons"]:
            print(f"[completeness_sims]   {f['runner']}: {reason}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
