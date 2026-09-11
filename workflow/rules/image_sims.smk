"""Image-simulation orchestration: raw SKiLLS sim images -> shear m/c bias.

This rule set drives the image-simulation validation chain end to end and is
the sp_validation-side half of the split described in
``UNIONS-WL/MultiBand_ImSim#1``: ShapePipe turns the simulated tiles into
per-tile shape catalogues, then sp_validation merges, extracts, calibrates and
finally measures the multiplicative/additive shear bias.

One ShapePipe processing definition.  The per-tile ShapePipe processing (tile
and exposure stages, up to ``final_cat``) is NOT defined here: ``im_shapepipe``
drives ShapePipe's own Snakemake workflow (``workflow/bin/sp run`` in
``shapepipe_repo``) once per shear branch, with ``input_type: image_sims``.
The sims therefore pass through the same module order, config chain and
PSF-model switch as the real catalogue -- the overlay config dir
``workflow/config/cfis_image_sims`` differs from the real-data one only in
input naming, the fake-PSF INIs and the merge column list -- so the m measured
here calibrates the pipeline that is actually used.  ``psf_model`` is ``fake``
(the true simulation PSF; star-free ``*_grid_N`` sims) or ``psfex``/``mccd``
(sims with stars), exactly as ShapePipe's workflow spells it.

Two images, one prefix shape.  Architecturally one image could run every
stage -- the sp_validation image is built ``FROM`` the ShapePipe image, so it
carries both stacks -- but the *published* sp_validation image's environment
is not yet trustworthy for the ShapePipe half: sp_validation has no lockfile
and does not declare its numba-bearing dependency (``cosmo_numba``), so
unpinned install layers can drift NumPy past numba's window (a 2026-07-11
gate run hit exactly this: ``Numba needs NumPy 2.4 or less. Got NumPy 2.5``
at the ngmix stage).  PYTHONPATH shadowing covers pure-Python *code*, never
binary deps, so until sp_validation is uv-locked with its deps declared
(spun off as its own task), each half runs in its own repo's image -- the
same split the gate766 baseline ran:

* ShapePipe stages -> ``shapepipe`` (raw images -> per-tile cats, via ``sp
  run``; ``sif_pipeline`` is the run config's ``container:``) and ``merge``
  (``create_final_cat`` -> ``final_cat_{sim}.hdf5``) run in ``sif_pipeline``
  (the ShapePipe image).
* sp_validation stages -> ``manifest``, ``extract`` (-> comprehensive cat),
  ``calibrate`` (-> cut cat) and ``m_bias`` (-> ``m_bias_results.yaml``) run
  in ``sif`` (the sp_validation image).

Every rule sets ``container: None`` and calls ``apptainer exec`` explicitly
through a shared prefix template (``EXEC_PIPELINE`` / ``EXEC`` -- identical
env injections, different image), because the images are not the workflow's
top-level container.  ``im_shapepipe`` is the exception: ``sp`` owns its jobs'
container through ShapePipe's own profile (``sp_profile``).  Everything is parameterised under
``config["image_sims"]`` -- the two ``sif`` keys, repository roots, data roots,
the PSF dictionary, the explicit ``tile_ids`` list and the sim/calibration
knobs -- so a fresh user drives it from config alone, with no hard-coded clone
layout.  Configuration is fail-fast: a schema check at load rejects an unknown
key (typo) and a missing science key (see ``workflow/image_sims/config.yaml``
for the operational/science split).  The ``PYTHONPATH`` override injects both
repos' ``src`` so the *branch* source (ShapePipe's ``#766`` build;
sp_validation's ``image_sims.py``, ``catalog.match_catalogs_radec``) wins over
whatever is baked into the image.

The five simulations per grid are the reference ``1z2z`` (no input shear) plus
the ``+/-`` shear pairs ``1p2z``/``1m2z`` (g1) and ``1z2p``/``1z2m`` (g2); the
m-bias estimator matches each to the reference by RA/Dec.
"""

import os
from pathlib import Path

IMSIM = config["image_sims"]

# --- fail-fast schema check ----------------------------------------------
# One home for every fact: the run config carries the science knobs, the
# workflow config.yaml carries the operational defaults, and *this* block is
# where a typo or a missing knob dies -- at DAG parse, before any compute.
#
# Every key must be declared below.  An unknown key under ``image_sims:`` is a
# hard error (typo protection); a missing *science* key is a hard error naming
# the key (no silent code default anywhere).  Operational keys default in the
# workflow config.yaml and nowhere else: the .smk reads them as bare
# ``IMSIM[key]`` (never ``.get`` with a second literal), so their value comes
# from config.yaml alone -- the single home for an operational default.
#
# Science keys: required from the *run* config; no default in config.yaml (only
# a commented template line) and no default in code.  These fix the estimator's
# scientific behaviour, so they must be stated per run, never inherited.
_SCIENCE_KEYS = {
    "w_cols",
    "pair_match",
    "match_radius_deg",
    "n_bootstrap",
    "bootstrap_seed",
    "mask_config",
}
# Deprecated science keys: accepted (so a pre-``w_cols`` run config still parses
# and the estimator's back-compat path runs) but not *required* -- our configs
# state ``w_cols``.  Listed here only to keep them out of the unknown-key error.
_DEPRECATED_KEYS = {
    "w_col",
}
# Operational keys: default (visibly) in the workflow config.yaml; the .smk
# reads them bare, so config.yaml is their one home.
_OPERATIONAL_KEYS = {
    "binds",
    "sims_type",
    "branches",
    "shape",
    "psf_model",
    "sp_profile",
    "sp_jobs",
    "clean_exposures",
    "extract_script",
    "calibrate_script",
}
# Optional keys: recognized but not required; defaults applied in code below.
_OPTIONAL_KEYS = {
    "exp_num",
}
# Structural keys: paths/identifiers the run must supply (no sensible default).
_STRUCTURAL_KEYS = {
    "sif",
    "sif_pipeline",
    "shapepipe_repo",
    "sp_validation_repo",
    "grids_base",
    "input_sims_base",
    "psf_dict",
    "num",
    "tile_ids",
}
_ALLOWED_KEYS = (
    _SCIENCE_KEYS
    | _DEPRECATED_KEYS
    | _OPERATIONAL_KEYS
    | _OPTIONAL_KEYS
    | _STRUCTURAL_KEYS
)

_unknown = set(IMSIM) - _ALLOWED_KEYS
if _unknown:
    raise ValueError(
        "image_sims: unknown config key(s) "
        f"{sorted(_unknown)} -- check for a typo (allowed keys: "
        f"{sorted(_ALLOWED_KEYS)})"
    )
_missing_science = sorted(_SCIENCE_KEYS - set(IMSIM))
if _missing_science:
    raise ValueError(
        "image_sims: missing required science key(s) "
        f"{_missing_science} -- these have no default and must be set in the "
        "run config (see the commented template in workflow/image_sims/config.yaml)"
    )
_missing_structural = sorted(_STRUCTURAL_KEYS - set(IMSIM))
if _missing_structural:
    raise ValueError(
        "image_sims: missing required key(s) "
        f"{_missing_structural} -- set them in the run config"
    )

# --- containers -----------------------------------------------------------
# Two images (see module docstring): the ShapePipe image for the pipeline and
# merge stages, the sp_validation image for everything downstream.  Collapse
# back to one image once sp_validation's env is lock-managed.
SIF = IMSIM["sif"]  # sp_validation stages
SIF_PIPELINE = IMSIM["sif_pipeline"]  # ShapePipe stages
BINDS = IMSIM["binds"]

# --- repositories (bound into the image; branch code overrides) -----------
SHAPEPIPE_REPO = IMSIM["shapepipe_repo"]
SPV_REPO = IMSIM["sp_validation_repo"]

# --- data and run directories --------------------------------------------
GRIDS_BASE = IMSIM["grids_base"]  # run/output root; one sub-dir per sim
INPUT_SIMS_BASE = IMSIM["input_sims_base"]  # SKiLLS sim images
PSF_DICT = IMSIM["psf_dict"]  # Herve's Full_psf_dict.pickle

# --- simulation grid ------------------------------------------------------
# SIM_BASES is the set of branches *this run requests* -- the reference plus the
# four +/- sheared branches.  Their injected shear (amplitude, per-branch
# (g1,g2), pairing) is NOT a literal here: it lives only in manifest.yaml, built
# by im_manifest from each branch's basic_info.txt and read back by im_mbias.
NUM = IMSIM["num"]
SIMS_TYPE = IMSIM["sims_type"]
_SUFFIX = f"_{SIMS_TYPE}_{NUM}" if SIMS_TYPE == "grid" else f"_{NUM}"
SIM_BASES = list(IMSIM["branches"])
SIMS = [f"{base}{_SUFFIX}" for base in SIM_BASES]
# Optional ``exp_num`` pulls exposures from a different realization than tiles:
# tiles stay on ``{branch}_{num}``, exposures come from ``{branch}_{exp_num}``.
# Non-grid sims only (grid names embed sims_type); see config.yaml for the
# blended use case.
EXP_NUM = IMSIM.get("exp_num", NUM)
if SIMS_TYPE == "grid":
    SIM_EXP_BASE = {sim: sim for sim in SIMS}
else:
    SIM_EXP_BASE = {f"{base}{_SUFFIX}": f"{base}_{EXP_NUM}" for base in SIM_BASES}
MANIFEST = f"{GRIDS_BASE}/manifest.yaml"
BUILD_MANIFEST = f"{SPV_REPO}/workflow/scripts/im_build_manifest.py"

# --- tiles ----------------------------------------------------------------
# tile_ids is the one tile-input mechanism: an explicit list in the run config.
TILE_IDS = list(IMSIM["tile_ids"])

# --- calibration / m-bias knobs ------------------------------------------
SHAPE = IMSIM["shape"]
MASK_CONFIG = IMSIM["mask_config"]  # e.g. config/calibration/mask_v1.X.9_im_sim.yaml
PARAMS_TEMPLATE = f"{SPV_REPO}/workflow/image_sims/params_im_sim.py"

# --- ShapePipe processing: ShapePipe's own workflow ------------------------
# ``sp`` is ShapePipe's workflow launcher; SP_CONFIG_DIR is the config chain it
# selects for ``input_type: image_sims``.  The merge and extract steps read that
# dir's ``final_cat.param`` (through the per-sim ``cfis`` link), so the column
# list and the processing that wrote the columns come from one checkout.
SP_LAUNCHER = f"{SHAPEPIPE_REPO}/workflow/bin/sp"
SP_CONFIG_DIR = f"{SHAPEPIPE_REPO}/workflow/config/cfis_image_sims"
PSF_MODEL = IMSIM["psf_model"]
SP_PROFILE = IMSIM["sp_profile"]
SP_JOBS = IMSIM["sp_jobs"]
CLEAN_EXPOSURES = bool(IMSIM["clean_exposures"])

# ShapePipe scripts live in the ShapePipe repo (also baked into its image).
CREATE_FINAL_CAT = f"{SHAPEPIPE_REPO}/scripts/python/create_final_cat.py"
# Extract/calibrate run from the sp_validation *repo* checkout (bind-mounted),
# not the baked copies: the container tracks the branch but lags it, and the
# image-sims path needs branch-only fixes (star-catalogue-optional extract,
# FITS-aware CalibrateCat.read_cat). Overridable for a different checkout.
EXTRACT_INFO = IMSIM["extract_script"]
CALIBRATE = IMSIM["calibrate_script"]
# m-bias is *this branch's* extracted core, injected on PYTHONPATH.
COMPUTE_M_BIAS = f"{SPV_REPO}/scripts/compute_m_bias_image_sims.py"

# --- container exec prefixes ----------------------------------------------
# One prefix *shape* for every containerised stage (all but im_shapepipe) --
# two instances, one per image.  The env injections make the on-disk branch
# code win over the image's baked copies:
#
#   * PYTHONPATH prepends BOTH repos' ``src`` (ShapePipe first, then
#     sp_validation), so Python resolves the worktree build before
#     ``/app``/``/sp_validation`` -- the local-testing counterpart of the
#     git-ref deps, letting the branch code run without an image rebuild.  This
#     covers the Python *packages* only: the ShapePipe/sp_validation *scripts*
#     are still invoked at the repo paths resolved from config
#     (CREATE_FINAL_CAT, EXTRACT_INFO, ...), not shadowed by PYTHONPATH.
#
# The SLURM env vars are stripped (``env -u ...``) so that an MPI stack
# initialising inside the image does not try to attach to the host SLURM
# launcher (cf. apptainer_noslurm.sh); harmless for these pure-Python stages.
#
# ``OMP_NUM_THREADS=1`` is injected here, at the ``apptainer exec`` call, and
# not left to the SLURM profile: the slurm executor submits with
# ``--export=ALL``, which propagates the *driver's* ambient environment, and a
# Snakemake profile only sets CLI flags, never the driver's own env, so an
# ``OMP_NUM_THREADS`` there would depend on the operator having exported it by
# hand.  Injecting it on the ``apptainer exec`` line puts it where the compute
# actually runs.  (ShapePipe's own profile pins it the same way for the jobs
# ``sp`` submits.)
_EXEC_PREFIX = (
    "env -u SLURM_JOBID -u SLURM_JOB_ID -u SLURM_PROCID "
    f"apptainer exec --bind {BINDS} "
    f"--env PYTHONPATH={SHAPEPIPE_REPO}/src:{SPV_REPO}/src "
    "--env OMP_NUM_THREADS=1 "
)
EXEC = _EXEC_PREFIX + SIF  # sp_validation stages
EXEC_PIPELINE = _EXEC_PREFIX + SIF_PIPELINE  # ShapePipe merge stage


def sp_run_config(sim):
    """ShapePipe workflow run config for one shear branch (one ``sp`` campaign).

    It REPLACES ShapePipe's committed ``workflow/config.yaml`` (``bin/sp``'s
    ``SP_RUN_CONFIG``), so every key the campaign needs is stated; processing
    knobs this config does not state (``ngmix_chunks``) take the ShapePipe
    Snakefile's defaults, never a copy here.  ``clean_tiles`` stays off: the
    merge reads each tile's make_cat output from the tile store.  ``clean``
    (rolling exposure-store reclamation) is ``clean_exposures`` -- a 40-tile
    branch touches ~300 exposures at ~8 GB of stores each.
    """
    run_dir = f"{GRIDS_BASE}/{sim}"
    cfg = {
        "input_type": "image_sims",
        "psf_model": PSF_MODEL,
        "tile_list": f"{run_dir}/tiles_{sim}.txt",
        "inputs": {
            "tiles": f"{INPUT_SIMS_BASE}/{sim}/images/SP_tiles",
            "exposures": f"{INPUT_SIMS_BASE}/{SIM_EXP_BASE[sim]}/images/SP_exp",
        },
        "container": SIF_PIPELINE,
        "outputs": {
            "run_dir": run_dir,
            "index_db": f"{run_dir}/index/run_index.sqlite",
        },
        "clean": CLEAN_EXPOSURES,
        "clean_tiles": False,
        "clean_ignore_tiles": [],
    }
    if PSF_MODEL == "fake":
        cfg["psf_dict"] = PSF_DICT
    return cfg


localrules:
    im_shapepipe,


wildcard_constraints:
    sim="|".join(SIMS),
    tile="|".join(t.replace(".", r"\.") for t in TILE_IDS),


# ==========================================================================
# Convenience targets (run in order)
# ==========================================================================
rule im_manifest_only:
    input:
        MANIFEST,


rule im_init_all:
    input:
        expand(f"{GRIDS_BASE}/{{sim}}/params.py", sim=SIMS),


rule im_shapepipe_all:
    input:
        expand(f"{GRIDS_BASE}/{{sim}}/logs/shapepipe_campaign.yaml", sim=SIMS),


rule im_merge_all:
    input:
        expand(f"{GRIDS_BASE}/{{sim}}/final_cat_{{sim}}.hdf5", sim=SIMS),


rule im_extract_all:
    input:
        expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_comprehensive_{SHAPE}.fits",
            sim=SIMS,
        ),


rule im_calibrate_all:
    input:
        expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits", sim=SIMS
        ),


# ==========================================================================
# Rules
# ==========================================================================
rule im_manifest:
    """Build the campaign manifest at the head of the DAG.

    Parses ``g_cosmic`` from every requested branch's ``basic_info.txt``,
    cross-checks each against its ``1{X}2{Y}`` name and the (0,0) reference,
    derives the single injected amplitude, and writes ``manifest.yaml`` into the
    run root.  This is the one home for the injected-shear facts; im_mbias reads
    the amplitude and branch map from here, nowhere else.  Pure sp_validation
    stage (stdlib parse of basic_info; PyYAML to write).
    """
    input:
        # basic_info.txt for each requested branch, so editing a sim's record
        # rebuilds the manifest (and re-validates) rather than reusing a stale one.
        basic_info=expand(
            f"{INPUT_SIMS_BASE}/{{sim}}/basic_info.txt", sim=SIMS
        ),
    output:
        manifest=MANIFEST,
    params:
        branch_args=lambda wc: " ".join(f"--branch {b}" for b in SIM_BASES),
        input_sims_base=INPUT_SIMS_BASE,
        sims_type=SIMS_TYPE,
        num=NUM,
    shell:
        "{EXEC} python {BUILD_MANIFEST} "
        "--input-sims-base {params.input_sims_base} "
        "--sims-type {params.sims_type} --num {params.num} "
        "{params.branch_args} -o {output.manifest}"


rule im_init:
    """Stage per-sim run directory: params.py, mask config, tile list, and the
    ``cfis`` link to ShapePipe's image-sims config dir.

    ``params_im_sim.py`` derives the field name from the directory basename, so
    the same template serves every sim; ``config_mask.yaml`` and ``cfis`` are
    symlinks the downstream calibration and merge steps read from cwd.
    ``tiles_{sim}.txt`` is both the ShapePipe campaign's ``tile_list`` and the
    tile-ID file ``params.py`` names for the found/missing-tile check.
    """
    input:
        # Tracked so that editing the params template or mask config re-stages
        # them into every run dir (a plain params: value would not retrigger,
        # silently leaving stale params.py behind after a grammar change).
        template=PARAMS_TEMPLATE,
        mask_src=os.path.join(SPV_REPO, MASK_CONFIG),
    output:
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
        mask=f"{GRIDS_BASE}/{{sim}}/config_mask.yaml",
        tiles=f"{GRIDS_BASE}/{{sim}}/tiles_{{sim}}.txt",
    params:
        config_dir=SP_CONFIG_DIR,
        cfis=lambda wc: f"{GRIDS_BASE}/{wc.sim}/cfis",
        tile_ids=" ".join(TILE_IDS),
    shell:
        # cfis is a stable read-only symlink (used by merge and extract);
        # created here but not tracked as an output, which snakemake will not
        # accept for a symlink to a directory.
        "mkdir -p $(dirname {output.params}) && "
        "cp {input.template} {output.params} && "
        "ln -sf {input.mask_src} {output.mask} && "
        "ln -sfT {params.config_dir} {params.cfis} && "
        "printf '%s\\n' {params.tile_ids} > {output.tiles}"


rule im_shapepipe:
    """Run ShapePipe's own workflow over one shear branch (ShapePipe stage).

    Writes the branch's ShapePipe run config (``sp_run_config``) and runs
    ``sp run`` on it: one campaign, PREPARE then COMPUTE, whose jobs ``sp``
    submits to SLURM itself through ShapePipe's profile ``sp_profile``.  That
    is why this is a localrule -- it is a scheduler, not a compute job.  The
    campaign resumes: rerunning after a failure re-does only incomplete units.

    The output is the campaign record: which ShapePipe checkout and image
    produced the per-tile catalogues, and how many of the requested tiles have
    a ``final_cat``.  ``sp run`` exits non-zero if any tile failed (keep-going
    lets the rest finish), and so does this rule; read the campaign's
    ``index/run_report.json`` for the per-stage verdicts.
    """
    input:
        tiles=f"{GRIDS_BASE}/{{sim}}/tiles_{{sim}}.txt",
    output:
        record=f"{GRIDS_BASE}/{{sim}}/logs/shapepipe_campaign.yaml",
    log:
        f"{GRIDS_BASE}/{{sim}}/logs/sp_run.log",
    resources:
        # One unit per concurrent campaign; the profile caps the total
        # (``resources: sp_campaign=N``).  The cap is a DISK bound: a branch's
        # exposure stores (~8 GB each, ~300 per 40-tile branch) are reclaimed
        # only once every tile reading them has its vignets, and snakemake
        # runs the ready exposure jobs first, so a campaign can hold ~2 TB at
        # its peak.
        sp_campaign=1,
    params:
        run_config=lambda wc: sp_run_config(wc.sim),
        run_config_path=lambda wc: f"{GRIDS_BASE}/{wc.sim}/shapepipe_run.yaml",
    run:
        import subprocess

        import yaml

        with open(params.run_config_path, "w") as fh:
            fh.write(
                "# ShapePipe workflow run config, written by sp_validation's\n"
                "# im_shapepipe rule (workflow/rules/image_sims.smk).\n"
            )
            yaml.safe_dump(params.run_config, fh, sort_keys=False)
        env = (
            f"SP_PROFILE={SP_PROFILE} "
            f"SP_RUN_CONFIG={params.run_config_path}"
        )
        # Appended, not truncated: a retry (or a relaunch of the campaign)
        # must not erase the log of the attempt that failed.
        shell(
            f"echo \"=== sp run $(date -Iseconds) ===\" >> {log} && "
            f"{env} {SP_LAUNCHER} run --jobs {SP_JOBS} >> {log} 2>&1"
        )

        def _out(*cmd):
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True
            ).stdout.strip()

        run_dir = params.run_config["outputs"]["run_dir"]
        tiles = [t.strip() for t in open(input.tiles) if t.strip()]
        final_cats = [
            t for t in tiles
            if os.path.exists(f"{run_dir}/tiles/{t[:2]}/{t}/final_cat-{t}.fits")
        ]
        record = {
            "run_config": params.run_config_path,
            "sp_profile": SP_PROFILE,
            "shapepipe": {
                "repo": SHAPEPIPE_REPO,
                "branch": _out("git", "-C", SHAPEPIPE_REPO, "rev-parse", "--abbrev-ref", "HEAD"),
                "commit": _out("git", "-C", SHAPEPIPE_REPO, "rev-parse", "HEAD"),
                "dirty": bool(_out("git", "-C", SHAPEPIPE_REPO, "status", "--porcelain")),
            },
            # sp's own image resolution (sandbox > cached SIF > ``container:``),
            # i.e. the image the campaign's jobs actually ran in.
            "image": _out(
                "env", f"SP_PROFILE={SP_PROFILE}",
                f"SP_RUN_CONFIG={params.run_config_path}",
                SP_LAUNCHER, "container", "resolve",
            ).splitlines()[-1],
            "run_report": f"{run_dir}/index/run_report.json",
            "n_tiles": len(tiles),
            "n_final_cats": len(final_cats),
        }
        with open(output.record, "w") as fh:
            yaml.safe_dump(record, fh, sort_keys=False)


rule im_merge:
    """Merge per-tile ShapePipe catalogues into final_cat_{sim}.hdf5.

    ``create_final_cat.py`` lives in the ShapePipe repo/image; run in image_sims
    mode (``-I``) it walks the campaign's tile stores under the run directory
    (``tiles/<shard>/<tile>/output/run_sp_tile_Mc``), selecting the columns in
    ShapePipe's image-sims ``final_cat.param``.
    """
    input:
        record=f"{GRIDS_BASE}/{{sim}}/logs/shapepipe_campaign.yaml",
    output:
        cat=f"{GRIDS_BASE}/{{sim}}/final_cat_{{sim}}.hdf5",
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
    shell:
        "cd {params.run_dir} && "
        "{EXEC_PIPELINE} python {CREATE_FINAL_CAT} "
        "-I -m final_cat_{wildcards.sim}.hdf5 -i .. "
        "-p cfis/final_cat.param -P {wildcards.sim} "
        "-o n_tiles_final.txt -v"


rule im_extract:
    """Extract the comprehensive ngmix catalogue (sp_validation stage).

    ``extract_info.py`` reads ``params.py`` from cwd and the merged catalogue,
    writing ``shape_catalog_comprehensive_{shape}``.
    """
    input:
        cat=f"{GRIDS_BASE}/{{sim}}/final_cat_{{sim}}.hdf5",
        params=f"{GRIDS_BASE}/{{sim}}/params.py",
    output:
        cat=f"{GRIDS_BASE}/{{sim}}/shape_catalog_comprehensive_{SHAPE}.fits",
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
    shell:
        "cd {params.run_dir} && {EXEC} python {EXTRACT_INFO}"


rule im_calibrate:
    """Calibrate and cut the comprehensive catalogue (sp_validation stage).

    ``calibrate_comprehensive_cat.py`` reads ``config_mask.yaml`` from cwd,
    applies the metacal calibration and selection, and writes
    ``shape_catalog_cut_{shape}.fits``.
    """
    input:
        cat=f"{GRIDS_BASE}/{{sim}}/shape_catalog_comprehensive_{SHAPE}.fits",
        mask=f"{GRIDS_BASE}/{{sim}}/config_mask.yaml",
    output:
        cat=f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits",
    params:
        run_dir=lambda wc: f"{GRIDS_BASE}/{wc.sim}",
    shell:
        "cd {params.run_dir} && "
        "{EXEC} python {CALIBRATE} -s calibrate"


rule im_mbias:
    """Multiplicative/additive shear bias from the calibrated grids.

    Produces the workflow's headline artifact, ``m_bias_results.yaml``.  The
    injected shear (``shear_amplitude`` and the branch map) comes from
    ``manifest.yaml`` alone -- no literal amplitude here or in config.yaml.  The
    generated ``m_bias_config.yaml`` carries the manifest's ``branches`` and
    ``pairs``, so the estimator's sim list and pairing are the campaign's, not a
    hard-coded default.
    """
    input:
        manifest=MANIFEST,
        cats=expand(
            f"{GRIDS_BASE}/{{sim}}/shape_catalog_cut_{SHAPE}.fits", sim=SIMS
        ),
    output:
        results=f"{GRIDS_BASE}/results/m_bias_results.yaml",
    resources:
        # Blended catalogues are ~5x grid size; the snakemake 1000M default OOMs.
        mem_mb=8000,
    params:
        cfg=f"{GRIDS_BASE}/results/m_bias_config.yaml",
        grids_base=GRIDS_BASE,
        num=NUM,
        sims_type=SIMS_TYPE,
        cat_name=f"shape_catalog_cut_{SHAPE}.fits",
        sif=SIF,
        sif_pipeline=SIF_PIPELINE,
        shapepipe_repo=SHAPEPIPE_REPO,
        sp_validation_repo=SPV_REPO,
        # Science knobs, read bare from the run config (no default here).
        match_radius_deg=IMSIM["match_radius_deg"],
        w_cols=IMSIM["w_cols"],
        n_bootstrap=IMSIM["n_bootstrap"],
        pair_match=IMSIM["pair_match"],
        bootstrap_seed=IMSIM["bootstrap_seed"],
    run:
        import hashlib
        import re
        import subprocess

        import yaml

        with open(input.manifest) as fh:
            manifest = yaml.safe_load(fh)

        def _git(repo, *args):
            """Read a git fact from ``repo``; ``None`` if it is not a checkout."""
            try:
                return subprocess.run(
                    ["git", "-C", repo, *args],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

        def _sif_revision(sif_path):
            """GHCR revision baked into the SIF's OCI labels.

            A plain-text scan of the image file (login-safe: no exec, no
            container start), reading org.opencontainers.image.revision -- the
            source commit GHCR built the image from.  ``None`` if absent.
            """
            try:
                with open(sif_path, "rb") as fh:
                    blob = fh.read()
            except OSError:
                return None
            m = re.search(
                rb'org\.opencontainers\.image\.revision"?[:=]"?([0-9a-f]{7,40})',
                blob,
            )
            return m.group(1).decode() if m else None

        # Manifest hash: sha256 of the exact bytes im_manifest wrote, so the
        # result records which injected-shear facts it was computed against.
        with open(input.manifest, "rb") as fh:
            manifest_sha256 = hashlib.sha256(fh.read()).hexdigest()

        provenance = {
            "manifest_sha256": manifest_sha256,
            "sp_validation": {
                "branch": _git(params.sp_validation_repo, "rev-parse", "--abbrev-ref", "HEAD"),
                "commit": _git(params.sp_validation_repo, "rev-parse", "HEAD"),
            },
            "shapepipe": {
                "branch": _git(params.shapepipe_repo, "rev-parse", "--abbrev-ref", "HEAD"),
                "commit": _git(params.shapepipe_repo, "rev-parse", "HEAD"),
            },
            "containers": {
                "sif": params.sif,
                "ghcr_revision": _sif_revision(params.sif),
                "sif_pipeline": params.sif_pipeline,
                "ghcr_revision_pipeline": _sif_revision(params.sif_pipeline),
            },
        }

        os.makedirs(os.path.dirname(output.results), exist_ok=True)
        # Emit *every* key the estimator requires -- pair_match and
        # bootstrap_seed included.  Requiring a key without emitting it would
        # be a KeyError at run time, so the generated config is the complete
        # contract between rule and estimator.  ``provenance`` rides along as a
        # top-level block: the compute script copies it verbatim into the output
        # results yaml, so a result file is self-describing (which manifest,
        # which repo commits, which container built the number).
        mbias_cfg = {
            "grids_dir": params.grids_base,
            "num": params.num,
            "sims_type": params.sims_type,
            "catalog_name": params.cat_name,
            # Injected shear: from the manifest, the single source of truth.
            "shear_amplitude": manifest["shear_amplitude"],
            "branches": list(manifest["branches"]),
            "pairs": manifest["pairs"],
            "match_radius_deg": params.match_radius_deg,
            "w_cols": list(params.w_cols),
            "pair_match": params.pair_match,
            "n_bootstrap": params.n_bootstrap,
            "bootstrap_seed": params.bootstrap_seed,
            "results_dir": os.path.dirname(output.results),
            "output_path": output.results,
            "provenance": provenance,
        }
        with open(params.cfg, "w") as fh:
            yaml.safe_dump(mbias_cfg, fh)
        shell(
            "{EXEC} python {COMPUTE_M_BIAS} -c {params.cfg} -v"
        )
