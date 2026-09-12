"""Tile post-chain, step 3 of the port: gather exposures, then detect / PSF /
shape / catalogue.

    tile_exp_forest
    tile_merge_headers -> tile_detect -> {tile_fake_psf, tile_vignets} -> tile_ngmix -> tile_make_cat

Differences from shapepipe/workflow/rules/tile.smk, and why -- see also the
Snakefile's module docstring:

  * tile_fake_psf has no real-data counterpart: image sims use a known,
    injected PSF (fake_psf_runner, reading the tile's own sexcat from
    tile_detect), not a fitted PSFEx model. It is a SIBLING of tile_vignets
    (both depend only on tile_detect), not a prerequisite of it -- Gen-2 ran
    them sequentially only because its bash dispatch is inherently serial;
    ngmix is the first stage that actually needs both.
  * No tile_merge_cats or ngmix chunking: Gen-2 ran ngmix as ONE call per
    tile (config_tile_Ng_batch_psfex_sx.ini), so tile_make_cat reads
    tile_ngmix's output directly. No node-local staging or group fusion
    either -- everything here runs on shared storage, plain ungrouped
    rules. Both are Phase-3 concerns once this chain is verified correct.
  * tile_detect depends on tile_merge_headers (config_tile_Sx.ini reads
    log_exp_headers as one of its inputs), which real data's tile_detect
    does not -- an image-sims-specific ordering.
"""

rule tile_exp_forest:
    input:
        tile_exp_split
    output:
        forest = directory(f"{TILE_DIR}/exp_forest")
    params:
        cmd = lambda wc: (f"python {SCRIPTS}/build_forest_sims.py --tile {wc.tile} "
                          f"--run-dir {RUN_DIR} --index {INDEX_JSON}")
    threads: 1
    resources:
        mem_mb = 2000,
        runtime = 20
    shell:
        "{params.cmd} --forest {output.forest}"


rule tile_merge_headers:
    input:
        forest = rules.tile_exp_forest.output.forest,
        split  = tile_exp_split,
        fe     = f"{TILE_DIR}/manifests/tile_find_exposures.json",
    output:
        manifest = f"{TILE_DIR}/manifests/tile_merge_headers.json"
    log:
        f"{TILE_DIR}/logs/tile_merge_headers.json"
    params:
        pre = lambda wc: unit_pre("tile_merge_headers", wc.tile,
                                  forest=forest_dir(wc.tile))
    threads: 4
    resources:
        mem_mb = lambda wc, attempt: 8000 * attempt,
        runtime = 120
    shell:
        sp_shell("tile_merge_headers", "config_tile_Mh_exp.ini")


rule tile_detect:
    input:
        uz = f"{TILE_DIR}/manifests/tile_uncompress.json",
        mh = rules.tile_merge_headers.output.manifest,
    output:
        manifest = f"{TILE_DIR}/manifests/tile_detect.json"
    log:
        f"{TILE_DIR}/logs/tile_detect.json"
    params:
        pre = lambda wc: unit_pre("tile_detect", wc.tile)
    threads: 8
    resources:
        mem_mb = lambda wc, attempt: 16000 * attempt,
        runtime = 180
    shell:
        sp_shell("tile_detect", "config_tile_Sx.ini")


rule tile_fake_psf:
    input:
        sx = rules.tile_detect.output.manifest,
    output:
        manifest = f"{TILE_DIR}/manifests/tile_fake_psf.json"
    log:
        f"{TILE_DIR}/logs/tile_fake_psf.json"
    params:
        pre = lambda wc: unit_pre("tile_fake_psf", wc.tile,
                                  pre_run=[f"export PSF_DICT='{PSF_DICT}'"])
    threads: 4
    resources:
        mem_mb = lambda wc, attempt: 8000 * attempt,
        runtime = 60
    shell:
        sp_shell("tile_fake_psf", "config_exp_psfex.ini")


rule tile_vignets:
    input:
        sx     = rules.tile_detect.output.manifest,
        forest = rules.tile_exp_forest.output.forest,
        split  = tile_exp_split,
        fe     = f"{TILE_DIR}/manifests/tile_find_exposures.json",
        uz     = f"{TILE_DIR}/manifests/tile_uncompress.json",
    output:
        manifest = f"{TILE_DIR}/manifests/tile_vignets.json"
    log:
        f"{TILE_DIR}/logs/tile_vignets.json"
    params:
        pre = lambda wc: unit_pre("tile_vignets", wc.tile,
                                  forest=forest_dir(wc.tile))
    threads: 8
    resources:
        mem_mb = lambda wc, attempt: 16000 * attempt,
        runtime = 60
    shell:
        sp_shell("tile_vignets", "config_tile_PiViVi_canfar_sx.ini")


rule tile_ngmix:
    input:
        sx      = rules.tile_detect.output.manifest,
        fpsf    = rules.tile_fake_psf.output.manifest,
        vignets = rules.tile_vignets.output.manifest,
        mh      = rules.tile_merge_headers.output.manifest,
    output:
        manifest = f"{TILE_DIR}/manifests/tile_ngmix.json"
    log:
        f"{TILE_DIR}/logs/tile_ngmix.json"
    params:
        pre = lambda wc: unit_pre("tile_ngmix", wc.tile)
    threads: 4
    retries: 2
    resources:
        mem_mb = lambda wc, attempt: 16000 * attempt,
        runtime = lambda wc, attempt: 180 * attempt
    shell:
        sp_shell("tile_ngmix", "config_tile_Ng_batch_psfex_sx.ini")


rule tile_make_cat:
    input:
        sx   = rules.tile_detect.output.manifest,
        fpsf = rules.tile_fake_psf.output.manifest,
        ng   = rules.tile_ngmix.output.manifest,
    output:
        manifest  = f"{TILE_DIR}/manifests/tile_make_cat.json",
        final_cat = f"{PROD_TILE_DIR}/final_cat-{{tile}}.fits",
    log:
        f"{TILE_DIR}/logs/tile_make_cat.json"
    params:
        pre = lambda wc: unit_pre("tile_make_cat", wc.tile)
    threads: 4
    resources:
        mem_mb = lambda wc, attempt: 8000 * attempt,
        runtime = 30
    shell:
        # Publish guarded on rc, same as the real design: a job whose
        # shapepipe_run died must never publish a catalogue for a manifest
        # completeness.py is about to refuse to write.
        sp_shell("tile_make_cat", "config_tile_Mc_psfex.ini",
                 post='if [ $rc -eq 0 ]; then\n'
                      '  mkdir -p "$(dirname {output.final_cat})"\n'
                      '  cp -f "$(ls -1 "$SP_RUN"/output/run_sp_tile_Mc_psfex'
                      '/make_cat_runner/output/final_cat*.fits | head -1)" '
                      '{output.final_cat}\n'
                      'fi\n')


# --- reclamation (see the Snakefile's module docstring) ---------------------
# Reuses shapepipe/workflow/scripts/clean_tile.py UNMODIFIED. Input is this
# tile's final_cat on the persistent root -- a tile's store has no consumer
# outside itself, so unlike clean_exposure there is no eligibility test, only
# the ordering edge. clean_tile.py's own survivor list already matches our
# layout byte for byte (tile_vignets.json, tile_find_exposures.json, and
# output/run_sp_tile_Fe/find_exposures_runner/output/exp_numbers-<dash>.txt --
# verified: our own Fe output uses exactly that dashed name already).
rule clean_tile:
    input:
        lambda wc: final_cat(wc.tile)
    output:
        tombstone = f"{TILE_DIR}/cleaned.json"
    threads: 1
    resources:
        mem_mb = 2000,
        runtime = 30
    shell:
        f"python {SHAPEPIPE_SCRIPTS}/clean_tile.py"
        " --tile-dir $(dirname {output.tombstone}) --tile {wildcards.tile}"
        " --tombstone {output.tombstone}"
