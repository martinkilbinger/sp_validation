"""Exposure chain, step 2 of the port: per exposure.

    exp_get_images -> exp_split

NO exp_psf here, unlike shapepipe/workflow/rules/exposure.smk -- image sims
use a known, injected PSF (fake_psf_runner), not a fitted PSFEx model.
Gen-2's own job dispatch treats exposure-level PSF (job 64) as a pure
placeholder for image_sims ("fake PSF runs as part of job 512"); the real
fake_psf call happens at the TILE level, added in tile.smk (the next step),
reading the tile's own sexcat rather than any exposure product.

Retrieval is symlink (config_exp_Gie_symlink.ini), not VOSpace.
"""

rule exp_get_images:
    output:
        manifest = f"{EXP_DIR}/manifests/exp_get_images.json"
    log:
        f"{EXP_DIR}/logs/exp_get_images.json"
    params:
        pre = lambda wc: unit_pre("exp_get_images", wc.exp)
    threads: 1
    retries: 2
    resources:
        mem_mb = lambda wc, attempt: 4000 * attempt,
        runtime = 60
    shell:
        sp_shell("exp_get_images", "config_exp_Gie_symlink.ini")


rule exp_split:
    input:
        rules.exp_get_images.output.manifest
    output:
        manifest = f"{EXP_DIR}/manifests/exp_split.json"
    log:
        f"{EXP_DIR}/logs/exp_split.json"
    params:
        pre = lambda wc: unit_pre("exp_split", wc.exp)
    threads: 8
    resources:
        mem_mb = lambda wc, attempt: 8000 * attempt,
        runtime = 120
    shell:
        sp_shell("exp_split", "config_exp_Sp.ini")


# --- reclamation (see the Snakefile's module docstring) ---------------------
# Reuses shapepipe/workflow/scripts/clean_exposure.py UNMODIFIED. Input is
# every IN-SCOPE consuming tile's tile_vignets manifest -- vignets is the
# last stage that reads exposure products here too (tile_ngmix/tile_make_cat
# read only tile-level output) -- so writer, then readers, then cleaner, is
# the whole ordering argument, DAG-enforced.
rule clean_exposure:
    input:
        lambda wc: [tile_manifest(t, "tile_vignets")
                    for t in clean_consumers(wc.exp) if t in READY_SET]
    output:
        tombstone = f"{EXP_DIR}/cleaned.json"
    params:
        consumers = lambda wc: ",".join(clean_consumers(wc.exp))
    threads: 1
    resources:
        mem_mb = 2000,
        runtime = 30
    shell:
        f"python {SHAPEPIPE_SCRIPTS}/clean_exposure.py"
        " --exp-dir $(dirname {output.tombstone}) --exp {wildcards.exp}"
        " --tombstone {output.tombstone} --consumers '{params.consumers}'"
