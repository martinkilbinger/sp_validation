"""PREPARE chain, step 1 of the port: one static chain per tile.

    tile_get_images -> tile_uncompress (faked) -> tile_find_exposures

See the Snakefile's module docstring for what differs from
shapepipe/workflow/rules/prepare.smk and why.
"""

rule tile_get_images:
    output:
        manifest = f"{TILE_DIR}/manifests/tile_get_images.json"
    log:
        f"{TILE_DIR}/logs/tile_get_images.json"
    params:
        pre = lambda wc: unit_pre("tile_get_images", wc.tile)
    threads: 1
    retries: 2
    resources:
        mem_mb = lambda wc, attempt: 4000 * attempt,
        runtime = 60
    shell:
        sp_shell("tile_get_images", "config_tile_Git_symlink.ini")


def _tile_uncompress_shell(stage="tile_uncompress"):
    """Not a shapepipe_run call: the image-sims weight is already
    uncompressed, so this fakes the uncompress_fits_runner output by
    symlinking it in directly (mirrors run_job_sp_canfar_v2.0.bash's job-2
    special case for image_sims), then goes through the same completeness
    contract as every other stage with --job-rc 0.

    Also registers the fake run in $SP_RUN/output/log_run_sp.txt, in the
    same "<run dir> <module>" format a real shapepipe_run call appends --
    ShapePipe's own input-resolution ("last:uncompress_fits_runner", read by
    config_tile_Sx.ini) walks that registry, not the filesystem, so a run
    that never calls shapepipe_run must still register itself there or every
    downstream `last:`/exact-name reference to it fails with "No previous
    run of module 'uncompress_fits_runner' found" even though the output
    directory genuinely exists.
    """
    _, subdir = STAGE_DIR[stage]
    return (
        "{params.pre}\n"
        "rc=0\n"
        f'run_dir="$SP_RUN/output/{subdir}"\n'
        'out="$run_dir/uncompress_fits_runner/output"\n'
        'mkdir -p "$out"\n'
        'weight_src="$SP_DIR/input_tiles/{params.weight_name}"\n'
        'if [ -e "$weight_src" ]; then\n'
        '  ln -sf "$weight_src" "$out/{params.weight_name}"\n'
        '  printf "%s %s\\n" "$run_dir" uncompress_fits_runner >> "$SP_RUN/output/log_run_sp.txt"\n'
        "else\n"
        '  echo "tile_uncompress: missing $weight_src" >&2\n'
        "  rc=1\n"
        "fi\n"
        f"python {SCRIPTS}/completeness_sims.py check {stage} {{output.manifest}}"
        ' --log {log} --job-rc "$rc" || rc=1\n'
        "exit $rc\n"
    )


rule tile_uncompress:
    input:
        rules.tile_get_images.output.manifest
    output:
        manifest = f"{TILE_DIR}/manifests/tile_uncompress.json"
    log:
        f"{TILE_DIR}/logs/tile_uncompress.json"
    params:
        pre = lambda wc: unit_pre("tile_uncompress", wc.tile),
        weight_name = lambda wc: f"CFIS_simu_weight-{tile_dash(wc.tile)}.fits"
    threads: 1
    resources:
        mem_mb = 2000,
        runtime = 15
    shell:
        _tile_uncompress_shell()


rule tile_find_exposures:
    input:
        rules.tile_uncompress.output.manifest
    output:
        manifest = f"{TILE_DIR}/manifests/tile_find_exposures.json"
    log:
        f"{TILE_DIR}/logs/tile_find_exposures.json"
    params:
        pre = lambda wc: unit_pre("tile_find_exposures", wc.tile)
    threads: 1
    resources:
        mem_mb = lambda wc, attempt: 2000 * attempt,
        runtime = 30
    shell:
        sp_shell("tile_find_exposures", "config_tile_Fe.ini")
