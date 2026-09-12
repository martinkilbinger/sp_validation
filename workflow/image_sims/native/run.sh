#!/usr/bin/env bash
# Launcher for the native image-sims workflow -- the sp_validation-side
# analogue of shapepipe/workflow/bin/sp. Runs the two SP_PHASE invocations
# (prepare, compute) that a static Snakemake DAG needs because the exposure
# list is only known after tile_find_exposures actually runs -- see the
# Snakefile's module docstring and CLAUDE.md's SP_PHASE explanation.
#
# A run config is REQUIRED (-c/--config): everything that changes from run
# to run or user to user (tile_list, sim, inputs, the run dir, the
# container) has no default in the committed config.yaml -- see
# run.example.yaml for the template. Keep your actual run config and tile
# list OUTSIDE this repo, in your own run dir.
#
# Usage:
#   ./run.sh -c my_run.yaml                    # both phases, in order
#   ./run.sh prepare -c my_run.yaml [-n]       # just PREPARE
#   ./run.sh compute -c my_run.yaml [-n]       # just COMPUTE
set -euo pipefail

usage() {
  echo "Usage: $0 [prepare|compute] -c <run-config.yaml> [extra snakemake args]" >&2
  exit 1
}

# --- argument parsing (before the cd below, so a relative -c path resolves
# against the CALLER's directory, not this script's) -------------------------
PHASE_ARG="both"
if [[ "${1:-}" == "prepare" || "${1:-}" == "compute" ]]; then
  PHASE_ARG=$1
  shift
fi

RUN_CONFIG=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) RUN_CONFIG=${2:-}; shift 2 ;;
    -h|--help) usage ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -z "$RUN_CONFIG" ]] && { echo "Missing -c/--config <run-config.yaml>" >&2; usage; }
[[ -f "$RUN_CONFIG" ]] || { echo "run config not found: $RUN_CONFIG" >&2; exit 1; }
RUN_CONFIG=$(readlink -f "$RUN_CONFIG")

cd "$(dirname "${BASH_SOURCE[0]}")"

# --- cluster / container settings, shared by both phases -------------------
PARTITION=comp
ACCOUNT=cusers
EXCLUDE=n17,n09,n36            # flaky candide nodes (n17 mount issues, n09 no
                                # internet, n36) -- matches sp_validation's
                                # own workflow/profiles/candide/config.yaml
BIND=/home,/scratch,/automnt,/n17data,/n23data1,/n09data
SHAPEPIPE_SRC=/n17data/mkilbing/astro/repositories/github/shapepipe/src

APPTAINER_ARGS="--cleanenv --bind $BIND --env PYTHONPATH=$SHAPEPIPE_SRC"

# --- per-phase resources -----------------------------------------------------
# PREPARE is three light rules per tile (Git/Uz/Fe). COMPUTE now runs the
# WHOLE exposure + tile chain in one srun allocation (local `--cores`
# execution, not the SLURM executor -- rule-level mem_mb/runtime in the
# .smk files are bookkeeping only until Phase 3), so its wall clock must
# cover the slowest chain, not just one rule: the existing Gen-2 run of the
# reference tile (240.291) took ~80 min end to end, ngmix alone ~54 min.
# 2h leaves real margin; revisit once more tiles/branches are added.
declare -A CPUS=( [prepare]=2  [compute]=8 )
declare -A MEM=(  [prepare]=8G [compute]=16G )
declare -A TIME=( [prepare]=00:15:00 [compute]=02:00:00 )
declare -A TARGET=( [prepare]="prepare_all_tiles" [compute]="" )

# sbatch, not srun: srun's behavior depends on whether the CALLING shell is
# already inside a SLURM allocation -- it then runs as a STEP of that job,
# constrained by whatever resources the outer job happens to have (bitten us
# twice: "job has expired" from a stale outer allocation, then "Job step's
# --cpus-per-task value exceeds that of job (8 > 2)" from a live but smaller
# one). sbatch always submits a brand-new, independent job regardless of the
# submitting shell's own SLURM_JOB_ID, so it can't nest. --wait blocks until
# the job finishes (same UX as srun); since sbatch writes output to a file
# rather than the terminal, this tails that file live and cleans up after.
run_phase() {
  local phase=$1
  echo "=== SP_PHASE=$phase (config: $RUN_CONFIG) ==="

  local batch_script logfile
  batch_script=$(mktemp "/tmp/im_sims_${phase}_XXXXXX.sh")
  logfile="${batch_script%.sh}.log"
  cat > "$batch_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export SP_PHASE=$phase
# rerun-triggers drops \`input\` (matches shapepipe's own profiles/nibi/config.yaml):
# required for clean_exposure's cascade-avoidance cut (tile_exp_split's
# ancient()-and-drop for a finished tile's reclaimed edges) to actually
# suppress reruns -- with \`input\` included, a since-cleaned exposure's
# vanished manifest reads as "input files updated" and rebuilds it anyway.
# Target(s) go BEFORE --configfile: --configfile is nargs='+' (so is
# --rerun-triggers) and greedily swallows any bare token after it -- a
# target name with nothing dash-prefixed in between gets silently eaten as
# a second config file ("FileNotFoundError: prepare_all_tiles"). Verified:
# swapping the two fixes it. --configfile is therefore placed LAST, with
# only EXTRA_ARGS (always dash-prefixed flags like -n) after it.
exec snakemake --cores ${CPUS[$phase]} --sdm apptainer \\
  --apptainer-args "$APPTAINER_ARGS" \\
  --rerun-triggers mtime params code software-env \\
  -s Snakefile --directory . \\
  ${TARGET[$phase]} \\
  --configfile "$RUN_CONFIG" \\
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
EOF
  chmod +x "$batch_script"

  sbatch --wait \
    --partition="$PARTITION" --account="$ACCOUNT" \
    --cpus-per-task="${CPUS[$phase]}" --mem="${MEM[$phase]}" \
    --time="${TIME[$phase]}" --exclude="$EXCLUDE" \
    -o "$logfile" -e "$logfile" \
    "$batch_script" &
  local sbatch_pid=$!

  # Stream the log once sbatch creates it, until the job (and sbatch --wait)
  # finishes.
  until [[ -s "$logfile" ]] || ! kill -0 "$sbatch_pid" 2>/dev/null; do sleep 0.5; done
  tail -f "$logfile" &
  local tail_pid=$!

  wait "$sbatch_pid"; local rc=$?
  kill "$tail_pid" 2>/dev/null || true
  rm -f "$batch_script" "$logfile"
  return $rc
}

case "$PHASE_ARG" in
  prepare) run_phase prepare ;;
  compute) run_phase compute ;;
  both)    run_phase prepare && run_phase compute ;;
esac
