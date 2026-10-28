#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=4
#SBATCH --time=360
#SBATCH --job-name=phase2
#SBATCH --output=/tc1home/UG/limj0279/sc4001_nndl/phase_2/track_1_and_3/logs/output_%x_%j.out
#SBATCH --error=/tc1home/UG/limj0279/sc4001_nndl/phase_2/track_1_and_3/logs/error_%x_%j.err

# Phase 2 (B and C). Slurm may execute a submitted script from its spool
# directory, so paths must not be derived from BASH_SOURCE.
# The job stops before the Slurm limit and resubmits itself when a stage
# returns exit code 3.

REPO_ROOT="/tc1home/UG/limj0279/sc4001_nndl"
CODE_DIR="$REPO_ROOT/phase_2/track_1_and_3"
TRACK1_DIR="$REPO_ROOT/phase_1/track_1"
BASELINE_DIR="$REPO_ROOT/phase_1/track_2/results"
COOCCURRENCE="$REPO_ROOT/phase_1/track_3/eval/cooccurrence_train.pkl"
CLIP_ROOT="$REPO_ROOT/CLIP_prefix_caption"
OUT_DIR="$CODE_DIR/out"
JOB_LIMIT_MIN=340

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch job_phase2_updated.sh <check|prepare|sweep|select|ablation|unlikelihood|summary|all> [args]" >&2
    exit 2
fi

STAGE="$1"
shift
EXTRA=("$@")

if [ "$STAGE" = "all" ]; then
    STAGES=(sweep select ablation summary)
else
    STAGES=("$STAGE")
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    exit 1
fi

if [ ! -f "$REPO_ROOT/pyproject.toml" ] || [ ! -f "$REPO_ROOT/uv.lock" ]; then
    echo "UV project files are missing from $REPO_ROOT." >&2
    exit 1
fi

if [ ! -f "$CODE_DIR/run_phase2.py" ]; then
    echo "Phase 2 code is missing from $CODE_DIR." >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export JAVA_TOOL_OPTIONS=-Xlog:disable
export TQDM_MININTERVAL=60

cd "$CODE_DIR" || exit 1
mkdir -p logs "$OUT_DIR"

resubmit() {
    echo "Time limit reached, checkpoint saved. Submitting the next job..."
    sbatch "$CODE_DIR/job_phase2_updated.sh" "$STAGE" "${EXTRA[@]}" \
        || echo "Could not resubmit automatically. Run: sbatch $CODE_DIR/job_phase2_updated.sh $STAGE ${EXTRA[*]}"
}

for s in "${STAGES[@]}"; do
    left=$((JOB_LIMIT_MIN - SECONDS / 60))
    if [ "$left" -le 0 ]; then
        echo "No time remains for stage $s; resubmitting." >&2
        resubmit
        exit 0
    fi

    echo "=== stage $s ($left min left in this job) ==="
    uv run --locked --project "$REPO_ROOT" python -u "$CODE_DIR/run_phase2.py" \
        --stage "$s" \
        --track1-dir "$TRACK1_DIR" \
        --baseline-results "$BASELINE_DIR" \
        --cooccurrence "$COOCCURRENCE" \
        --clip-root "$CLIP_ROOT" \
        --out "$OUT_DIR" \
        --time-limit-min "$left" \
        "${EXTRA[@]}"
    status=$?

    if [ "$status" -eq 3 ]; then
        resubmit
        exit 0
    elif [ "$status" -ne 0 ]; then
        if [ "$s" = "select" ]; then
            echo "select did not choose a lambda (sweep not finished, or no lambda kept CIDEr above the floor)." >&2
            echo "See the table above; pass --best-lambda or wait for the other workers, then resubmit." >&2
        else
            echo "Stage $s stopped with error code $status. Check logs/error_phase2_${SLURM_JOB_ID}.err" >&2
        fi
        exit "$status"
    fi
done

echo "Done: ${STAGES[*]}. Results are in $OUT_DIR/results (run sheet: runs.csv)"
