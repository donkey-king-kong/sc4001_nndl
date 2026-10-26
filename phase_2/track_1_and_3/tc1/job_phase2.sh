#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=4
#SBATCH --time=360
#SBATCH --job-name=phase2
#SBATCH --output=logs/output_%x_%j.out
#SBATCH --error=logs/error_%x_%j.err

# Phase 2 (B and C) on TC1: runs run_phase2.py stages and resubmits itself at the 6 h job limit.
# Reuses the Track 2 setup (phase_1/track_2/tc1/setup_tc1.sh): the "track2" conda env, GPT-2 and NLTK
# data, and ~/track2/CLIP_prefix_caption with its AdamW import already patched.
#
# Layout on TC1 (change the paths below if yours differ):
#   ~/phase2/code/      run_phase2.py  object_swap.py  track_2.py  track1_eval.py  job_phase2.sh
#   ~/phase2/track_1/   data/karpathy_index.json
#                       eval/chair_gt_train.pkl  chair_gt_val.pkl  coco_objects.json  synonyms.txt
#                       features/coco_rn50x4_train.pkl  coco_rn50x4_val.pkl   (or symlinks to ~/track2/data/)
#                       models/clipcap_transformer_coco_pretrained.pt         (only the check stage uses it)
#   ~/phase2/baseline/  baseline_summary.json  clipcap_baseline_seed{0,1,2}_val.per_image.json
#   ~/phase2/cooccurrence_train.pkl   (optional: phase_1/track_3/eval/; rebuilt if missing)
#   ~/phase2/out/       everything this job writes (runs/, results/, models/, cache/)
#
# Submit from ~/phase2/code (mkdir -p logs first, or SLURM cannot write the log files):
#   sbatch --time=30 job_phase2.sh check     # environment + loss smoke test, ~10 min
#   sbatch job_phase2.sh prepare             # negatives caches, spot check, review sheet (no checkpointing:
#                                            #   must finish inside one job; just resubmit if it is cut off)
#   sbatch job_phase2.sh all                 # sweep -> select -> ablation -> summary, resubmits itself
#   sbatch job_phase2.sh unlikelihood        # optional, after select
#   sbatch job_phase2.sh <stage> [extra run_phase2.py args]   # any single stage
#
# Splitting across GPUs (e.g. each teammate on their own TC1 account, same ~/phase2 layout):
#   sbatch job_phase2.sh sweep --worker 0 --n-workers 3     # teammates use --worker 1 and 2
# Then copy every worker's out/runs/ and out/results/ into one out/ folder and run select / summary there.
#
# Exit codes from run_phase2.py: 0 = done, 3 = time limit (this job resubmits itself), other = error.

PHASE2=~/phase2
CODE_DIR=$PHASE2/code
TRACK1_DIR=$PHASE2/track_1
BASELINE_DIR=$PHASE2/baseline
COOCCURRENCE=$PHASE2/cooccurrence_train.pkl
CLIP_ROOT=~/track2/CLIP_prefix_caption
OUT_DIR=$PHASE2/out
JOB_LIMIT_MIN=340                    # job time is 360 min; stop and save ~20 min early

if [ $# -lt 1 ]; then
    echo "usage: sbatch job_phase2.sh <check|prepare|sweep|select|ablation|unlikelihood|summary|all> [args]"
    exit 2
fi
STAGE=$1
shift
EXTRA=("$@")

if [ "$STAGE" = "all" ]; then
    STAGES=(sweep select ablation summary)
else
    STAGES=("$STAGE")
fi

module load anaconda
source activate track2

export HF_HUB_OFFLINE=1              # GPT-2 was downloaded by setup_tc1.sh
export JAVA_TOOL_OPTIONS=-Xlog:disable
export TQDM_MININTERVAL=60           # progress bar once a minute, keeps the log short

cd "$CODE_DIR" || exit 1
mkdir -p logs "$OUT_DIR"

resubmit() {
    echo "Time limit reached, checkpoint saved. Submitting the next job..."
    sbatch job_phase2.sh "$STAGE" "${EXTRA[@]}" \
        || echo "Could not resubmit automatically. Run: cd $CODE_DIR && sbatch job_phase2.sh $STAGE ${EXTRA[*]}"
}

for s in "${STAGES[@]}"; do
    left=$(( JOB_LIMIT_MIN - SECONDS / 60 ))
    echo "=== stage $s ($left min left in this job) ==="
    python -u run_phase2.py --stage "$s" \
        --track1-dir "$TRACK1_DIR" --baseline-results "$BASELINE_DIR" --cooccurrence "$COOCCURRENCE" \
        --clip-root "$CLIP_ROOT" --out "$OUT_DIR" --time-limit-min "$left" "${EXTRA[@]}"
    status=$?

    if [ $status -eq 3 ]; then
        resubmit
        exit 0
    elif [ $status -ne 0 ]; then
        if [ "$s" = "select" ]; then
            echo "select did not choose a lambda (sweep not finished, or no lambda kept CIDEr above the floor)."
            echo "See the table above; then pass --best-lambda or wait for the other workers, and resubmit."
        else
            echo "Stage $s stopped with an error (exit code $status). Check logs/error_phase2_${SLURM_JOB_ID}.err"
        fi
        exit $status
    fi
done

echo "Done: ${STAGES[*]}. Results are in $OUT_DIR/results (run sheet: runs.csv)"