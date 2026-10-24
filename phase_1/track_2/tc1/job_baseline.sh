#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=4
#SBATCH --time=360
#SBATCH --job-name=t2base
#SBATCH --output=logs/output_%x_%j.out
#SBATCH --error=logs/error_%x_%j.err

# Trains/resumes the 3 baseline seeds one after another (seed 0, then 1, then 2).
# The job limit is 6 h, so the script saves a checkpoint and stops at ~5 h 40 min,
# then this job submits itself again to continue. Submit once from ~/track2/code with:
#   sbatch job_baseline.sh

module load anaconda
source activate track2

export HF_HUB_OFFLINE=1              # GPT-2 was downloaded by setup_tc1.sh
export JAVA_TOOL_OPTIONS=-Xlog:disable
export TQDM_MININTERVAL=60           # progress bar once a minute, keeps the log short

cd ~/track2/code
python -u train_baseline.py --time-limit-min 340
status=$?

if [ $status -eq 3 ]; then
    echo "Time limit reached, checkpoint saved. Submitting the next job..."
    sbatch job_baseline.sh || echo "Could not resubmit automatically. Run: cd ~/track2/code && sbatch job_baseline.sh"
elif [ $status -eq 0 ]; then
    echo "All done. Results are in ~/track2/out/results"
else
    echo "Stopped with an error (exit code $status). Check logs/error_t2base_${SLURM_JOB_ID}.err"
fi
