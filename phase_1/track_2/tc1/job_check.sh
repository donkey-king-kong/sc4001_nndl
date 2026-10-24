#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=4
#SBATCH --time=30
#SBATCH --job-name=t2check
#SBATCH --output=logs/output_%x_%j.out
#SBATCH --error=logs/error_%x_%j.err

# Quick environment check (~5 min): GPU, Java, GPT-2, data files, Track 1 scoring, checkpoints.
# Submit from ~/track2/code with:  sbatch job_check.sh

module load anaconda
source activate track2

export HF_HUB_OFFLINE=1              # GPT-2 was downloaded by setup_tc1.sh
export JAVA_TOOL_OPTIONS=-Xlog:disable

cd ~/track2/code
python -u train_baseline.py --check
