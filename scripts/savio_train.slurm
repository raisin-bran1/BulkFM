#!/bin/bash
# Binformer Single-GPU Sweep Agent
#
#SBATCH --job-name=binformer_trial
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --exclude=n0227.savio2,n0228.savio2
#SBATCH --nodes=1                
#SBATCH --ntasks=1               # 1 agent per task
#SBATCH --gres=gpu:1             # 1 GPU per task
#SBATCH --cpus-per-task=2        # 2 CPUs per task
#SBATCH --mem=15G                # Lower memory request (more likely to be scheduled)
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

set -euo pipefail

# 1. Environment Setup
module purge
module load anaconda3
export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
conda activate nasa

# 2. Execution
# Each Slurm job starts ONE W&B agent which runs ONE 1-GPU trial.
if [[ -n "${WANDB_SWEEP_ID:-}" ]]; then
  echo "Starting W&B Sweep Agent for trial: ${WANDB_SWEEP_ID}"
  python -m wandb agent --count 1 "${WANDB_SWEEP_ID}"
else
  # Fallback: Just run a single-GPU training run directly
  echo "Starting Single-GPU Binformer Training..."
  torchrun --standalone --nproc_per_node=1 train.py
fi

echo "Job ${SLURM_JOB_ID} complete."
