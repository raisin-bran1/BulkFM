#!/bin/bash
# Binformer Single-Node Multi-GPU Training (4 GPUs)
#
#SBATCH --job-name=binformer_4gpu
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --exclude=n0227.savio2,n0302.savio2
#SBATCH --nodes=1                
#SBATCH --ntasks-per-node=1      # One torchrun process per node
#SBATCH --gres=gpu:4             # Request all 4 GPUs on the node
#SBATCH --cpus-per-task=8        # 2 CPUs per GPU x 4 GPUs
#SBATCH --mem=60G                
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

set -euo pipefail

# Force unbuffered python output for real-time tailing
export PYTHONUNBUFFERED=1

# 1. Environment Setup
module purge
module load anaconda3
export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
conda activate nasa

echo "--- Single-Node 4-GPU Training Setup ---"
echo "Job ID:         $SLURM_JOB_ID"
echo "GPUs Requested: 4"
echo "Excluded Nodes: n0227, n0302"
echo "---------------------------------------"

# 2. Execution using torchrun
# --standalone is perfect for single-node runs
# --nproc_per_node=4 launches 4 processes (one per GPU)
torchrun \
    --standalone \
    --nproc_per_node=4 \
    training/train.py

echo "Job ${SLURM_JOB_ID} complete."
