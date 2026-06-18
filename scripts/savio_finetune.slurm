#!/bin/bash
# Binformer OSDR Fine-tuning
#
#SBATCH --job-name=osdr_finetune
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --nodes=1                
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.out

set -euo pipefail

# 1. Environment Setup
module purge
module load anaconda3
export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
conda activate nasa

# Create logs directory if it doesn't exist
mkdir -p logs

# 2. Execution
echo "Starting OSDR Fine-tuning..."
python downstream/osdr/finetune_osdr.py

echo "Job ${SLURM_JOB_ID} complete."
