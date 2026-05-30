#!/bin/bash
# Usage: ./wandb_sweep.sh <WANDB_SWEEP_ID>

if [ -z "$1" ]; then
  echo "Usage: ./wandb_sweep.sh <WANDB_SWEEP_ID>"
  exit 1
fi

SWEEP_ID=$1

for i in {1..2}; do
  echo "Submitting trial $i/10 for sweep $SWEEP_ID..."
  sbatch --export=WANDB_SWEEP_ID=$SWEEP_ID scripts/savio_train.slurm
done
