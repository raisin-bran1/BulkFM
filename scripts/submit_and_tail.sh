#!/usr/bin/env bash
# Binformer Submission Script
#
# Usage:
#   cd ~/binformer
#   bash scripts/submit_and_tail.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SLURM_SCRIPT="scripts/savio_multi_gpu.slurm"

# Submit or attach to existing job
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  JOB_ID="$1"
  echo "Watching existing job ${JOB_ID}"
else
  # Submit job from the project root
  out="$(sbatch --parsable --export=ALL "${SLURM_SCRIPT}")"
  JOB_ID="${out}"
  echo "Submitted job ${JOB_ID}"
fi

LOG_FILE="logs/slurm-${JOB_ID}.out"
echo "Waiting for job ${JOB_ID} to start..."

while [[ ! -f "${LOG_FILE}" ]]; do
  state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || echo "FINISHED")
  if [[ "$state" == "FINISHED" ]]; then
    echo "Job ${JOB_ID} finished or failed immediately."
    exit 1
  fi
  sleep 5
done

echo "Tailing logs (Ctrl+C to stop watching):"
tail -f "${LOG_FILE}"
