#!/bin/bash
#SBATCH --job-name=archs4_preprocess
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=~/archs4/logs/preprocess_%A_%a.out
#SBATCH --error=~/archs4/logs/preprocess_%A_%a.err

module load anaconda3
source activate nasa

DATA_DIR="/global/scratch/users/brianzhou/"
OUTPUT_DIR="/global/scratch/users/brianzhou/archs4_human/"

echo "Processing chunk $SLURM_ARRAY_TASK_ID on $(hostname)"

# split_data.py downloads, filters samples, and calls preprocessing_human.py
python3 -u split_data.py --chunk_id $SLURM_ARRAY_TASK_ID --datadir $DATA_DIR --outputdir $OUTPUT_DIR
