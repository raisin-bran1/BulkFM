#!/bin/bash
#SBATCH --job-name=archs4_preprocess
#SBATCH --account=ic_cdss170    
#SBATCH --partition=savio3       
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2          # Adjust based on your script's intensity
#SBATCH --time=01:00:00
#SBATCH --array=8                # This creates 10 tasks (0 to 9)
#SBATCH --output=~/archs4/logs/preprocess_%A_%a.out
#SBATCH --error=~/archs4/logs/preprocess_%A_%a.err

module load anaconda3
source activate nasa

# Use the array ID to pick the specific CSV file
INPUT_FILE="archs4_human_raw/input_chunk_$SLURM_ARRAY_TASK_ID.parquet"
DATA_DIR="/global/scratch/users/brianzhou/"
OUTPUT_DIR="/global/scratch/users/brianzhou/archs4_human/"

echo "Processing $INPUT_FILE on $(hostname)"

# Run your existing preprocessing script
python3 -u preprocessing_human.py --chunk_id $SLURM_ARRAY_TASK_ID --input $INPUT_FILE --datadir $DATA_DIR --outputdir $OUTPUT_DIR
