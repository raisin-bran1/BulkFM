#!/bin/bash
#SBATCH --job-name=archs4_split
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio3
#SBATCH --array=0-119%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # Each task gets 16 cores for archs4py
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/global/home/users/brianzhou/logs/split_%A_%a.out
#SBATCH --error=/global/home/users/brianzhou/logs/split_%A_%a.err

module load anaconda3
source activate nasa

# $SLURM_ARRAY_TASK_ID will be 0 for the first job, 1 for the second, etc.
python3 -u split_data.py --chunk_id $SLURM_ARRAY_TASK_ID \
                      --datapath "/global/scratch/users/brianzhou/human_gene_v2.latest.h5" \
                      --outputdir "/global/scratch/users/brianzhou/archs4_human_raw/" \
                      --total_subset 1050137 # CHANGE between human (1050137) / mouse (1145967)
