#!/bin/bash
#SBATCH -J epochs               # Job name
#SBATCH -o ./%j.out          # Name of stdout output file (%j expands to jobId)
#SBATCH -p batch             # Name of partition
#SBATCH -w niolon14
#SBATCH -N 1                 # Total number of nodes requested
#SBATCH -t 23:00:00          # Run time (hh:mm:ss) - 1.5 hours# Launch
#SBATCH --ntasks-per-node=16 # Number of tasks per node (32 CPUs)
#SBATCH --mem=75G            # Amount of memory requested
module purge
module load all
module load anaconda/3
# moving to the working directory
source activate gbpy38
export PYTHONPATH=/envau/work/comco/bardanikas.g/Vision4Action/vision4action/analysis_scripts:$PYTHONPATH
cd /envau/work/comco/bardanikas.g/Vision4Action/vision4action/analysis_scripts/preprocessing/
python generate_epochs.py

