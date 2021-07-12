#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

module load cuDNN/cuda10.0

# This loads the anaconda virtual environment with our packages
source /home/kmy220/.bashrc
conda activate

# Base directory for the experiment
cd /var/scratch/kmy220

# Simple trick to create a unique directory for each run of the script
echo $$
mkdir o`echo $$`
cd o`echo $$`

# Run the actual experiment 
python /var/scratch/kmy220/BachelorThesisKen/mpqe/run_model.py