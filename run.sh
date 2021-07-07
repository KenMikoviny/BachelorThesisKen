#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=<partition name,  see below>
#SBATCH --gres=gpu:1

## in the list above, the partition name depends on where you are running your job. 
## On DAS5 the default would be `defq` on Lisa the default would be `gpu` or `gpu_shared`
## Typing `sinfo` on the server command line gives a column called PARTITION.  
## There, one can find the name of a specific node, the state (down, alloc, idle etc), the availability and how long is the time limit . Ask your supervisor before running jobs on queues you do not know.


# Load GPU drivers
## These are for DAS5


module load cuDNN/cuda10.0
## For Lisa modules are usually not needed, so remove the previous 2 lines. https://userinfo.surfsara.nl/systems/shared/modules 


# This loads the anaconda virtual environment with our packages
source /home/kmy220/.bashrc
conda activate

# Base directory for the experiment
cd /home/kmy220/experiments/sparse/convolution

# Simple trick to create a unique directory for each run of the script
echo $$
mkdir o`echo $$`
cd o`echo $$`

# Run the actual experiment 
python -u /home/pbloem/git/sparse-hyper/experiments/convolution.py -e 100 -l 0.0001 -c -b 32 -T ../runs/baseline --augmentation --schedule warmup --lr-warmup 1500000 -D ../data -m david --weight-decay 0.0005 --optimizer adamw --gradient-clipping 1.0 --mul 1.0 --test-every 1


wait      	# (only required for parallel bash scripts)