#!/bin/bash

#SBATCH --job-name=rnn   # Job name
#SBATCH --partition=standard # Partition name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --nodelist=mastiff           # Node name
#SBATCH --cpus-per-task=3            # Number of CPU cores per task
#SBATCH --output=slurm_log/rmsrun_%j.log  # Standard output and error log
#SBATCH --gres=gpu:1                 # Request GPU resource

source /data/cl/user/bailinw/miniconda3/etc/profile.d/conda.sh
conda activate seq-test
python -m zoology.launch zoology/experiments/mqar/rms.py
# python -m zoology.launch zoology/experiments/mqar/gla.py
# python -m zoology.launch zoology/experiments/regbench/rms.py
