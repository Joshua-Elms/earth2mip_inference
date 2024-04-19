#!/bin/bash

#SBATCH -J fcnv2_sm_inference
#SBATCH -p debug
#SBATCH -o output.out
#SBATCH -e log.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH -A r00389

source activate ~/envs/earth2mip

echo "Beginning inference script"

python run_inference.py

echo "Inference script complete"