#!/bin/bash

#SBATCH --partition=hgnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00
#SBATCH --job-name=ARProteinPDB
#SBATCH --mail-user=jjung2@uvm.edu
#SBATCH --mail-type=ALL


cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
conda activate lignadmpnn_env

cd ${SLURM_SUBMIT_DIR}

python3 download.py