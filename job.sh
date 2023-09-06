#!/bin/bash


## SLURM FEATURES

#SBATCH --partition=gpu
#SBATCH --job-name=question
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ary24/face2painting/logs/output.out
#SBATCH --error=/scratch/ary24/face2painting/logs/error.out


## Commands on cluster
source ~/.bashrc
cd /scratch/ary24/face2painting
conda activate paintings
module load cuda/11.7.1 cudnn/8.1.3-jlb638
python3 train.py
