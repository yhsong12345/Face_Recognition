#!/bin/bash
#SBATCH --job-name="Face Recognition"
#SBATCH --nodelist=hpe159,hpe160,nv170,nv172,nv174,nv176,nv178,nv180
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="train model for reid"
#SBATCH --nodes=1
#SBATCH --gres=gpu:4


################ Number of total process ##########################

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "



echo "Run started at:- "
date

##### if using huggingface model, set for location to store huggingface model
export HF_HOME=./cache/huggingface


# Kagglehub cache directory
export KAGGLEHUB_CACHE=./cache/kagglehub


python train.py --data /purestorage/AILAB/AI_4/datasets/cctv/image/stage1_data/2025-10-15 \
    --model SE_LResnet101 --name arcface

echo Done