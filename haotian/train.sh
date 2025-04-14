#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=olmo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=./logs_1b/olmo_test1.out
# module purge
# module load 2023




source activate olmo
conda init
conda activate olmo
export NCCL_P2P_DISABLE=1
torchrun \
  --nproc_per_node=4 \
  scripts/train.py \
  /home/thuang/projects/OLMo/configs/official-1124/OLMo2-7B-stage1.yaml \
  --save_overwrite \
