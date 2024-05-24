#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=20
#SBATCH --time=1-00:00:00
#SBATCH --job-name=trail2_normal_2node_2gpu
#SBATCH --error=%J.err_
#SBATCH --output=%J.out_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prasannaprabhakar666@gmail.com

# Get the number of GPUs per node
gpus_per_node=$(srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:2 /usr/bin/env bash -c 'echo $CUDA_VISIBLE_DEVICES | wc -w')


# Launch one task per GPU
srun --nodes=2 --ntasks=$((2*gpus_per_node)) --ntasks-per-node=$gpus_per_node --gres=gpu:2 --cpus-per-task=1 python train_bert.py --checkpoint_dir ./experime$

