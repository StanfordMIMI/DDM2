#!/bin/bash
#
#SBATCH --job-name=stage2
#
#SBATCH --time=780:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH -C GPU_MEM:11GB

#ml reset
#ml load py-pytorch/1.6.0_py36

# stage3 train model
# python3 train_diff_model.py -p train -c config/hardi_150.json
# python3 train.py -p train -c config/sb_1.json
# python3 train.py -p train -c config/ablation_1prior.json
# python3 train.py -p train -c config/ablation_1prior.json

python3 train_diff_model.py -p train -c config/qiyuan.json
