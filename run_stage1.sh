#!/bin/bash
#
#SBATCH --job-name=stage1_noisemodel
#
#SBATCH --time=400:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH -C GPU_MEM:11GB

#ml reset
#ml load py-pytorch/1.6.0_py36

# phase1 train noise model
# python3 train_noise_model.py -p train -c config/sb_2.json
# python3 train_noise_model.py -p train -c config/hardi_150.json
# python3 train_noise_model.py -p train -c config/gslider.json
# python3 train_noise_model.py -p train -c config/sb_1.json
# python3 train_noise_model.py -p train -c config/ablation_1prior.json
# python3 train_noise_model.py -p train -c config/ablation_7prior.json
# #SBATCH -c 4 #SBATCH --mem-per-cpu=4G #SBATCH --cpus-per-task=1

python3 train_noise_model.py -p train -c config/qiyuan.json

