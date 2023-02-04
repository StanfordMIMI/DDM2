#!/bin/bash
#
#SBATCH --job-name=get_alpha
#
#SBATCH --time=300:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8

#ml reset
#ml load py-pytorch/1.6.0_py36

# python3 get_alpha.py -p train -c config/hardi_150.json
python3 match_state.py -p train -c config/qiyuan.json
# python3 get_alpha.py -p train -c config/ablation_7prior.json