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


python sample.py -c config/hardi_150.json