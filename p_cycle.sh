#!/bin/bash
#SBATCH --job-name=P_cycle
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=15
#SBATCH --gres=gpu:0

source /home/hua089/newmont/py39/bin/activate
python main_code_parallel.py