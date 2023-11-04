#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=train-rna
#SBATCH --output=train.out
#SBATCH --mail-type=END
#SBATCH --mail-user=cs5526@nyu.edu

module purge
module load python/intel/3.8.6
module load cuda/11.6.2

python ./train.py train_data.csv --d_model 512 --nhead 8 --num_layers 8 --num_epochs 10
