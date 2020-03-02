#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

srun python make_conv_data.py \
  --conversations_path ./dialogue/dialogues.csv \
  --output_path ./dialogue