#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=14000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha

#Either uncomment the following, or call this using run_all_dialogue_baselines.sh
#TASK=music
#TASK=books
#TASK=movies

for SEED in 42 1 2 3 4
do
  srun python run_dialogue_baseline.py \
    --task $TASK \
    --data_folder $REPO_DIR/data/dialogue/ \
    --seed $SEED \
    --ranker bert \
    --output_dir $REPO_DIR/data/output_data/bert4dialogue \
    --early_stopping_steps 2500 \
    --logging_steps 3000
done
