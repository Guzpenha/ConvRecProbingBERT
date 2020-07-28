#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:5
#SBATCH --mem-per-cpu=20000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha

# TASK=music
# TASK=books
# TASK=movies
TASK=redial

for MULTI_TASK_DATA in 'probe_type_recommendation-pop' 'probe_type_search'
do
  for SEED in 42 1 2 3 4
  do        
    srun python run_dialogue_baseline.py \
      --task $TASK \
      --data_folder $REPO_DIR/data/dialogue/ \
      --seed $SEED \
      --ranker bert \
      --output_dir $REPO_DIR/data/output_data/mt_bert4dialogue \
      --early_stopping_steps 2000 \
      --logging_steps 200000 \
      --num_epochs 1 \
      --learning_rate 5e-6 \
      --multi_task_for $MULTI_TASK_DATA
  done
done