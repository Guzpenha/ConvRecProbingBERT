#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=14000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha

# TASK=music
TASK=books
# TASK=movies


# for MULTI_TASK_DATA in 'probe_type_recommendation-pop' 'probe_type_search'
for MULTI_TASK_DATA in 'probe_type_recommendation-pop'
do
  for SEED in 42
  do        
    srun python run_dialogue_baseline.py \
      --task $TASK \
      --data_folder $REPO_DIR/data/dialogue/ \
      --seed $SEED \
      --ranker bert \
      --output_dir $REPO_DIR/data/output_data/mt_bert4dialogue \
      --early_stopping_steps 200000 \
      --logging_steps 1000 \
      --learning_rate 5e-6 \
      --multi_task_for $MULTI_TASK_DATA
  done
done