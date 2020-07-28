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
# PRE_TRAIN_TASK=music

TASK=books
PRE_TRAIN_TASK=gr

# TASK=movies
# PRE_TRAIN_TASK=ml25m

# for PROBE_TYPE in mlm recommendation-pop search-inv
for PROBE_TYPE in recommendation-pop
do
  for SEED in 42
  do
    PRE_TRAINED_MODEL=pre_trained_on_probe_type_${PROBE_TYPE}_task_${PRE_TRAIN_TASK}_num_candidates_1_num_queries_100004_model_bert-base-cased
    
    srun python run_dialogue_baseline.py \
      --task $TASK \
      --data_folder $REPO_DIR/data/dialogue/ \
      --seed $SEED \
      --ranker bert \
      --output_dir $REPO_DIR/data/output_data/infused_bert4dialogue \
      --early_stopping_steps 100000 \
      --logging_steps 200000 \
      --learning_rate 5e-6 \
      --bert_model $REPO_DIR/data/output_data/probes/${PRE_TRAINED_MODEL}
  done
done