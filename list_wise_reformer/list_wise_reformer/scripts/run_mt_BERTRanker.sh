export CUDA_VISIBLE_DEVICES=5,6,7
# TASK=music
# TASK=books
# TASK=movies
TASK=redial
REPO_DIR=/ssd/home/gustavo/recsys2020penha

for MULTI_TASK_DATA in 'probe_type_recommendation-pop' 'probe_type_search'
do
  for SEED in 42 1 2 3 4
  do        
    python run_dialogue_baseline.py \
      --task $TASK \
      --data_folder $REPO_DIR/data/dialogue/ \
      --seed $SEED \
      --ranker bert \
      --output_dir $REPO_DIR/data/output_data/mt_bert4dialogue \
      --early_stopping_steps 100000 \
      --logging_steps 200000 \
      --learning_rate 5e-6 \
      --multi_task_for $MULTI_TASK_DATA
  done
done