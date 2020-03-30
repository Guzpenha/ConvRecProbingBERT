#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=END

MAX_LEN=200
BATCH_SIZE=128
REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha
#TASK=ml25m

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_2.7/bin/activate # I think this does not work if
                                                  # you are inside another env when calling sbatch


# Creating TASK dataset in SASRec format (run this only the first time)
#source /home/nfs/gustavopenha/env_slice_learning/bin/activate
#mkdir $REPO_DIR/list_wise_reformer/list_wise_reformer/models/SASRec/data/
#srun python create_sasrec_data.py  \
#    --task ${TASK} \
#    --data_folder $REPO_DIR/data/recommendation/ \
#    --sasrec_folder $REPO_DIR/list_wise_reformer/list_wise_reformer/models

#This takes more than 72h with 1 GPU:
# Finetunning hyperparameters using grid-search
#cd ../models/SASRec/
#x=16
#for DROPOUT in 0.5 0.9
#do
#  for HIDDEN_SIZE in 64 128 256
#  do
#    for NUM_EPOCHS in 300
#    do
#      x=$((x+1))
#      python main.py --dataset=train_${TASK} \
#      --dropout_rate=$DROPOUT \
#      --l2_emb=0.0001 \
#      --hidden_units=$HIDDEN_SIZE \
#      --train_dir=default \
#      --num_epochs=$NUM_EPOCHS \
#      --eval_epochs=10000 \
#      --maxlen=$MAX_LEN \
#      --batch_size $BATCH_SIZE\
#      --dataset_list_valid valid_${TASK}.csv \
#      --output_predictions_folder $REPO_DIR/data/output_data/sasrec/${x}
#    done
#  done
#done

cd ../models/SASRec/
#x=22
for SEED in 42 1 2 3 4
do
x=$((x+1))
python main.py --dataset=train_${TASK} \
  --seed $SEED \
  --dropout_rate=0.1 \
  --l2_emb=0.0001 \
  --hidden_units=128 \
  --train_dir=default \
  --num_epochs=300 \
  --eval_epochs=10000 \
  --maxlen=$MAX_LEN \
  --batch_size $BATCH_SIZE\
  --dataset_list_valid valid_${TASK}.csv \
  --output_predictions_folder $REPO_DIR/data/output_data/sasrec/${x}
done
