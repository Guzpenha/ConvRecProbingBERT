#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_2.7/bin/activate # I think this does not work if
                                                  # you are inside another env when calling sbatch

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha

# Creating ml25m dataset in SASRec format (run this only the first time)
#source /home/nfs/gustavopenha/env_slice_learning/bin/activate
#mkdir $REPO_DIR/list_wise_reformer/list_wise_reformer/models/SASRec/data/
#srun python create_sasrec_data.py  \
#    --task ml25m \
#    --data_folder $REPO_DIR/data/recommendation/ \
#    --sasrec_folder $REPO_DIR/list_wise_reformer/list_wise_reformer/models

# Running SASRec

cd ../models/SASRec/
MAX_LEN=200

# Finetunning hyperparameters using grid-search
x=4
for DROPOUT in 0.1 0.5 0.9
do
  for L2 in 0.1 0.001 0.0001
  do
    for HIDDEN_SIZE in 64 128 256
    do
      for NUM_EPOCHS in 300
      do
        python main.py --dataset=train_ml25m \
        --dropout_rate=$DROPOUT \
        --l2_emb=$L2 \
        --hidden_units=$HIDDEN_SIZE \
        --train_dir=default \
        --num_epochs=$NUM_EPOCHS \
        --eval_epochs=10000 \
        --maxlen=$MAX_LEN \
        --dataset_list_valid valid_ml25m.csv \
        --output_predictions_folder $REPO_DIR/data/output_data/sasrec/${x}
      done
    done
  done
done