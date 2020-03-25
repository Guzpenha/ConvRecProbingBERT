#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

TASK=ml25m

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha

## Creating ml25m dataset in BERT4Rec format (run this only the first time)
#source /home/nfs/gustavopenha/env_slice_learning/bin/activate
#mkdir $REPO_DIR/list_wise_reformer/list_wise_reformer/models/BERT4Rec/data/
#srun python create_BERT4rec_data.py  \
#    --task ml25m \
#    --data_folder $REPO_DIR/data/recommendation/ \
#    --bert4rec_folder $REPO_DIR/list_wise_reformer/list_wise_reformer/models


## Running BERT4Rec
source /home/nfs/gustavopenha/env_2.7/bin/activate # I think this does not work if you are inside another env when calling sbatch
cd ../models/BERT4Rec/
CKPT_DIR=${REPO_DIR}/list_wise_reformer/list_wise_reformer/models/BERT4Rec
dataset_name="train_${TASK}"
max_seq_length=200
masked_lm_prob=0.2
max_predictions_per_seq=20

dim=64
batch_size=256
#num_train_steps=400000
num_train_steps=50

prop_sliding_window=0.5
mask_prob=1.0
dupe_factor=1
pool_size=5

signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}"

srun python -u gen_data_fin.py \
    --dataset_name=${dataset_name} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --mask_prob=${mask_prob} \
    --dupe_factor=${dupe_factor} \
    --masked_lm_prob=${masked_lm_prob} \
    --prop_sliding_window=${prop_sliding_window} \
    --signature=${signature} \
    --pool_size=${pool_size} \


#Finetunning
x=3
srun python -u run.py \
    --train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
    --test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
    --vocab_filename=./data/${dataset_name}${signature}.vocab \
    --user_history_filename=./data/${dataset_name}${signature}.his \
    --checkpointDir=${CKPT_DIR}/${dataset_name} \
    --signature=${signature}-${dim} \
    --do_train=True \
    --do_eval=True \
    --dataset_list_valid=./data/valid_${TASK}.csv \
    --output_predictions_folder $REPO_DIR/data/output_data/bert4rec/${x} \
    --bert_config_file=./bert_train/bert_config_${dataset_name}_${dim}.json \
    --batch_size=${batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
    --learning_rate=1e-4
x=$((x+1))
