#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=END

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha
TASK=music

# Creating TASK dataset in U2U format (run this only the first time)
source /home/nfs/gustavopenha/env_slice_learning/bin/activate
srun python create_U2UIMN_data.py  \
    --task ${TASK} \
    --data_folder $REPO_DIR/data/dialogue/ \
    --u2u_folder $REPO_DIR/list_wise_reformer/list_wise_reformer/models/

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_2.7/bin/activate

cd ../models/U2U-IMN/

DATA_DIR=${parentdir}/data/$TASK

train_file=$DATA_DIR/train.txt
valid_file=$DATA_DIR/valid.txt
response_file=$DATA_DIR/responses.txt
vocab_file=$DATA_DIR/vocab.txt
char_vocab_file=$DATA_DIR/char_vocab.txt
embedded_vector_file=$DATA_DIR/glove_42B_300d_vec_plus_word2vec_100.txt

max_utter_len=50
max_utter_num=10
max_response_len=50
max_word_length=18
num_layer=1
embedding_dim=400
rnn_size=200

batch_size=102
lambda=0
dropout_keep_prob=0.8
num_epochs=1
evaluate_every=1

PKG_DIR=${parentdir}

x=0
for SEED in 42
do
srun python -u ${PKG_DIR}/model/train.py \
                --train_file $train_file \
                --valid_file $valid_file \
                --response_file $response_file \
                --vocab_file $vocab_file \
                --char_vocab_file $char_vocab_file \
                --embedded_vector_file $embedded_vector_file \
                --max_utter_len $max_utter_len \
                --max_utter_num $max_utter_num \
                --max_response_len $max_response_len \
                --max_word_length $max_word_length \
                --num_layer $num_layer \
                --embedding_dim $embedding_dim \
                --rnn_size $rnn_size \
                --batch_size $batch_size \
                --l2_reg_lambda $lambda \
                --dropout_keep_prob $dropout_keep_prob \
                --num_epochs $num_epochs \
                --evaluate_every $evaluate_every \
                --task $TASK \
                --output_predictions_folder $REPO_DIR/data/output_data/u2u/${x} \
                --seed $SEED
x=$((x+1))
done
