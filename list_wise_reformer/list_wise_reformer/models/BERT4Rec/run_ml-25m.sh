CKPT_DIR="/Users/gustavopenha/personal/recsys20/list_wise_reformer/list_wise_reformer/models/BERT4Rec"
dataset_name="train_ml25m"
max_seq_length=200
masked_lm_prob=0.2
max_predictions_per_seq=20

dim=64
batch_size=256
#num_train_steps=400000
num_train_steps=10

prop_sliding_window=0.5
mask_prob=1.0
dupe_factor=1
pool_size=10

signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}"


#python -u gen_data_fin.py \
#    --dataset_name=${dataset_name} \
#    --max_seq_length=${max_seq_length} \
#    --max_predictions_per_seq=${max_predictions_per_seq} \
#    --mask_prob=${mask_prob} \
#    --dupe_factor=${dupe_factor} \
#    --masked_lm_prob=${masked_lm_prob} \
#    --prop_sliding_window=${prop_sliding_window} \
#    --signature=${signature} \
#    --pool_size=${pool_size} \


CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
    --test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
    --vocab_filename=./data/${dataset_name}${signature}.vocab \
    --user_history_filename=./data/${dataset_name}${signature}.his \
    --checkpointDir=${CKPT_DIR}/${dataset_name} \
    --signature=${signature}-${dim} \
    --do_train=True \
    --do_eval=True \
    --dataset_list_valid=./data/valid_ml25m.csv \
    --output_predictions_folder /Users/gustavopenha/personal/recsys20/data/output_data/bert4rec/1 \
    --bert_config_file=./bert_train/bert_config_${dataset_name}_${dim}.json \
    --batch_size=${batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
    --learning_rate=1e-4

