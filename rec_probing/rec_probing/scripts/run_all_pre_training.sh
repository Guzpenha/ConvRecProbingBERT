export CUDA_VISIBLE_DEVICES=4,5
source /ssd/home/gustavo/recsys2020penha/env/bin/activate

REPO_DIR=/ssd/home/gustavo/recsys2020penha
NUMBER_PROBE_QUERIES=100000

# for PROBE_TYPE in 'recommendation-pop'
# do
#     for TASK in 'ml25m' 'gr' 'music'
#     do
#         python pre_train_BERT.py \
#             --task $TASK \
#             --probe_type ${PROBE_TYPE} \
#             --input_folder $REPO_DIR/data/recommendation/ \
#             --output_folder $REPO_DIR/data/output_data/probes/ \
#             --number_queries $NUMBER_PROBE_QUERIES \
#             --number_candidates 1 \
#             --batch_size 32 \
#             --num_epochs 5 \
#             --bert_model "bert-base-cased"
#     done
# done

# for PROBE_TYPE in  'search-inv'
# do
#     for TASK in 'ml25m' 'gr' 'music'
#     do
#         python pre_train_BERT.py \
#             --task $TASK \
#             --probe_type ${PROBE_TYPE} \
#             --input_folder $REPO_DIR/data/search/ \
#             --output_folder $REPO_DIR/data/output_data/probes/ \
#             --number_queries $NUMBER_PROBE_QUERIES \
#             --number_candidates 1 \
#             --batch_size 32 \
#             --num_epochs 5 \
#             --bert_model "bert-base-cased"
#     done
# done


for PROBE_TYPE in 'mlm'
do
    for TASK in 'ml25m' 'gr' 'music'
    do
        python pre_train_BERT.py \
            --task $TASK \
            --probe_type ${PROBE_TYPE} \
            --input_folder $REPO_DIR/data/recommendation/ \
            --output_folder $REPO_DIR/data/output_data/probes/ \
            --number_queries $NUMBER_PROBE_QUERIES \
            --number_candidates 1 \
            --batch_size 32 \
            --num_epochs 5 \
            --bert_model "bert-base-cased"
    done
done