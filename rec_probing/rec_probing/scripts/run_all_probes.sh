export CUDA_VISIBLE_DEVICES=4,5,6,7
source /ssd/home/gustavo/recsys2020penha/env/bin/activate

REPO_DIR=/ssd/home/gustavo/recsys2020penha
NUMBER_PROBE_QUERIES=100000

for PROBE_TYPE in 'recommendation' 'search'
do
    for TASK in 'ml25m' 'gr' 'music'
    do
        python run_probes.py \
            --task $TASK \
            --probe_type ${PROBE_TYPE} \
            --input_folder $REPO_DIR/data/${PROBE_TYPE}/ \
            --output_folder $REPO_DIR/data/output_data/probes/ \
            --number_queries $NUMBER_PROBE_QUERIES \
            --number_candidates 5 \
            --batch_size 64 \
            --bert_model 'bert-base-cased'
    done
done


for PROBE_TYPE in 'recommendation' 'search'
do
    for TASK in 'ml25m' 'gr' 'music'
    do
        python run_probes.py \
            --task $TASK \
            --probe_type ${PROBE_TYPE} \
            --input_folder $REPO_DIR/data/${PROBE_TYPE}/ \
            --output_folder $REPO_DIR/data/output_data/probes/ \
            --number_queries $NUMBER_PROBE_QUERIES \
            --number_candidates 5 \
            --batch_size 32 \
            --bert_model 'bert-large-cased'
    done
done