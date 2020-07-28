export CUDA_VISIBLE_DEVICES=6,7
source /ssd/home/gustavo/recsys2020penha/env/bin/activate

REPO_DIR=/ssd/home/gustavo/recsys2020penha
NUMBER_PROBE_QUERIES=100000

# for SENTENCE_TYPE in 'no-item' 'type-I' 'type-II'
# do
#     for TASK in 'ml25m' 'gr' 'music'
#     do
#         python run_mlm_probe.py \
#             --task $TASK \
#             --input_folder $REPO_DIR/data/recommendation/ \
#             --output_folder $REPO_DIR/data/output_data/probes/ \
#             --number_queries $NUMBER_PROBE_QUERIES \
#             --batch_size 32 \
#             --sentence_type ${SENTENCE_TYPE} \
#             --bert_model 'bert-base-cased'    
              
#     done

#     for TASK in 'ml25m' 'gr' 'music'
#     do
#         python run_mlm_probe.py \
#             --task $TASK \
#             --input_folder $REPO_DIR/data/recommendation/ \
#             --output_folder $REPO_DIR/data/output_data/probes/ \
#             --number_queries $NUMBER_PROBE_QUERIES \
#             --batch_size 32 \
#             --sentence_type ${SENTENCE_TYPE} \
#             --bert_model 'bert-large-cased'
#     done
# done

for SENTENCE_TYPE in 'no-item' 'type-I' 'type-II'
do
    for TASK in 'ml25m' 'gr' 'music'
    do
        python run_mlm_probe.py \
            --task $TASK \
            --input_folder $REPO_DIR/data/recommendation/ \
            --output_folder $REPO_DIR/data/output_data/probes/ \
            --number_queries $NUMBER_PROBE_QUERIES \
            --batch_size 32 \
            --sentence_type ${SENTENCE_TYPE} \
            --bert_model 'roberta-large'
    done
done

# for PROBE_TECHNIQUE in 'mean-sim' 'cls-sim' 'nsp'
# do
    # for PROBE_TYPE in 'recommendation' 'search'
    # do
    #     for TASK in 'ml25m' 'gr' 'music'
    #     do
    #         python run_probes.py \
    #             --task $TASK \
    #             --probe_type ${PROBE_TYPE} \
    #             --input_folder $REPO_DIR/data/${PROBE_TYPE}/ \
    #             --output_folder $REPO_DIR/data/output_data/probes/ \
    #             --number_queries $NUMBER_PROBE_QUERIES \
    #             --number_candidates 5 \
    #             --batch_size 64 \
    #             --probe_technique ${PROBE_TECHNIQUE} \
    #             --bert_model 'bert-base-cased' 
    #     done
    # done

    # for PROBE_TYPE in 'recommendation' 'search'
    # do
    #     for TASK in 'ml25m' 'gr' 'music'
    #     do
    #         python run_probes.py \
    #             --task $TASK \
    #             --probe_type ${PROBE_TYPE} \
    #             --input_folder $REPO_DIR/data/${PROBE_TYPE}/ \
    #             --output_folder $REPO_DIR/data/output_data/probes/ \
    #             --number_queries $NUMBER_PROBE_QUERIES \
    #             --number_candidates 5 \
    #             --batch_size 32 \
    #             --probe_technique ${PROBE_TECHNIQUE} \
    #             --bert_model 'bert-large-cased' 
    #     done
    # done
# done

# for PROBE_TECHNIQUE in 'mean-sim' 'cls-sim'
# do
#     for PROBE_TYPE in 'recommendation' 'search'
#     do
#         for TASK in 'ml25m' 'gr' 'music'
#         do
#             python run_probes.py \
#                 --task $TASK \
#                 --probe_type ${PROBE_TYPE} \
#                 --input_folder $REPO_DIR/data/${PROBE_TYPE}/ \
#                 --output_folder $REPO_DIR/data/output_data/probes/ \
#                 --number_queries $NUMBER_PROBE_QUERIES \
#                 --number_candidates 5 \
#                 --batch_size 32 \
#                 --probe_technique ${PROBE_TECHNIQUE} \
#                 --bert_model 'roberta-large'
#         done
#     done
# done