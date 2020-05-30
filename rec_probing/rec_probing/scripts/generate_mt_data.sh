source /ssd/home/gustavo/recsys2020penha/env/bin/activate
REPO_DIR=/ssd/home/gustavo/recsys2020penha

for TASK in 'redial'
# for TASK in 'ml25m' 'gr' 'music' 'redial'
do
    python generate_data_for_mt.py \
        --task $TASK \
        --probe_type 'recommendation-pop' \
        --input_folder $REPO_DIR/data/recommendation/ \
        --output_folder $REPO_DIR/data/output_data/probes/ \
        --number_candidates 52 
done

for TASK in 'redial'
# for TASK in 'ml25m' 'gr' 'music' 'redial'
do
    python generate_data_for_mt.py \
        --task $TASK \
        --probe_type 'search' \
        --input_folder $REPO_DIR/data/search/ \
        --output_folder $REPO_DIR/data/output_data/probes/ \
        --number_candidates 52 
done