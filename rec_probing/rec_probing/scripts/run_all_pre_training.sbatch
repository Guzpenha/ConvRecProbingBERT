#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha
NUMBER_PROBE_QUERIES=100001

for PROBE_TYPE in 'recommendation-pop'
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
