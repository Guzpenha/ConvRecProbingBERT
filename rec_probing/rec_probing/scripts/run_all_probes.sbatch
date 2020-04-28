#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha
for TASK in 'ml25m' 'gr' 'music'
do
    for NUM_CANDIDATES in 1 5    
    do
        srun python run_probes.py \
            --task $TASK \
            --probe_type recommendation \
            --input_folder $REPO_DIR/data/recommendation/ \
            --output_folder $REPO_DIR/data/output_data/probes/ \
            --number_queries 100 \
            --number_candidates $NUM_CANDIDATES \
            --batch_size 8

    done
done
