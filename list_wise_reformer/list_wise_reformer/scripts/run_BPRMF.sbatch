#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha

#TASK=ml25m
#TASK=gr
#TASK=music

#for NUM_EPOCHS in 2 5 10 15 30 50
#do
#  srun python run_rec_model.py \
#    --num_epochs $NUM_EPOCHS \
#    --task ml25m \
#    --data_folder $REPO_DIR/data/recommendation/ \
#    --seed 42 \
#    --recommender bprmf \
#    --output_dir $REPO_DIR/data/output_data/bprmf
#done

for SEED in 42 1 2 3 4
do
  srun python run_rec_model.py \
    --num_epochs 30 \
    --task $TASK \
    --data_folder $REPO_DIR/data/recommendation/ \
    --seed $SEED \
    --recommender bprmf \
    --output_dir $REPO_DIR/data/output_data/bprmf
done