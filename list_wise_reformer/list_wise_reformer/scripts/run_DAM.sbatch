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
source /home/nfs/gustavopenha/env_2.7/bin/activate

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha
TASK=music

# Creating TASK dataset in DAM format (run this only the first time)
#source /home/nfs/gustavopenha/env_slice_learning/bin/activate
#srun python create_DAM_data.py  \
#    --task ${TASK} \
#    --data_folder $REPO_DIR/data/dialogue/ \
#    --dam_folder $REPO_DIR/list_wise_reformer/list_wise_reformer/models/

cd ../models/DAM/

x=0
for SEED in 42
do
srun python main.py \
    --seed $SEED \
    --num_epochs 10 \
    --output_predictions_folder $REPO_DIR/data/output_data/dam/${x} \
    --task $TASK
x=$((x+1))
done
