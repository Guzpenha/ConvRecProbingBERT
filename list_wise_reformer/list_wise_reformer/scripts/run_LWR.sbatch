#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=END


source /home/nfs/gustavopenha/env_slice_learning/bin/activate
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha
TASK=ml25m

cd ../models/LWR/

for SEED in 42
do
srun python main.py \
    --seed $SEED \
    --num_epochs 200 \
    --data_folder $REPO_DIR/data/recommendation/ \
    --output_dir $REPO_DIR/data/output_data/lwr \
    --task $TASK \
    --validate_epochs 1 \
    --sample_data 10000 \
    --max_seq_len 2048 \
    --train_batch_size 10 \
    --val_batch_size 10 \
    --save_model False
done
