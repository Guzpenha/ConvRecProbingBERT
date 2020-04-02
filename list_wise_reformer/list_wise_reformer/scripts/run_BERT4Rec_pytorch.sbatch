#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=END


source /home/nfs/gustavopenha/env_slice_learning/bin/activate
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha
#TASK=ml25m

# Creating TASK dataset in bert4rec pytorch format (run this only the first time)
srun python create_BERT4rec_data_pytorch.py  \
    --task ${TASK} \
    --data_folder $REPO_DIR/data/recommendation/ \
    --bert4rec_folder $REPO_DIR/list_wise_reformer/list_wise_reformer/models/

cd ../models/BERT4Rec-VAE-Pytorch/

#x=0
for SEED in 42
do
python main.py \
    --model_init_seed $SEED \
    --num_epochs 200 \
    --output_predictions_folder $REPO_DIR/data/output_data/bert4rec_torch/${x} \
    --task $TASK \
    --template train_bert
x=$((x+1))
done
