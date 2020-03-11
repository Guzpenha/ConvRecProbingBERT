#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
source /home/nfs/gustavopenha/env_slice_learning/bin/activate


REPO_DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha

#Creating ml25m dataset in SASRec format
mkdir $REPO_DIR/list_wise_reformer/list_wise_reformer/models/SASRec/data/
srun python create_sasrec_data.py  \
    --task ml25m \
    --data_folder $REPO_DIR/data/recommendation/ \
    --sasrec_folder $REPO_DIR/list_wise_reformer/list_wise_reformer/models


#Running SASRec
source /home/nfs/gustavopenha/env_2.7/bin/activate
cd SASRec
python main.py --dataset=train_ml25m \
  --train_dir=default \
  --num_epochs=200 \
  --eval_epochs=10 \
  --maxlen=100 \
  --dataset_list_valid valid_ml25m.csv \
  --output_predictions_folder /tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/output_data/sasrec


