#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/output_data/sasrec/

srun python evaluate_models.py \
  --predictions_folders ${DIR}0/ \
  --output_folder /tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/results/