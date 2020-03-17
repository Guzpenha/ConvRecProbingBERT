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

DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/output_data/
ALL_RUNS=${DIR}random/1/,${DIR}random/2/,${DIR}random/3/,${DIR}random/4/,${DIR}random/5/,${DIR}popularity/1/,${DIR}bprmf/3/,${DIR}sasrec/9/,${DIR}sasrec/10/,${DIR}sasrec/11/


srun python evaluate_models.py \
  --predictions_folders $ALL_RUNS \
  --output_folder /tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/results/