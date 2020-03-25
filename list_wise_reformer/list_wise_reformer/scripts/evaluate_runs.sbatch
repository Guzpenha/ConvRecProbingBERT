#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/output_data/
ALL_RUNS=${DIR}random/1/,${DIR}random/2/,${DIR}random/3/,${DIR}random/4/,${DIR}random/5/,${DIR}popularity/1/,${DIR}bprmf/11/,${DIR}bprmf/12/,${DIR}bprmf/13/,${DIR}bprmf/14/,${DIR}bprmf/15/,${DIR}sasrec/23/,${DIR}sasrec/24/,${DIR}sasrec/25/,${DIR}sasrec/26/,${DIR}sasrec/27/

#finetunning SASREC
#ALL_RUNS=${DIR}sasrec/9/,${DIR}sasrec/10/,${DIR}sasrec/11/,${DIR}sasrec/12/,${DIR}sasrec/13/,${DIR}sasrec/14/,${DIR}sasrec/15/,${DIR}sasrec/16/,${DIR}sasrec/17/,${DIR}sasrec/18/,${DIR}sasrec/19/,${DIR}sasrec/20/,${DIR}sasrec/21/,${DIR}sasrec/22/

srun python evaluate_models.py \
  --predictions_folders $ALL_RUNS \
  --output_folder /tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/results/