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

$REPO_DIR=/Users/gustavopenha/personal/recsys20/list_wise_reformer/

srun python run_sasrec.py  \
    --task ml25m \
    --data_folder $REPO_DIR/data/recommendation/ \
    --sasrec_folder $REPO_DIR/models

#source /home/nfs/gustavopenha/env_2.7/bin/activate


