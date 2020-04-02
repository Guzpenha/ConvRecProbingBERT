#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
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
srun python trainer_example.py
done
