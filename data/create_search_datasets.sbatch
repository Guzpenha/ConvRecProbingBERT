#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=15000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

srun python make_prod_search_data.py \
  --task gr \
  --reviews_path ./search/goodreads_reviews_dedup.json \
  --items_path ./recommendation/books_names.csv \
  --output_path ./search/gr/

srun python make_prod_search_data.py \
  --task ml25m \
  --reviews_path ./search/reviews.csv \
  --items_path ./recommendation/movies_names.csv \
  --output_path ./search/ml25m/