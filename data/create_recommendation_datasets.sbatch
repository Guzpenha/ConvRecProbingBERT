#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=25000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

srun python make_seq_rec_data.py \
  --task music \
  --ratings_path ./search/CDs_and_Vinyl.json \
  --item_names_path ./search/meta_CDs_and_Vinyl.json \
  --output_path ./recommendation/music

srun python make_seq_rec_data.py \
  --task gr \
  --ratings_path ./recommendation/goodreads_interactions_dedup.json \
  --item_names_path ./recommendation/books_names.csv \
  --output_path ./recommendation/gr

srun python make_seq_rec_data.py \
  --task ml25m \
  --ratings_path ./recommendation/ml25m_ratings.csv \
  --item_names_path ./recommendation/movies_names.csv \
  --output_path ./recommendation/ml25m