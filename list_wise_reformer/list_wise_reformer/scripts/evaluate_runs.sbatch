#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_slice_learning/bin/activate

DIR=/tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/output_data/

# #ML25M
# ML25M_random="${DIR}random/1/,${DIR}random/2/,${DIR}random/3/,${DIR}random/4/,${DIR}random/5/,"
# ML25M_pop="${DIR}popularity/1/,"
# ML25M_bprmf="${DIR}bprmf/11/,${DIR}bprmf/12/,${DIR}bprmf/13/,${DIR}bprmf/14/,${DIR}bprmf/15/,"
# ML25M_sasrec="${DIR}sasrec/23/,${DIR}sasrec/24/,${DIR}sasrec/25/,${DIR}sasrec/26/,${DIR}sasrec/27/,"
# #ML25M_bert4rec="${DIR}bert4rec_torch/10/,${DIR}bert4rec_torch/11/,${DIR}bert4rec_torch/12/,${DIR}bert4rec_torch/13/,${DIR}bert4rec_torch/14/"

# #Amazon-Music
# music_random="${DIR}random/16/,${DIR}random/18/,${DIR}random/19/,${DIR}random/21/,${DIR}random/20/,"
# music_pop="${DIR}popularity/6/,"
# music_bprmf="${DIR}bprmf/16/,${DIR}bprmf/18/,${DIR}bprmf/19/,${DIR}bprmf/21/,${DIR}bprmf/23/,"
# music_sasrec="${DIR}sasrec/36/,${DIR}sasrec/37/,${DIR}sasrec/38/,${DIR}sasrec/39/,${DIR}sasrec/40/,"
# #music_bert4rec="${DIR}bert4rec_torch/15/,${DIR}bert4rec_torch/16/,${DIR}bert4rec_torch/17/,${DIR}bert4rec_torch/18/,${DIR}bert4rec_torch/19/"

# #GR
# GR_random="${DIR}random/22/,${DIR}random/23/,${DIR}random/24/,${DIR}random/25/,${DIR}random/26/,"
# GR_pop="${DIR}popularity/7/,"
# GR_bprmf="${DIR}bprmf/25/,${DIR}bprmf/26/,${DIR}bprmf/27/,${DIR}bprmf/28/,${DIR}bprmf/29/,"
# GR_sasrec="${DIR}sasrec/31/,${DIR}sasrec/32/,${DIR}sasrec/33/,${DIR}sasrec/34/,${DIR}sasrec/35/"
# #GR_bert4rec="${DIR}bert4rec_torch/20/,${DIR}bert4rec_torch/21/,${DIR}bert4rec_torch/22/,${DIR}bert4rec_torch/23/,${DIR}bert4rec_torch/24/"


# #ALL_RUNS="${ML25M_random}${ML25M_pop}${ML25M_bprmf}${ML25M_sasrec}${ML25M_bert4rec}"
# ALL_RUNS="${ML25M_random}${ML25M_pop}${ML25M_bprmf}${ML25M_sasrec}${music_random}${music_pop}${music_bprmf}${music_sasrec}${GR_random}${GR_pop}${GR_bprmf}${GR_sasrec}"

# srun python evaluate_models.py \
#   --predictions_folders $ALL_RUNS \
#   --output_folder /tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/results/

  #DIALOGUE
music_traditional_IR="${DIR}ql/1/,${DIR}rm3/7/,"
music_DAM="${DIR}dam/10/,${DIR}dam/11/,${DIR}dam/12/,${DIR}dam/13/,${DIR}dam/14/,"
music_MSN="${DIR}msn/10/,${DIR}msn/11/,${DIR}msn/12/,${DIR}msn/13/,${DIR}msn/14/,"
# music_BERT="${DIR}bert4dialogue/16/,${DIR}bert4dialogue/23/,${DIR}bert4dialogue/24/,${DIR}bert4dialogue/25/,${DIR}bert4dialogue/26/,"
# music_BERT_I_REC=
# music_BERT_I_SEARCH=

movies_traditional_IR="${DIR}ql/20/,${DIR}rm3/5/,"
movies_DAM="${DIR}dam/0/,${DIR}dam/1/,${DIR}dam/2/,${DIR}dam/3/,${DIR}dam/4/,"
movies_MSN="${DIR}msn/0/,${DIR}msn/1/,${DIR}msn/2/,${DIR}msn/3/,"
movies_BERT="${DIR}bert4dialogue/96/,${DIR}bert4dialogue/97/,${DIR}bert4dialogue/98/,${DIR}bert4dialogue/99/,${DIR}bert4dialogue/100/,"
# movies_BERT_I_REC=
# movies_BERT_I_SEARCH=

books_traditional_IR="${DIR}ql/21/,${DIR}rm3/6/,"
books_DAM="${DIR}dam/5/,${DIR}dam/6/,${DIR}dam/7/,${DIR}dam/8/,"
books_MSN="${DIR}msn/5/,${DIR}msn/6/,${DIR}msn/7/,${DIR}msn/8/,${DIR}msn/9/,"
books_BERT="${DIR}bert4dialogue/101/,${DIR}bert4dialogue/102/,${DIR}bert4dialogue/103/,${DIR}bert4dialogue/104/,${DIR}bert4dialogue/105/"
# books_BERT_I_REC=
# books_BERT_I_SEARCH=

ALL_RUNS="${music_traditional_IR}${music_DAM}${music_MSN}${music_BERT}${movies_traditional_IR}${movies_DAM}${movies_MSN}${movies_BERT}${books_traditional_IR}${books_DAM}${books_MSN}${books_BERT}"

srun python evaluate_models.py \
 --predictions_folders $ALL_RUNS \
 --output_folder /tudelft.net/staff-umbrella/conversationalsearch/recsys2020penha/data/results/ \
 --model_type ranker
                                           