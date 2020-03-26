#Running traditional IR baselines (QL, BM25, RM3)
sbatch --export=ALL,TASK=movies run_traditional_IR_dialogue.sbatch
sbatch --export=ALL,TASK=books run_traditional_IR_dialogue.sbatch
sbatch --export=ALL,TASK=music run_traditional_IR_dialogue.sbatch

#Running U2U
sbatch --export=ALL,TASK=movies run_U2U.sbatch
sbatch --export=ALL,TASK=books run_U2U.sbatch
sbatch --export=ALL,TASK=music run_U2U.sbatch

#Running DAM
sbatch --export=ALL,TASK=movies run_DAM.sbatch
sbatch --export=ALL,TASK=books run_DAM.sbatch
sbatch --export=ALL,TASK=music run_DAM.sbatch

#Running BERT
sbatch --export=ALL,TASK=movies run_BERTRanker_dialogue.sbatch
sbatch --export=ALL,TASK=books run_BERTRanker_dialogue.sbatch
sbatch --export=ALL,TASK=music run_BERTRanker_dialogue.sbatch