#Running traditional IR baselines (QL, BM25, RM3)
sbatch --export=ALL,TASK=ml25m run_traditional_IR_dialogue.sbatch
sbatch --export=ALL,TASK=gr run_traditional_IR_dialogue.sbatch
sbatch --export=ALL,TASK=music run_traditional_IR_dialogue.sbatch

#Running BERT
sbatch --export=ALL,TASK=ml25m run_BERTRanker_dialogue.sbatch
sbatch --export=ALL,TASK=gr run_BERTRanker_dialogue.sbatch
sbatch --export=ALL,TASK=music run_BERTRanker_dialogue.sbatch