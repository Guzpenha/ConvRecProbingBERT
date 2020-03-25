#Running random and popularity baselines
sbatch --export=ALL,TASK=ml25m run_Random_and_Pop.sbatch
sbatch --export=ALL,TASK=gr run_Random_and_Pop.sbatch
sbatch --export=ALL,TASK=music run_Random_and_Pop.sbatch

#Runnning BPR-MF
sbatch --export=ALL,TASK=ml25m run_BPRMF.sbatch
sbatch --export=ALL,TASK=gr run_BPRMF.sbatch
sbatch --export=ALL,TASK=music run_BPRMF.sbatch

#Running SASRec (run it first uncommenting the dataset part first)
sbatch --export=ALL,TASK=ml25m,x=5 run_SASRec.sbatch
sbatch --export=ALL,TASK=gr,x=10 run_SASRec.sbatch
sbatch --export=ALL,TASK=music,x=15 run_SASRec.sbatch

#Running SASRec (run it first uncommenting the dataset part first)
sbatch --export=ALL,TASK=ml25m,x=5 run_BERT4Rec.sbatch
sbatch --export=ALL,TASK=gr,x=10 run_BERT4Rec.sbatch
sbatch --export=ALL,TASK=music,x=15 run_BERT4Rec.sbatch