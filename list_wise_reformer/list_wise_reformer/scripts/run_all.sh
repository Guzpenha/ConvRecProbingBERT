
#Running random and popularity baselines
sbatch --export=ALL,TASK=ml25m run_Random_and_Pop.sbatch
sbatch --export=ALL,TASK=gr run_Random_and_Pop.sbatch
sbatch --export=ALL,TASK=music run_Random_and_Pop.sbatch

