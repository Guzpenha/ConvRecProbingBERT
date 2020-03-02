mkdir data/recommendation/gr
mkdir data/recommendation/ml25m
mkdir data/search/gr
mkdir data/search/ml25m


./data/create_dialogue_datasets.sbatch
./data/create_recommendation_datasets.sbatch
./data/create_search_datasets.sbatch