# Probing LMs for Conversational Recommendation

In our paper *"What does BERT know about books, movies and music? Probing BERT for Conversational Recommendation"* we devise probing tasks to evaluate language models knowledge already stored in its parameters. We probe LMs (without any finetunning) for three types of knowledge: genre, search and recommendation.

## Running probes

1. Clone repo and install rec_probing in a python (>=3.6) virtual env:  
```
git clone https://github.com/Guzpenha/ConvRecProbingBERT.git
cd ConvRecProbingBERT

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

cd rec_probing
pip install -e .
```

2. Download required datasets and use scripts to preprocess them:
```
./download_data.sh
./run_datasets_creation.sh
```

This will download and preprocess a few datasets:
  
|| Recommendation | Search | Conversational Recommendation |
|-------------|-------------|------------|------------|
|Movies | [ML25M](https://grouplens.org/datasets/movielens/25m/): 25m movie ratings | Reviews crawled from IMDB | Conversations crawled from [/r/moviesuggestions/](https://www.reddit.com/r/MovieSuggestions/)
| Books | [GoodReads](https://github.com/MengtingWan/goodreads): 200m book interactions | Reviews from GoodReads | Conversations crawled from [/r/booksuggestions/](https://www.reddit.com/r/MovieSuggestions/) |
| Music | [Amazon-Music](https://nijianmo.github.io/amazon/index.html): 2.3m ratings/reviews | Reviews from Amazon-Music | Conversations crawled from [/r/musicuggestions/](https://www.reddit.com/r/musicsuggestions/) | 

As well as categories information for items of the 3 domains.

3. Use our python scripts to run probes:

```
# Search and Recommendation
python run_probes.py \
    --task $TASK \
    --probe_type ${PROBE_TYPE} \
    --input_folder $REPO_DIR/data/${PROBE_TYPE}/ \
    --output_folder $REPO_DIR/data/output_data/probes/ \
    --number_queries $NUMBER_PROBE_QUERIES \
    --number_candidates 5 \
    --batch_size 64 \
    --probe_technique ${PROBE_TECHNIQUE} \
    --bert_model 'bert-base-cased' 
```

Where PROBE_TYPE can be ['recommendation', 'search'], PROBE_TECHNIQUE can be ['mean-sim', 'cls-sim', 'nsp'] and TASK can be ['ml25m' 'gr' 'music'] for the domains of movies, books and music respectivelly.

```
# Genres
python run_mlm_probe.py \
    --task $TASK \
    --input_folder $REPO_DIR/data/recommendation/ \
    --output_folder $REPO_DIR/data/output_data/probes/ \
    --number_queries $NUMBER_PROBE_QUERIES \
    --batch_size 32 \
    --sentence_type ${SENTENCE_TYPE} \
    --bert_model 'roberta-large'
```
Where SENTENCE_TYPE can be ['no-item', 'type-I', 'type-II'] and TASK can be ['ml25m' 'gr' 'music'] for the domains of movies, books and music respectivelly.

## Running response ranking for reddit conv. recommendation data

In order to get the results from Table 7 of the paper, regarding models conversation response ranking results on the conversational recomendation reddit data, use:


```
cd list_wise_reformer
pip install -e .
cd list_wise_reformer/scripts
./run_all_dialogue_baselines.sh
```

Ignore that the package is named list_wise_reformer. It contains several baselines for dialogue, search and recommendation, including a prototype of a list wise Reformer model.

## Infusing knowledge
We interleave the probing tasks with the response ranking task by creating a dataset with half instances from each task. We create the dataset using the script rec_probing/rec_probing/scripts/generate_data_for_mt.py. We then use the previous script to train the model on this data.

## Experiments with ReDial Data

We use the same framework from the other tasks, the difference is that we need to create the adversarial test data. For that we use the script data/genereate_adversarial_test.py.


Reference
```
@inproceedings{10.1145/3383313.3412249,
  author = {Penha, Gustavo and Hauff, Claudia},
  title = {What Does BERT Know about Books, Movies and Music? Probing BERT for Conversational Recommendation},
  year = {2020},
  isbn = {9781450375832},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3383313.3412249},
  doi = {10.1145/3383313.3412249},
  booktitle = {Fourteenth ACM Conference on Recommender Systems},
  pages = {388–397},
  numpages = {10},
  keywords = {conversational search, probing, conversational recommendation},
  location = {Virtual Event, Brazil},
  series = {RecSys '20}
}
```
