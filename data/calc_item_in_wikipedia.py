import pandas as pd
from IPython import embed
import logging
from tqdm import tqdm
import wikipedia as wiki
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[ logging.StreamHandler() ]
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", default=None, type=str, required=True,
                    help="dataset to use")

args = parser.parse_args()

# data_path = "/home/guzpenha/personal/recsys2020penha/data"
data_path = "/ssd/home/gustavo/recsys2020penha/data"
task = args.task
logging.info("Calculating wikipedia page existance for {}".format(task))
df = pd.read_csv("{}/recommendation/{}/categories.csv".format(data_path, task), nrows=100000)
title_in_wiki = []
for row in tqdm(df.itertuples(), total=df.shape[0]):
    search_wiki = wiki.search(row.title)
    try:        
        if len(search_wiki)>0:
            page = wiki.page(search_wiki[0])
            title_in_wiki.append([row.title, len(search_wiki)>0, search_wiki, len(page.content)])
        else:
            title_in_wiki.append([row.title, len(search_wiki)>0, search_wiki, 0])
    except:
        title_in_wiki.append([row.title, False, "exception", 0])

in_wiki_df = pd.DataFrame(title_in_wiki, columns=["title", "in_wiki", "res_wiki_search", "wiki_page_length"])
in_wiki_df.to_csv("{}/recommendation/{}/in_wiki.csv".format(data_path, task), index=False)