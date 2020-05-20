import pandas as pd
from IPython import embed
import logging
from tqdm import tqdm
import wikipedia as wiki
import argparse
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[ logging.StreamHandler() ]
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", default=None, type=str, required=True,
                    help="dataset to use")

args = parser.parse_args()

data_path = "/ssd/home/gustavo/recsys2020penha/data"
task = args.task
logging.info("Calculating wikipedia page existance for {}".format(task))
df = pd.read_csv("{}/recommendation/{}/categories.csv".format(data_path, task), nrows=100000)

if args.task == "ml25m":
    df["year"] = df.apply(lambda r: r["title"].split("(")[-1].split(")")[0], axis=1)
    df.to_csv("{}/recommendation/{}/item_years.csv".format(data_path, task), index=False)
elif args.task == "gr":
    book_titles = pd.read_csv("{}/recommendation/books_names.csv".format(data_path))
    book_titles['bookId'] = book_titles['bookId'].astype(str)
    book_titles = book_titles. \
        set_index('bookId').to_dict()['title']
    df_years = []
    with open("goodreads_books.json", "r") as f:
        for line in f:
            book = json.loads(line)
            if "publication_year" in book and "book_id" in book and book["book_id"] in book_titles:
                df_years.append([book_titles[book["book_id"]], book["publication_year"]])
    df_years = pd.DataFrame(df_years, columns=["title", "year"])
    df_years.to_csv("{}/recommendation/{}/item_years.csv".format(data_path, task), index=False)