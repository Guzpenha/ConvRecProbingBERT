import json
import pandas as pd
from IPython import embed

book_titles = pd.read_csv("./recommendation/books_names.csv")
book_titles['bookId'] = book_titles['bookId'].astype(str)
book_titles = book_titles.\
    set_index('bookId').to_dict()['title']

df = []
with open("gooreads_book_genres_initial.json","r") as f:
    for i, line in enumerate(f):
        json_line = json.loads(line)
        if "book_id" in json_line and "genres" in json_line:
            genres = json_line["genres"]
            genres_flatten = []
            for genre in genres:
                for g in genre.split(", "):
                    genres_flatten.append(g)
            genres = set(genres_flatten)
            book_id = json_line["book_id"]
            if book_id in book_titles and len(genres)>0:
                title = book_titles[book_id]
                df.append([i, title, "|".join(genres)])
df = pd.DataFrame(df, columns=["index", "title", "genres"])
df.to_csv("categories.csv", index=False)