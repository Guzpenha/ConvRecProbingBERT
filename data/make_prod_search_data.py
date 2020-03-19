from tqdm import tqdm
from IPython import embed

import pandas as pd
import argparse
import json
import random
from nltk.corpus import stopwords
import fasttext

language_pred_model = fasttext.load_model('lid.176.ftz')
stpwrds = set(stopwords.words('english'))

negative_samples = 50

def generate_product_search_movie_lens(path_reviews,
                                       path_items,
                                       path_ratings,
                                       negative_samples):
    movie_titles = pd.read_csv(path_items)
    movie_titles['movieId'] = movie_titles['movieId'].astype(str)
    movie_titles = movie_titles. \
        set_index('movieId').to_dict()['title']

    #getting items that were rated
    train, valid, test = pd.read_csv(path_ratings + "/train.csv"), \
                         pd.read_csv(path_ratings + "/valid.csv"), \
                         pd.read_csv(path_ratings + "/test.csv")
    train, valid, test = train[~train['query'].isnull()], \
                         valid[~valid['query'].isnull()], \
                         test[~test['query'].isnull()]
    items_rated = set()
    for _, r in train.iterrows():
        for item in r["query"].split(" [SEP] "):
            items_rated.add(item)
        items_rated.add(r["relevant_doc"])
    for _, r in valid.iterrows():
        items_rated.add(r["relevant_doc"])
    for _, r in test.iterrows():
        items_rated.add(r["relevant_doc"])
    del(train)
    del(valid)
    del(test)
    all_movies = list(movie_titles.values())

    instances = []
    with open(path_reviews, 'r', encoding="utf-8") as f:
        for line in tqdm(f):
            movie_name, review = line.split("\t")
            query = review[0:-2].lower() #remove \n
            relevant_doc = movie_name
            if movie_name in items_rated:
                for word in relevant_doc.lower().replace(")"," ") \
                            .replace("(", " ") \
                            .split(" "):
                    if word != "by" and word not in stpwrds:
                        # removing the book name from the review text
                        query = query.replace(" "+word+" ", " [ITEM_NAME] ")
                non_relevant_docs = random.sample(all_movies, negative_samples)
                instances.append([query, relevant_doc] + non_relevant_docs)

    train, valid, test = (instances[0: int(0.8*len(instances))],
                        instances[int(0.8*len(instances)) : int(0.9*len(instances))],
                        instances[int(0.9*len(instances)):])
    cols = ["query","relevant_doc"] + \
           ["non_relevant_"+str(i+1) for i in range(negative_samples)]
    train, valid, test = (pd.DataFrame(train, columns=cols),
                          pd.DataFrame(valid, columns=cols),
                          pd.DataFrame(test, columns=cols))
    return train, test, valid

def generate_product_search_good_reads(path_reviews,
                                       path_items,
                                       path_ratings,
                                       negative_samples):

    book_titles = pd.read_csv(path_items)
    book_titles['bookId'] = book_titles['bookId'].astype(str)
    book_titles = book_titles. \
        set_index('bookId').to_dict()['title']

    #getting items that were rated
    train, valid, test = pd.read_csv(path_ratings + "/train.csv"), \
                         pd.read_csv(path_ratings + "/valid.csv"), \
                         pd.read_csv(path_ratings + "/test.csv")
    train, valid, test = train[~train['query'].isnull()], \
                         valid[~valid['query'].isnull()], \
                         test[~test['query'].isnull()]
    items_rated = set()
    for _, r in train.iterrows():
        for item in r["query"].split(" [SEP] "):
            items_rated.add(item)
        items_rated.add(r["relevant_doc"])
    for _, r in valid.iterrows():
        items_rated.add(r["relevant_doc"])
    for _, r in test.iterrows():
        items_rated.add(r["relevant_doc"])
    del(train)
    del(valid)
    del(test)

    all_books = list(book_titles.values())
    instances = []
    with open(path_reviews) as f:
        for line in tqdm(f):
            review = json.loads(line)
            if review['book_id'] in book_titles and \
                    book_titles[review['book_id']] in items_rated:
                query = review['review_text'].replace("\n", " ")
                relevant_doc = book_titles[review['book_id']]
                language = language_pred_model.predict(relevant_doc, k=1)
                if len(query) > 50 and language[0][0] == '__label__en':
                    for word in relevant_doc.lower().replace(")"," ") \
                            .replace("(", " ") \
                            .split(" "):
                        if word != "by" and word not in stpwrds:
                            # removing the book name from the review text
                            query = query.replace(" "+word+" ", " [ITEM_NAME] ")
                    non_relevant_docs = random.sample(all_books, negative_samples)
                    instances.append([query, relevant_doc] + non_relevant_docs)

    train, valid, test = (instances[0: int(0.8*len(instances))],
                        instances[int(0.8*len(instances)) : int(0.9*len(instances))],
                        instances[int(0.9*len(instances)):])

    cols = ["query","relevant_doc"] + \
           ["non_relevant_"+str(i+1) for i in range(negative_samples)]

    train, valid, test = (pd.DataFrame(train, columns=cols),
                          pd.DataFrame(valid, columns=cols),
                          pd.DataFrame(test, columns=cols))
    return train, test, valid

def generate_product_search_amazon_music(path_reviews,
                                       path_items,
                                       path_ratings,
                                       negative_samples):

    album_titles = {}
    with open(path_items, 'r') as f:
        for l in tqdm(f):
            item = json.loads(l)
            if "title" in item:
                if len(item["title"]) < 400:
                    album_titles[item["asin"]] = item["title"]

    #getting items that were rated
    train, valid, test = pd.read_csv(path_ratings + "/train.csv"), \
                         pd.read_csv(path_ratings + "/valid.csv"), \
                         pd.read_csv(path_ratings + "/test.csv")
    train, valid, test = train[~train['query'].isnull()], \
                         valid[~valid['query'].isnull()], \
                         test[~test['query'].isnull()]
    items_rated = set()
    for _, r in train.iterrows():
        for item in r["query"].split(" [SEP] "):
            items_rated.add(item)
        items_rated.add(r["relevant_doc"])
    for _, r in valid.iterrows():
        items_rated.add(r["relevant_doc"])
    for _, r in test.iterrows():
        items_rated.add(r["relevant_doc"])

    del(train)
    del(valid)
    del(test)

    all_albums = list(album_titles.values())
    instances = []
    with open(path_reviews) as f:
        i=0
        for line in tqdm(f):
            i+=1
            review = json.loads(line)
            if review['asin'] in album_titles and 'reviewText' in review\
                    and album_titles[review['asin']] in items_rated:
                query = review['reviewText'].replace("\n", " ")
                relevant_doc = album_titles[review['asin']]
                language = language_pred_model.predict(relevant_doc, k=1)
                if len(query) > 50 and language[0][0] == '__label__en':
                    for word in relevant_doc.lower().replace(")"," ") \
                            .replace("(", " ") \
                            .split(" "):
                        if word != "by" and word not in stpwrds:
                            # removing the book name from the review text
                            query = query.replace(" "+word+" ", " [ITEM_NAME] ")
                    non_relevant_docs = random.sample(all_albums, negative_samples)
                    instances.append([query, relevant_doc] + non_relevant_docs)

    train, valid, test = (instances[0: int(0.8*len(instances))],
                        instances[int(0.8*len(instances)) : int(0.9*len(instances))],
                        instances[int(0.9*len(instances)):])

    cols = ["query","relevant_doc"] + \
           ["non_relevant_"+str(i+1) for i in range(negative_samples)]

    train, valid, test = (pd.DataFrame(train, columns=cols),
                          pd.DataFrame(valid, columns=cols),
                          pd.DataFrame(test, columns=cols))
    return train, test, valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to generate data for ['ml25m', 'gr', 'music']")
    parser.add_argument("--reviews_path", default=None, type=str, required=True,
                        help="the path with gr reviews file or paths for ml-25m")
    parser.add_argument("--items_path", default=None, type=str, required=True,
                        help="the path with item names")
    parser.add_argument("--ratings_path", default=None, type=str, required=True,
                        help="the path with ratings file in [query, relevant, non_relevant_1, ...] format .")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="the path to_write files")
    args = parser.parse_args()

    if args.task == 'ml25m':
        train, valid, test = generate_product_search_movie_lens(args.reviews_path,
                                                                args.items_path,
                                                                args.ratings_path,
                                                                negative_samples)
    elif args.task == 'gr':
        train, valid, test = generate_product_search_good_reads(args.reviews_path,
                                                                args.items_path,
                                                                args.ratings_path,
                                                                negative_samples)
    elif args.task == 'music':
        train, valid, test = generate_product_search_amazon_music(args.reviews_path,
                                                                args.items_path,
                                                                args.ratings_path,
                                                                negative_samples)
    else:
        raise Exception("task not accepted, choose from [ml25m, gr, music]")

    train.to_csv(args.output_path + "/train.csv", index=False)
    valid.to_csv(args.output_path + "/valid.csv", index=False)
    test.to_csv(args.output_path + "/test.csv", index=False)

if __name__ == "__main__":
    main()