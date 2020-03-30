from transformers import BertTokenizer
from tqdm import tqdm
from IPython import embed

import pandas as pd
import numpy as np
import argparse
import json
import datetime

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
ITEM_SEP_TOKEN = " " + tokenizer.sep_token + " "
MASK_TOKEN = tokenizer.mask_token

negative_samples = 50

def generate_seq_data_amazon_music(path, path_item_names,
                                   negative_samples):
    album_titles = {}
    with open(path_item_names, 'r') as f:
        for l in tqdm(f):
            item = json.loads(l)
            if "title" in item:
                if len(item["title"]) < 400:
                    album_titles[item["asin"]] = item["title"]

    ratings = []
    with open(path, 'r') as f:
        for l in tqdm(f):
            interaction = json.loads(l)
            if interaction["asin"] in album_titles:
                ratings.append([interaction["reviewerID"],
                                album_titles[interaction["asin"]],
                                interaction["unixReviewTime"]])
    ratings_df = pd.DataFrame(ratings, columns=["userId", "albumId", "timestamp"])

    print("Sorting and getting music album by user")
    rated_albums = ratings_df.sort_values(["userId", "timestamp"]).\
        groupby('userId')['albumId'].\
        apply(list).reset_index(name='rated_albums').\
        set_index("userId").to_dict()['rated_albums']
    album_by_popularity = ratings_df['albumId'].values

    train = []
    valid = []
    test = []

    print("Sampling movies for each user")
    for user in tqdm([u for u in rated_albums.keys()], desc="User"):
        user_rated_albums = rated_albums[user]
        if len(user_rated_albums) > 3 :
            test_album = user_rated_albums[-1]
            valid_album = user_rated_albums[-2]
            train_albums = user_rated_albums[0:-2]

            neg_samples_by_set = []
            for _ in range(3):
                neg_samples = [m for m in
                               np.random.choice(album_by_popularity, negative_samples)
                               if m not in user_rated_albums]
                while len(neg_samples) < negative_samples:
                    add_samples = [m for m in
                                   np.random.choice(album_by_popularity, negative_samples)
                                   if m not in user_rated_albums]
                    neg_samples+=add_samples
                neg_samples_by_set.append(neg_samples[:negative_samples])

            train.append([ITEM_SEP_TOKEN.join(train_albums[:-1]),
                          train_albums[-1]] + neg_samples_by_set[0])
            valid.append([ITEM_SEP_TOKEN.join(train_albums),
                          valid_album] + neg_samples_by_set[1])
            test.append([ITEM_SEP_TOKEN.join(train_albums),
                         test_album] + neg_samples_by_set[2])

    cols = ["query","relevant_doc"] + \
           ["non_relevant_"+str(i+1) for i in range(negative_samples)]

    train, valid, test = pd.DataFrame(train, columns=cols), \
                         pd.DataFrame(valid, columns=cols), \
                         pd.DataFrame(test, columns=cols)

    return train, valid, test

def generate_seq_data_movie_lens(path, path_item_names, negative_samples):
    ratings = pd.read_csv(path)
    ratings['movieId'] = ratings['movieId'].astype(str)
    movie_names = pd.read_csv(path_item_names)
    movie_names['movieId'] = movie_names['movieId'].astype(str)
    id_to_name = movie_names.set_index('movieId').to_dict()['title']

    print("Sorting and getting movies by user")
    seen_movies = ratings.sort_values(["userId", "timestamp"]).\
        groupby('userId')['movieId'].\
        apply(list).reset_index(name='seen_movies').\
        set_index("userId").to_dict()['seen_movies']

    movies_by_popularity = ratings['movieId'].values

    train = []
    valid = []
    test = []

    print("Sampling movies for each user")
    for user in tqdm(seen_movies.keys(), desc="User"):
        user_seen_movies = seen_movies[user]
        if len(user_seen_movies) > 3 :
            test_movie = id_to_name[user_seen_movies[-1]]
            valid_movie = id_to_name[user_seen_movies[-2]]
            train_movies = [id_to_name[m] for m in user_seen_movies[0:-2]]

            neg_samples_by_set = []
            for _ in range(3):
                neg_samples = [id_to_name[m] for m in
                               np.random.choice(movies_by_popularity, negative_samples)
                               if m not in user_seen_movies]
                while len(neg_samples) < negative_samples:
                    add_samples = [id_to_name[m] for m in
                                   np.random.choice(movies_by_popularity, negative_samples)
                                   if m not in user_seen_movies]
                    neg_samples+=add_samples
                neg_samples_by_set.append(neg_samples[:negative_samples])

            train.append([ITEM_SEP_TOKEN.join(train_movies[:-1]),
                          train_movies[-1]] + neg_samples_by_set[0])
            valid.append([ITEM_SEP_TOKEN.join(train_movies),
                          valid_movie] + neg_samples_by_set[1])
            test.append([ITEM_SEP_TOKEN.join(train_movies),
                         test_movie] + neg_samples_by_set[2])

    cols = ["query","relevant_doc"] + \
           ["non_relevant_"+str(i+1) for i in range(negative_samples)]

    train, valid, test = pd.DataFrame(train, columns=cols), \
                         pd.DataFrame(valid, columns=cols), \
                         pd.DataFrame(test, columns=cols)
    return train, valid, test

def generate_seq_data_good_reads(path, path_item_names,
                                 negative_samples, max_users = 200000):
    book_titles = pd.read_csv(path_item_names)
    book_titles['bookId'] = book_titles['bookId'].astype(str)
    book_titles = book_titles.\
        set_index('bookId').to_dict()['title']

    path_ratings = path
    ratings = []
    print("reading and filtering from big interactions file (78GB)")
    with open(path_ratings, 'r') as f:
        next(f) #header
        i=0
        for l in tqdm(f):
            # i+=1
            # if i > 3000:
            #     break
            interaction = json.loads(l)
            if interaction["is_read"] and interaction["book_id"] in book_titles:
                ratings.append([interaction["user_id"],
                                book_titles[interaction["book_id"]],
                                interaction["date_updated"]])

    def convert_time(r):
        clean_str = r['timestamp'][0:-10] + r['timestamp'][-4:]
        ts = datetime.datetime.\
            strptime(clean_str, '%a %b %d %H:%M:%S %Y').timestamp()
        return ts

    ratings_df = pd.DataFrame(ratings, columns=["userId", "bookId", "timestamp"])
    ratings_df["timestamp_unix"] = ratings_df.\
        apply(lambda r,f=convert_time: f(r), axis=1)

    users_read_books = ratings_df.sort_values(["userId", "timestamp_unix"]).\
        groupby('userId')['bookId'].\
        apply(list).reset_index(name='users_read_books').\
        set_index("userId").to_dict()['users_read_books']

    books_by_popularity = ratings_df['bookId'].values

    train = []
    valid = []
    test = []

    print("Sampling books for each user")
    for user in tqdm([u for u in users_read_books.keys()][:max_users], desc="User"):
        user_read_books = users_read_books[user]
        if len(user_read_books) > 3 :
            test_book = user_read_books[-1]
            valid_book = user_read_books[-2]
            train_books = user_read_books[0:-2]

            neg_samples_by_set = []
            for _ in range(3):
                neg_samples = [b for b in
                               np.random.choice(books_by_popularity, negative_samples)
                               if b not in user_read_books]
                while len(neg_samples) < negative_samples:
                    add_samples = [b for b in
                                   np.random.choice(books_by_popularity, negative_samples)
                                   if b not in user_read_books]
                    neg_samples+=add_samples
                neg_samples_by_set.append(neg_samples[:negative_samples])

            train.append([ITEM_SEP_TOKEN.join(train_books[:-1]),
                          train_books[-1]] + neg_samples_by_set[0])
            valid.append([ITEM_SEP_TOKEN.join(train_books),
                          valid_book] + neg_samples_by_set[1])
            test.append([ITEM_SEP_TOKEN.join(train_books),
                         test_book] + neg_samples_by_set[2])

    cols = ["query", "relevant_doc"] + \
           ["non_relevant_" + str(i + 1) for i in range(negative_samples)]

    train, valid, test = pd.DataFrame(train, columns=cols), \
                         pd.DataFrame(valid, columns=cols), \
                         pd.DataFrame(test, columns=cols)
    return train.loc[~train['query'].isnull()],\
           valid.loc[~valid['query'].isnull()],\
           test.loc[~test['query'].isnull()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to generate data for ['ml25m', 'gr', 'music']")
    parser.add_argument("--ratings_path", default=None, type=str, required=True,
                        help="the path with files")
    parser.add_argument("--item_names_path", default=None, type=str, required=True,
                        help="the path with files")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="the path to_write files")
    args = parser.parse_args()

    if args.task == 'ml25m':
        train, valid, test = generate_seq_data_movie_lens(args.ratings_path,
                                                          args.item_names_path,
                                                          negative_samples)
    elif args.task == 'gr':
        train, valid, test = generate_seq_data_good_reads(args.ratings_path,
                                                          args.item_names_path,
                                                          negative_samples)
    elif args.task == 'music':
        train, valid, test = generate_seq_data_amazon_music(args.ratings_path,
                                                          args.item_names_path,
                                                          negative_samples)
    else:
        raise Exception("task not accepted, choose from [ml25m, gr, music]")

    train.to_csv(args.output_path + "/train.csv", index=False)
    valid.to_csv(args.output_path + "/valid.csv", index=False)
    test.to_csv(args.output_path + "/test.csv", index=False)

if __name__ == "__main__":
    main()