from IPython import embed
from transformers import BertTokenizer
from tqdm import tqdm

import pandas as pd
import argparse
import random
import json
import datetime

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
ITEM_SEP_TOKEN = " " + tokenizer.sep_token + " "
MASK_TOKEN = tokenizer.mask_token

negative_samples = 50

def generate_seq_data_movie_lens(path, path_item_names):
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

    all_movies = ratings['movieId'].unique()

    train = []
    valid = []
    test = []

    print("Sampling movies for each user")
    for user in tqdm(seen_movies.keys(), desc="User"):
        user_seen_movies = seen_movies[user]
        if len(user_seen_movies) > 2 :
            test_movie = id_to_name[user_seen_movies[-1]]
            valid_movie = id_to_name[user_seen_movies[-2]]
            train_movies = [id_to_name[m] for m in user_seen_movies[0:-2]]

            unseen_movies = set(all_movies) - set(user_seen_movies)
            negative_samples_valid = [id_to_name[m] for m in random.sample(unseen_movies, 100)]
            negative_samples_test = [id_to_name[m] for m in random.sample(unseen_movies, 100)]

            train.append( ITEM_SEP_TOKEN.join(train_movies))
            valid.append([ ITEM_SEP_TOKEN.join(train_movies),
                           valid_movie,
                           ITEM_SEP_TOKEN.join(negative_samples_valid)])
            test.append([ ITEM_SEP_TOKEN.join(train_movies),
                          test_movie,
                          ITEM_SEP_TOKEN.join(negative_samples_test)])

    train, valid, test = pd.DataFrame(train, columns=['X']), \
                         pd.DataFrame(valid, columns=['X', 'y', 'ns_y_list']), \
                         pd.DataFrame(test, columns=['X', 'y', 'ns_y_list'])
    return train, valid, test

def generate_seq_data_good_reads(path, path_item_names):
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
            i+=1
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

    all_books = ratings_df['bookId'].unique()

    train = []
    valid = []
    test = []

    print("Sampling books for each user")
    for user in tqdm(users_read_books.keys(), desc="User"):
        user_read_books = users_read_books[user]
        if len(user_read_books) > 2 :
            test_book = user_read_books[-1]
            valid_book = user_read_books[-2]
            train_books = user_read_books[0:-2]

            unread_books = list(set(all_books) - set(user_read_books))
            negative_samples_valid = random.sample(unread_books, 100)
            negative_samples_test = random.sample(unread_books, 100)

            train.append( ITEM_SEP_TOKEN.join(train_books))
            valid.append([ ITEM_SEP_TOKEN.join(train_books),
                           valid_book,
                           ITEM_SEP_TOKEN.join(negative_samples_valid)])
            test.append([ ITEM_SEP_TOKEN.join(train_books),
                          test_book,
                          ITEM_SEP_TOKEN.join(negative_samples_test)])

    train, valid, test = pd.DataFrame(train, columns=['X']), \
                         pd.DataFrame(valid, columns=['X', 'y', 'ns_y_list']), \
                         pd.DataFrame(test, columns=['X', 'y', 'ns_y_list'])

    return train, valid, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to generate data for ['ml25m', 'gr']")
    parser.add_argument("--ratings_path", default=None, type=str, required=True,
                        help="the path with files")
    parser.add_argument("--item_names_path", default=None, type=str, required=True,
                        help="the path with files")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="the path to_write files")
    args = parser.parse_args()

    if args.task == 'ml25m':
        train, valid, test = generate_seq_data_movie_lens(args.ratings_path, args.item_names_path, negative_samples)
    elif args.task == 'gr':
        train, valid, test = generate_seq_data_good_reads(args.ratings_path, args.item_names_path, negative_samples)
    else:
        raise Exception("task not accepted, choose from [ml25m,gr]")

    train.to_csv(args.output_path + "/train.csv", index=False)
    valid.to_csv(args.output_path + "/valid.csv", index=False)
    test.to_csv(args.output_path + "/test.csv", index=False)

if __name__ == "__main__":
    main()