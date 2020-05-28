from gensim.summarization.bm25 import BM25
from IPython import embed
import pandas as pd
import numpy as np
import argparse
import heapq
from tqdm import tqdm
import random

negative_samples = 50
random.seed(42)

def generate_conv_data(in_path, subreddit):
    df_conv = pd.read_csv(in_path, lineterminator= "\n")
    df_conv = df_conv[df_conv['subreddit'] == subreddit]
    df_conv['query'] = df_conv['query'].astype(str)
    df_conv['relevant_response'] = df_conv['relevant_response'].astype(str)

    documents = np.array(df_conv['relevant_response'])
    corpus = [context.split(" ") for context in documents]
    bm25 = BM25(corpus)

    cache = {}
    instances = []
    index_subreddit = []

    for idx, r in tqdm([x for x in df_conv.iterrows()]):
        if r['query'] in cache:
            max_positions = cache[r['query']]
        else:
            scores = np.array(bm25.get_scores(str(r['query']).split(" ")))
            max_positions = heapq.nlargest(negative_samples,
                                       range(len(scores)),
                                       scores.take)
            cache[r['query']] = max_positions

        while idx in max_positions:
            new_doc = random.sample(range(len(documents)), 1)[0]
            max_positions[max_positions.index(idx)] = new_doc

        candidates = documents[max_positions]

        instances.append([
            r['query'],
            r['relevant_response']
        ] + list(candidates))
        index_subreddit.append([idx, r['subreddit']])

    # random.shuffle(instances) #<-- this shouldnt be here. instances from same dialogue will be spread over different data splits
    train, valid, test = (instances[0: int(0.8*len(instances))],
                        instances[int(0.8*len(instances)) : int(0.9*len(instances))],
                        instances[int(0.9*len(instances)):])

    cols = ["query", "relevant_doc"] + \
           ["non_relevant_"+str(i+1) for i in range(negative_samples)]

    train, valid, test = (pd.DataFrame(train, columns=cols),
                          pd.DataFrame(valid, columns=cols),
                          pd.DataFrame(test, columns=cols))

    indexes = pd.DataFrame(index_subreddit, columns = ['index', 'subreddit'])

    return train, valid, test, indexes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations_path", default=None, type=str, required=True,
                        help="the path with context-response pairs")
    parser.add_argument("--subreddit", default=None, type=str, required=True,
                        help="the subreddit to generate training instances ['MovieSuggestions', 'booksuggestions', musicsuggestions]")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="the path to_write files")
    args = parser.parse_args()

    train, valid, test, subreddit_index = generate_conv_data(args.conversations_path,
                                                             args.subreddit)
    train.to_csv(args.output_path + "/train.csv", index=False)
    valid.to_csv(args.output_path + "/valid.csv", index=False)
    test.to_csv(args.output_path + "/test.csv", index=False)

    subreddit_index.to_csv(args.output_path + "/subreddit_indexes.csv", index=False)

if __name__ == '__main__':
    main()