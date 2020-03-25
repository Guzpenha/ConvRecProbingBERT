import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import models.net as net
import utils.evaluation as eva
#for douban
#import utils.douban_evaluation as eva

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test
import argparse
import pandas as pd

# configure
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run baselines for ['music', 'books', 'movies']")
    parser.add_argument("--output_predictions_folder", default=None, type=str, required=True,
                        help="the folder to output data to")
    parser.add_argument("--seed", default=None, type=int, required=True,
                        help="random seed")
    parser.add_argument("--num_epochs", default=2, type=int, required=False,
                        help="Number of epochs for recommenders that do optimization.")
    args = parser.parse_args()

    vocab_size=0
    with open("./data/"+args.task+"/word2id", 'r') as f:
        for _ in f:
            vocab_size+=1

    conf = {
        "task": args.task,
        "output_predictions_folder": args.output_predictions_folder,
            # "/Users/gustavopenha/personal/recsys20/data/output_data/dam/1",
        # "data_path": "./data/ubuntu/data_small.pkl",
        "data_path": "./data/"+args.task+"/data.pkl",
        # "save_path": "./output/ubuntu/temp/",
        "save_path": "./output/"+args.task+"/temp/",
        # "word_emb_init": "./data/word_embedding.pkl",
        "word_emb_init": None,
        "init_model": None, #should be set for test

        "rand_seed": args.seed,

        "drop_dense": None,
        "drop_attention": None,

        "is_mask": True,
        "is_layer_norm": True,
        "is_positional": False,

        "stack_num": 5,
        "attention_type": "dot",

        "learning_rate": 1e-3,
        "vocab_size": vocab_size,
        "emb_size": 200,
        "batch_size": 51,

        "max_turn_num": 9,
        "max_turn_len": 50,

        "max_to_keep": 1,
        "num_scan_data": args.num_epochs,
        "_EOS_" : 0,
        # "_EOS_": 28270, #1 for douban data
        "final_n_class": 1,
    }


    model = net.Net(conf)
    train.train(conf, model)

#test and evaluation, init_model in conf should be set
# test.test(conf, model)

if __name__ == "__main__":
    main()