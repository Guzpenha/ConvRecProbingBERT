from list_wise_reformer.models.utils import toMSNFormat
import os
import pandas as pd
import argparse
import logging
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run baselines for ['music', 'books', 'movies']")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--msn_folder", default=None, type=str, required=True,
                        help="the folder containing code for bert4rec")
    args = parser.parse_args()

    train = pd.read_csv(args.data_folder+args.task+"/train.csv", lineterminator= "\n")
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv", lineterminator= "\n")

    #transform data to DAM format and write to files
    train, valid, vocab_embed = toMSNFormat(train, valid)
    os.makedirs(args.msn_folder+"/MSN/dataset/"+args.task, exist_ok=True)

    with open(args.msn_folder+"/MSN/dataset/"+args.task+"/train.pkl", 'wb') as f:
        pickle.dump(train, f)
    with open(args.msn_folder+"/MSN/dataset/"+args.task+"/test.pkl", 'wb') as f:
        pickle.dump(valid, f)
    with open(args.msn_folder + "/MSN/dataset/" + args.task + "/vocab_and_embeddings.pkl", 'wb') as f:
        pickle.dump(vocab_embed, f)


if __name__ == "__main__":
    main()