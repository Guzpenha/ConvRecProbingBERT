from list_wise_reformer.models.utils import toDAMFormat
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
    parser.add_argument("--dam_folder", default=None, type=str, required=True,
                        help="the folder containing code for bert4rec")
    args = parser.parse_args()

    train = pd.read_csv(args.data_folder+args.task+"/train.csv", lineterminator= "\n")
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv", lineterminator= "\n")

    #transform data to DAM format and write to files
    dataset, vocab = toDAMFormat(train, valid)

    os.makedirs(args.dam_folder+"/DAM/data/"+args.task, exist_ok=True)

    with open(args.dam_folder+"/DAM/data/"+args.task+"/data.pkl", 'wb') as f:
        pickle.dump(dataset, f, protocol=2)

    with open(args.dam_folder+"/DAM/data/"+args.task+"/word2id", 'wb') as f:
        for w, id in vocab.items():
            line = w+"\t"+str(id)+"\n"
            f.write(line.encode("utf8"))

if __name__ == "__main__":
    main()