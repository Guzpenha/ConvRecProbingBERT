from list_wise_reformer.models.utils import toBERT4RecPytorchFormat
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
                        help="the task to run baselines for ['ml25m', 'gr', 'music']")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--bert4rec_folder", default=None, type=str, required=True,
                        help="the folder containing code for bert4rec")
    args = parser.parse_args()

    train = pd.read_csv(args.data_folder+args.task+"/train.csv",
                        lineterminator= "\n")
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv",
                        lineterminator= "\n")

    #transform data to bert4rec format and write to files
    dataset = toBERT4RecPytorchFormat(train, valid)

    path = args.bert4rec_folder+"/BERT4Rec-VAE-Pytorch/Data/preprocessed/"+args.task+ \
                "_min_rating2-min_uc5-min_sc0-splitleave_one_out"
    os.makedirs(path, exist_ok=True)

    with open(path+"/dataset.pkl", 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()