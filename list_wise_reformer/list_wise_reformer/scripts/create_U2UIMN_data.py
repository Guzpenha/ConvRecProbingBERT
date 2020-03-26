from list_wise_reformer.models.utils import toU2UIMNFormat
import os
import pandas as pd
import argparse
import logging

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
    parser.add_argument("--u2u_folder", default=None, type=str, required=True,
                        help="the folder containing code for bert4rec")
    args = parser.parse_args()

    train = pd.read_csv(args.data_folder+args.task+"/train.csv", lineterminator= "\n")
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv", lineterminator= "\n")

    #transform data to DAM format and write to files
    train, valid, responses, vocab, char_vocab =  toU2UIMNFormat(train, valid)

    os.makedirs(args.u2u_folder+"/U2U-IMN/data/"+args.task, exist_ok=True)
    pd.DataFrame(train).\
        to_csv(args.u2u_folder+"/U2U-IMN/data/"+args.task+"/train.txt",
               sep='\t', index=False, header=False)
    pd.DataFrame(valid).\
        to_csv(args.u2u_folder+"/U2U-IMN/data/"+args.task+"/valid.txt",
               sep='\t', index=False, header=False)
    pd.DataFrame([[v, k] for (k,v) in responses.items()]). \
        to_csv(args.u2u_folder+"/U2U-IMN/data/"+args.task+"/responses.txt",
               sep='\t', index=False, header=False)
    pd.DataFrame([[k, v] for (k,v) in vocab.items()]). \
        to_csv(args.u2u_folder+"/U2U-IMN/data/"+args.task+"/vocab.txt",
               sep='\t', index=False, header=False)
    pd.DataFrame([[v, k] for (k,v) in char_vocab.items() if k != ' ']). \
        to_csv(args.u2u_folder+"/U2U-IMN/data/"+args.task+"/char_vocab.txt",
               sep='\t', index=False, header=False)
if __name__ == "__main__":
    main()