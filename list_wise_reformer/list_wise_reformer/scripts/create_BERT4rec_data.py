from list_wise_reformer.models.utils import toBERT4RecFormat
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
                        help="the task to run baselines for ['ml25m', 'gr']")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--bert4rec_folder", default=None, type=str, required=True,
                        help="the folder containing code for bert4rec")
    args = parser.parse_args()

    train = pd.read_csv(args.data_folder+args.task+"/train.csv")
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv")

    #transform data to SASRec format and write to files
    train_sasrec, valid_sasrec = toBERT4RecFormat(train, valid)

    for data_set, name in [(train_sasrec, "train_"+args.task+".txt")]:
        with open(args.bert4rec_folder+"/BERT4Rec/data/"+name, 'w') as f:
            for user, item in data_set:
                f.write(str(user)+" "+str(item)+"\n")

    valid_sasrec.\
        to_csv(args.bert4rec_folder+"/BERT4Rec/data/valid_"+args.task+".csv",
                                   index=False)

if __name__ == "__main__":
    main()