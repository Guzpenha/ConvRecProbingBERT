from list_wise_reformer.models.utils import generate_anserini_json_collection
from IPython import embed
import pandas as pd
import argparse
import logging
import json

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
                        help="the task to generate indexable anserini data for ['movies', 'music', books]")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="the folder to output collection in anserini json format.")
    args = parser.parse_args()

    train = pd.read_csv(args.data_folder+args.task+"/train.csv",
                        lineterminator= "\n").fillna(' ')
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv",
                        lineterminator= "\n").fillna(' ')

    all_df = pd.concat([train, valid])

    #transform data to anserini json format and write to files
    json_format = generate_anserini_json_collection(all_df)
    for i, dict in enumerate(json_format):
        with open(args.output_folder+'docs{:02d}.json'.format(i), 'w', encoding='utf-8', ) as f:
            f.write(json.dumps(dict) + '\n')



if __name__ == "__main__":
    main()