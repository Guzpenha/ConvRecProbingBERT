from IPython import embed
import pandas as pd
import argparse

# A few goodread instances come as NaN after generating the dataset.
# even filtering before saving in 'make_seq_rec_data.py', and I don't
# understand why. I am filtering those 3126 instances here.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, required=True,
                        help="path with gr files")
    args = parser.parse_args()

    for data in ['train.csv', 'valid.csv', 'test.csv']:
        df = pd.read_csv(args.path+data)
        df[~df['query'].isnull()].to_csv(args.path+data, index=False)

if __name__ == "__main__":
    main()