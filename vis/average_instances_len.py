from IPython import embed
import pandas as pd
REPO_DIR = "/Users/gustavopenha/personal/recsys20/"

def main():
    for path in [REPO_DIR+"data/dialogue/books",
                 REPO_DIR+"data/dialogue/movies",
                 REPO_DIR+"data/dialogue/music",
                 REPO_DIR+"data/recommendation/music",
                REPO_DIR+"data/recommendation/ml25m",
                REPO_DIR+"data/recommendation/gr"]:
        print(path)
        train, valid, test = pd.read_csv(path+"/train.csv", lineterminator='\n'), \
                             pd.read_csv(path+"/valid.csv", lineterminator='\n'), \
                             pd.read_csv(path+"/test.csv", lineterminator='\n')
        train, valid, test = train[~train['query'].isnull()], \
                             valid[~valid['query'].isnull()], \
                             test[~test['query'].isnull()]

        train["concat"] = train.apply(lambda r, cols=train.columns:
                                      ' '.join(str(r[c]) for c in cols), axis=1)
        train["concat_len"] = train.apply(lambda r: len(r['concat'].split(" ")), axis=1)
        print("Train average instance length", train['concat_len'].mean())
        print("Train max instance length", train['concat_len'].max())
        print("Train median instance length", train['concat_len'].median())

if __name__ == '__main__':
    main()