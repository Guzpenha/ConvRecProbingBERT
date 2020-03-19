from IPython import embed
import pandas as pd
import json

def main():
    base_path = "/Users/gustavopenha/personal/recsys20/data/search/"
    for path in ["music"]: #["ml25m", "gr", "music"]:
        print(path)
        train, valid, test = pd.read_csv(base_path+path+"/train.csv"), \
                             pd.read_csv(base_path+path+"/valid.csv"), \
                             pd.read_csv(base_path+path+"/test.csv")

        all_df = pd.concat([train, valid, test])
        n_queries = len(all_df)
        print("Num reviews", n_queries)

        items = set()
        for _, r in train.iterrows():
            for item in [r[col] for col in train.columns if "non_relevant" in col]:
                items.add(item)
            items.add(r["relevant_doc"])
        for _, r in valid.iterrows():
            for item in [r[col] for col in train.columns if "non_relevant" in col]:
                items.add(item)
            items.add(r["relevant_doc"])
        for _, r in test.iterrows():
            for item in [r[col] for col in train.columns if "non_relevant" in col]:
                items.add(item)
            items.add(r["relevant_doc"])
        print("Num items", len(items))

        all_df["query_length"] = all_df.apply(lambda r: len(r["query"].split(" ")), axis=1)
        print("average # words review", all_df["query_length"].mean())
        item_lengths=0
        for item in items:
            item_lengths+= len(item.split(" "))
        print("average # words items", item_lengths/len(items))


if __name__ == '__main__':
    main()