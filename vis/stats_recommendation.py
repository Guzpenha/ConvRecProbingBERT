from IPython import embed
import pandas as pd

def main():
    base_path = "/Users/gustavopenha/personal/recsys20/data/recommendation/"
    for path in ["music"]: #["ml25m", "gr", "music"]:
        print(path)
        train, valid, test = pd.read_csv(base_path+path+"/train.csv"), \
                             pd.read_csv(base_path+path+"/valid.csv"), \
                             pd.read_csv(base_path+path+"/test.csv")
        train, valid, test = train[~train['query'].isnull()], \
                             valid[~valid['query'].isnull()], \
                             test[~test['query'].isnull()]

        num_users = len(train)
        print("Num users", num_users)

        items = set()
        actions_count = 0
        lengths = 0
        for _, r in train.iterrows():
            l = 0
            for item in r["query"].split(" [SEP] "):
                items.add(item)
                actions_count+=1
                l+=1
            l+=3 # relevant for training, valid and test items
            items.add(r["relevant_doc"])
            actions_count += 1
            lengths+=l
        for _, r in valid.iterrows():
            items.add(r["relevant_doc"])
            actions_count += 1
        for _, r in test.iterrows():
            items.add(r["relevant_doc"])
            actions_count += 1
        print("Num items", len(items))
        print("Num actions", actions_count)
        print("Average length", lengths/num_users)
        print("Density", actions_count/(num_users*len(items)))
if __name__ == '__main__':
    main()