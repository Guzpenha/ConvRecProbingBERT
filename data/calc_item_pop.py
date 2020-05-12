import pandas as pd
from IPython import embed
from collections import Counter
import logging
from tqdm import tqdm
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[ logging.StreamHandler() ]
)
for task in ["gr", "ml25m", "music"]:
    counter = Counter()
    logging.info("Calculating popularity for {}".format(task))
    for fold in ["train", "test", "valid"]:
        df = pd.read_csv("./recommendation/{}/{}.csv".format(task, fold))
        for row in tqdm(df.itertuples()):        
            session_items = row.query.split(" [SEP] ") + [row.relevant_doc]
            for item in session_items:
                counter[item]+=1
    pop_df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    pop_df = pop_df.sort_values(pop_df.columns[1], ascending=False)
    pop_df.to_csv("./recommendation/{}/popularity.csv".format(task), index=False)