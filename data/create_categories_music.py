import json
import pandas as pd
from IPython import embed

df = []
with open("meta_CDs_and_Vinyl.json","r") as f:
    for i, line in enumerate(f):
        json_line = json.loads(line)
        if "category" in json_line and "title" in json_line:
            categories = json_line["category"]
            title = json_line["title"]
            if len(title) < 50 and len(categories)>1:
                df.append([i, title, "|".join(categories[1:])])
df = pd.DataFrame(df, columns=["index", "title", "genres"])
df.to_csv("categories.csv", index=False)