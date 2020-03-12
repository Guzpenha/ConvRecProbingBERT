from list_wise_reformer.models.utils import *
from IPython import embed
import random
import pandas as pd
from collections import Counter

class PopularityRecommender():
    """
    This recommender model ranks by popularity of the items:
    thus the predictions are independent of the user.
    """
    def __init__(self, seed=42):
        self.interaction_counts = Counter()

    def fit(self, sessions):
        for _, r in sessions.iterrows():
            for item in r['query'].split(" [SEP] "):
                self.interaction_counts[item] +=1

    def predict(self, sessions, doc_pred_columns):
        preds = []
        for idx, r in sessions.iterrows():
            user_preds = []
            for column in doc_pred_columns:
                user_preds.append(self.interaction_counts[r[column]])
            preds.append(user_preds)

        return preds

class RandomRecommender():
    # Prediction is a uniform between 0 and 1
    def __init__(self, seed=42):
        random.seed(seed)
        pass

    def fit(self, sessions):
        pass

    def predict(self, sessions, doc_pred_columns):
        preds = []
        for _, _ in sessions.iterrows():
            user_preds = []
            for _ in doc_pred_columns:
                user_preds.append(random.uniform(0,1))
            preds.append(user_preds)
        return preds

class SASRecommender():
    """
     SASRec uses python version 2 and TensorFlow 1.12, so I opted to
     create a different env and use the authors code. I implemented code
     to transform dataset to their format (create_sasrec_data.py)

     So I just run a different script (run_sasrec_local.sh or run_SASRec.sbatch)
     to save to a file and then get the results from this file.
     """
    def __init__(self):
        pass

    def fit(self, sessions):
        pass

    def predict(self, sessions, doc_pred_columns):
        pass

class BERT4Rec():
    # TODO: IMPLEMENT
    def __init__(self):
        pass

def test_popularity_recommender():
    train_data = [["Lord of the Rings: The Two Towers The (2002) [SEP] "+
                  "Back to the Future Part II (1989) [SEP] "+
                  "Gattaca (1997)"],
                  ["Lord of the Rings: The Two Towers The (2002) [SEP] " +
                   "Gattaca (1997)"],
                  ["Lord of the Rings: The Two Towers The (2002)"]]
    test_data = [["Forrest Gump (1994) [SEP] " +
                  "Silence of the Lambs, The (1991) [SEP] " +
                  "Back to the Future (1985) [SEP] " +
                  "Toy Story (1995)",
                  "Lord of the Rings: The Two Towers The (2002)",
                  "Back to the Future Part II (1989)",
                  "Gattaca (1997)"]]
    train_data = pd.DataFrame(train_data, columns=["query"])
    test_data = pd.DataFrame(test_data, columns=["query",
                                                 "candidate_doc_0",
                                                 "candidate_doc_1",
                                                 "candidate_doc_2"])

    pop = PopularityRecommender()
    pop.fit(train_data)
    predictions = pop.predict(test_data, test_data.columns[1:])
    assert predictions == [[3,1,2]]