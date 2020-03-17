import pandas as pd
import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from IPython import embed
from collections import Counter
from tqdm import tqdm

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
    # Same problem as SASRec.
    def __init__(self):
        pass
    def fit(self, sessions):
        pass
    def predict(self, sessions, doc_pred_columns):
        pass

class BPRMFRecommender(nn.Module):
    # Code based on https://github.com/AmazingDD/recommend-lib/blob/master/BPRMFRecommender.py

    def __init__(self, seed, num_user, num_item, item_map,
                 factor_num=200, lr= 0.01, wd=0.001, batch_size=4096,
                 negative_samples_train=1, epochs=2):
        super(BPRMFRecommender, self).__init__()

        torch.manual_seed(seed)

        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.num_item = num_item
        self.item_map = item_map
        self.ns = negative_samples_train
        self.epochs=epochs

        self.embed_user = nn.Embedding(num_user, factor_num)
        self.embed_item = nn.Embedding(num_item, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        pred_i = (user * item_i).sum(dim=-1)
        pred_j = (user * item_j).sum(dim=-1)

        return pred_i, pred_j

    def fit(self, sessions, gpu='0'):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        users, items_i, items_j = [], [], []
        for user_id, r in sessions.iterrows():
            for item in r['query'].split(" [SEP] "):
                for i in range(self.ns):
                    item_col = np.random.choice(len(sessions.columns[1:]), 1)
                    users.append([user_id])
                    items_i.append([self.item_map[item]])
                    items_j.append([self.item_map[r[sessions.columns[item_col+1][0]]]])

        dataset = data.TensorDataset(torch.tensor(users),
                                     torch.tensor(items_i),
                                     torch.tensor(items_j))
        data_loader = data.DataLoader(dataset, batch_size=self.batch_size)

        model = self
        if torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()

        optimizer = optim.SGD(model.parameters(), lr=self.lr,
                              weight_decay=self.wd)

        for _ in range(self.epochs):
            model.train()

            for user, item_i, item_j in tqdm(data_loader):

                if torch.cuda.is_available():
                    user = user.cuda()
                    item_i = item_i.cuda()
                    item_j = item_j.cuda()
                else:
                    user = user.cpu()
                    item_i = item_i.cpu()
                    item_j = item_j.cpu()

                model.zero_grad()
                pred_i, pred_j = model(user, item_i, item_j)
                loss = -(pred_i - pred_j).sigmoid().log().sum()
                loss.backward()
                optimizer.step()

    def predict(self, sessions, doc_pred_columns):
        users, items_i, items_j = [], [], []
        for user_id, r in sessions.iterrows():
            for column in doc_pred_columns:
                users.append([user_id])
                items_i.append([self.item_map[r[column]]])
                items_j.append([0])

        dataset = data.TensorDataset(torch.tensor(users),
                                     torch.tensor(items_i),
                                     torch.tensor(items_j))
        data_loader = data.DataLoader(dataset, batch_size=len(doc_pred_columns))

        model = self
        model.eval()
        preds = []
        for user, item_i, item_j in tqdm(data_loader):
            if torch.cuda.is_available():
                user = user.cuda()
                item_i = item_i.cuda()
                item_j = item_j.cuda()
            else:
                user = user.cpu()
                item_i = item_i.cpu()
                item_j = item_j.cpu()

            pred_i, _ = model(user,
                              item_i,
                              item_j) # ns item is not important here.
            preds.append(list(pred_i.flatten().data.cpu().numpy()))

        return preds