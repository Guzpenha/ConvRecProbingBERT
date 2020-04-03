from tqdm import tqdm
from abc import *

import torch
import torch.utils.data as data
import logging
import random
import os
import pickle

#Inspired by BERTREC-VAE-Pytorch
class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, train_df, val_df, test_df, tokenizer):
        self.args = args
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.num_gpu = torch.cuda.device_count()
        self.actual_train_batch_size = self.args.train_batch_size \
                                       * max(1, self.num_gpu)
        logging.info("Train instances per batch {}".
                     format(self.actual_train_batch_size))

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass

class LWRFineTuningDataLoader(AbstractDataloader):
    def __init__(self, args, train_df, val_df, test_df, tokenizer):
        super().__init__(args, train_df, val_df, test_df, tokenizer)

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = LWRFineTuningDataset(self.args, self.train_df,
                                    self.args.num_candidate_docs_train,
                                    self.tokenizer,'train')
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.actual_train_batch_size,
                                     shuffle=True)
        return dataloader

    def _get_val_loader(self):
        num_docs = len(self.val_df.columns) - 1 #the 'query' column
        dataset = LWRFineTuningDataset(self.args, self.val_df, num_docs,
                                       self.tokenizer, 'val')
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.args.val_batch_size,
                                     shuffle=False)
        return dataloader

    def _get_test_loader(self):
        num_docs = len(self.test_df.columns) - 1  # the 'query' column
        dataset = LWRFineTuningDataset(self.args, self.test_df, num_docs,
                                       self.tokenizer, 'test')
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.args.val_batch_size,
                                     shuffle=False)
        return dataloader

class LWRFineTuningDataset(data.Dataset):
    def __init__(self, args, data, num_candidate_docs, tokenizer, data_partition):
        random.seed(42)

        self.args = args
        self.data = data
        self.num_candidate_docs = num_candidate_docs
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        self.instances = []

        self._cache_instances()

    def _cache_instances(self):
        signature = "set_{}_docs_train_{}_seq_max_l_{}_sample_{}".\
            format(self.data_partition,
                   self.args.num_candidate_docs_train,
                   self.args.max_seq_len,
                   self.args.sample_data)
        path = self.args.data_folder + self.args.task + signature

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info("Loading instances from {}".format(path))
                self.instances = pickle.load(f)
        else:
            logging.info("Generating instances with signature {}".format(signature))
            # Input will look like this
            # [CLS] query [SEP] doc_1 [SEP] doc_2 ... [SEP] doc_n [PAD]
            for _, row in tqdm(self.data.iterrows()):
                labels = [1] + ([0] * (self.num_candidate_docs - 1))
                docs = [row['relevant_doc']] + \
                       [row["non_relevant_" + str(c + 1)]
                        for c in range((self.num_candidate_docs - 1))]
                docs_and_labels = [_ for _ in zip(docs, labels)]
                random.shuffle(docs_and_labels)

                input = str(row['query'])
                for doc, _ in docs_and_labels:
                    input += " " + self.tokenizer.sep_token + " " + doc

                tokenized_input = self._tokenize_input(input)[0]
                correct_order_labels = [t[1] for t in docs_and_labels]
                self.instances.append((torch.LongTensor(tokenized_input),
                                      torch.LongTensor(correct_order_labels)))
                with open(path, 'wb') as f:
                    pickle.dump(self.instances, f)

    def _tokenize_input(self, input, pad_to_max_length=True):
        return self.tokenizer.encode(input, add_special_tokens=True,
                                   max_length=self.tokenizer.max_len,
                                   pad_to_max_length=pad_to_max_length,
                                   return_tensors='pt')

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]