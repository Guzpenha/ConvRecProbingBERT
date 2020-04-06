from list_wise_reformer.models.utils import toItemIDFormat
from transformers import BertTokenizer, PreTrainedTokenizer
from IPython import embed
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
        if args.max_gpu != -1:
            self.num_gpu = args.max_gpu
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
        self.item_map = {}

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = LWRFineTuningDataset(self.args, self.train_df,
                                    self.args.num_candidate_docs_train,
                                    self.tokenizer,'train', self.item_map)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.actual_train_batch_size,
                                     shuffle=True)
        return dataloader

    def _get_val_loader(self):
        num_docs = len(self.val_df.columns) - 1 #the 'query' column
        dataset = LWRFineTuningDataset(self.args, self.val_df, num_docs,
                                       self.tokenizer, 'val', self.item_map)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.args.val_batch_size,
                                     shuffle=False)
        return dataloader

    def _get_test_loader(self):
        num_docs = len(self.test_df.columns) - 1  # the 'query' column
        dataset = LWRFineTuningDataset(self.args, self.test_df, num_docs,
                                       self.tokenizer, 'test', self.item_map)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.args.val_batch_size,
                                     shuffle=False)
        self.tokenizer = dataset.tokenizer
        return dataloader

class LWRFineTuningDataset(data.Dataset):
    def __init__(self, args, data, num_candidate_docs, tokenizer, data_partition, item_map):
        random.seed(42)

        self.args = args
        self.data = data
        self.num_candidate_docs = num_candidate_docs
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        self.instances = []
        self.item_map = item_map

        self._cache_instances()

    def _cache_instances(self):
        signature = "set_{}_n_cand_docs_{}_seq_max_l_{}_sample_{}_rep_{}".\
            format(self.data_partition,
                   self.num_candidate_docs,
                   self.args.max_seq_len,
                   self.args.sample_data,
                   self.args.input_representation)
        path = self.args.data_folder + self.args.task + signature
        path_tokenizer = self.args.data_folder + self.args.task + signature + "_tokenizer"

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info("Loading instances from {}".format(path))
                self.instances = pickle.load(f)
            if self.args.input_representation == 'item_ids':
                self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        else:
            logging.info("Generating instances with signature {}".format(signature))
            if self.args.input_representation == 'item_ids':
                self.data, self.tokenizer = toItemIDFormat(self.data, self.item_map, self.tokenizer)

            labels = [1] + ([0] * (self.num_candidate_docs-1))
            for row in tqdm(self.data.itertuples(index=False)):
                docs = row[1:(self.num_candidate_docs)+1]

                #randomize docs order so that rel is not always on first position
                docs_and_labels = [_ for _ in zip(docs, labels)]
                random.shuffle(docs_and_labels)
                correct_order_labels = [t[1] for t in docs_and_labels]

                # Input will look like this : [CLS] query [SEP] doc_1 [SEP] doc_2 ... [SEP] doc_n [PAD]
                # the sep in the query must be different than the doc sep
                if self.args.input_representation == 'item_ids':
                    # the items are tokens e.g. item_21131 , so no need to use separators
                    # This is not a 100% since we are using bert tokenizer which use sub-words as well.
                    q_str = str(row[0].replace("[SEP]", ""))
                else:
                    # the items are the titles of the items, e.g. "Stranger Things"
                    q_str = str(row[0].replace("[SEP]", "[ITEM_SEP]"))
                doc_str = (" " + self.tokenizer.sep_token + " "). \
                    join([t[0] for t in docs_and_labels])

                # Ideally we should cut only first from left to right, this is
                # an improvement we can implement over encode_plus, which prob.
                # cuts from right to left.
                tokenized_input = self.tokenizer.encode_plus(q_str, doc_str,
                                                             add_special_tokens=True,
                                                             max_length=self.tokenizer.max_len,
                                                             only_first=True)["input_ids"]

                padding_length = self.tokenizer.max_len - len(tokenized_input)
                tokenized_input = tokenized_input + ([self.tokenizer.pad_token_id] * padding_length)

                self.instances.append((torch.LongTensor(tokenized_input),
                                       torch.LongTensor(correct_order_labels)))

            with open(path, 'wb') as f:
                pickle.dump(self.instances, f)
            if self.args.input_representation == 'item_ids':
                os.makedirs(path_tokenizer)
                self.tokenizer.save_pretrained(path_tokenizer)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]