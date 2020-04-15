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

#Software Architecture inspired by BERTREC-VAE-Pytorch repo
class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, train_df, val_df, test_df, tokenizer):
        self.args = args
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        special_tokens_dict = {
            'additional_special_tokens': ['[UTTERANCE_SEP]','[ITEM_SEP]']
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
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
        dataset = LWRFineTuningDataset(self.args, self.val_df,
                                       self.args.num_candidate_docs_eval,
                                       self.tokenizer, 'val', self.item_map)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.args.val_batch_size,
                                     shuffle=False)
        return dataloader

    def _get_test_loader(self):
        dataset = LWRFineTuningDataset(self.args, self.test_df,
                                       self.args.num_candidate_docs_eval,
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
        signature = "set_{}_n_eval_docs_{}_n_train_docs{}_seq_max_l_{}_sample_{}_rep_{}".\
            format(self.data_partition,
                   self.num_candidate_docs,
                   self.args.num_candidate_docs_train,
                   self.args.max_seq_len,
                   self.args.sample_data,
                   self.args.input_representation)
        path = self.args.data_folder + self.args.task + signature
        path_tokenizer = self.args.data_folder + self.args.task + signature + "_tokenizer"

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info("Loading instances from {}".format(path))
                self.instances = pickle.load(f)
            if self.args.input_representation == 'ids':
                self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        else:
            logging.info("Generating instances with signature {}".format(signature))
            if self.args.input_representation == 'ids':
                self.data, self.tokenizer = toItemIDFormat(self.data, self.item_map, self.tokenizer)

            labels = [1] + ([0] * (self.num_candidate_docs-1))
            for idx, row in enumerate(tqdm(self.data.itertuples(index=False))):
                docs = row[1:(self.num_candidate_docs)+1]

                docs_and_labels = [_ for _ in zip(docs, labels)]
                #randomize docs order so that rel is not always on first position during training
                if self.data_partition == 'train':
                    random.shuffle(docs_and_labels)
                correct_order_labels = [t[1] for t in docs_and_labels]

                # Input will look like this : [CLS] query [SEP] doc_1 [SEP] doc_2 ... [SEP] doc_n [PAD]
                # the sep in the query must be different than the doc sep
                if self.args.input_representation == 'ids':
                    # the items are tokens e.g. item_21131 , so no need to use separators
                    # This is not a 100% since we are using bert tokenizer which use sub-words as well.
                    q_str = str(row[0].replace("[SEP]", ""))
                else:
                    # the items are the titles of the items, e.g. "Stranger Things"
                    q_str = str(row[0].replace("[SEP]", "[ITEM_SEP]"))

                # If we are training with less candidate documents than predicting we need to
                # generate one instance per candidate document and during test time aggregate
                # the results per query. Otherwise we only have one instance containing all
                # candidate documents
                explode_instances = self.data_partition != "train" and \
                        self.args.num_candidate_docs_eval != self.args.num_candidate_docs_train
                documents=[]
                if explode_instances:
                    for doc, _  in docs_and_labels:
                        documents.append(doc)
                else:
                    doc_str = (" " + self.tokenizer.sep_token + " "). \
                        join([t[0] for t in docs_and_labels])
                    documents.append(doc_str)

                for doc_idx, doc_str in enumerate(documents):
                    # Ideally we should cut only first from left to right, this is
                    # an improvement we can implement over encode_plus, which prob.
                    # cuts from right to left.
                    tokenized_input = self.tokenizer.encode_plus(q_str, doc_str,
                                                                 add_special_tokens=True,
                                                                 max_length=self.tokenizer.max_len,
                                                                 only_first=True)["input_ids"]

                    padding_length = self.tokenizer.max_len - len(tokenized_input)
                    tokenized_input = tokenized_input + ([self.tokenizer.pad_token_id] * padding_length)

                    if explode_instances:
                        ordered_l_tensor = torch.LongTensor([correct_order_labels[doc_idx]])
                    else:
                        ordered_l_tensor = torch.LongTensor(correct_order_labels)

                    self.instances.append((torch.LongTensor(tokenized_input),
                                           ordered_l_tensor))
                if idx < 5:
                    logging.info("Set {} Instance {} query string\n\n{}\n".format(self.data_partition, idx, q_str))
                    logging.info("Set {} Instance {} doc string ({} candidates)\n\n{}\n".
                                 format(self.data_partition, idx,self.num_candidate_docs, doc_str))
                    logging.info("Set {} Instance {} tokenized input \n\n{}\n".format(self.data_partition, idx, tokenized_input))
                    logging.info("Set {} Instance {} reconstructed input \n\n{}\n".format(self.data_partition, idx,
                        self.tokenizer.convert_ids_to_tokens(tokenized_input)))
            with open(path, 'wb') as f:
                pickle.dump(self.instances, f)
            if self.args.input_representation == 'ids':
                os.makedirs(path_tokenizer)
                self.tokenizer.save_pretrained(path_tokenizer)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

class LWRRecommenderPretrainingDataLoader(AbstractDataloader):
    def __init__(self, args, train_df, val_df, test_df, tokenizer):
        super().__init__(args, train_df, val_df, test_df, tokenizer)
        logging.info("Using {} pre-training objective.".
                     format(self.args.pre_training_objective))
        self.item_map = {}

    def get_pytorch_dataloaders(self):
        train_loader = self._get_loader(self.train_df)
        val_loader = self._get_loader(self.val_df)
        test_loader = self._get_loader(self.test_df)
        return train_loader, val_loader, test_loader

    def _get_loader(self, df):
        dataset = LWRRecommenderPretrainingDataset(self.args, df, self.tokenizer, self.item_map)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.actual_train_batch_size,
                                     shuffle=True)
        return dataloader

class LWRRecommenderPretrainingDataset(data.Dataset):
    def __init__(self, args, data, tokenizer, item_map):
        random.seed(42)
        self.args = args
        self.tokenizer = tokenizer
        self.data = data
        self.item_map = item_map
        self.idx = 0


        self.pre_training_objective = self.args.pre_training_objective
        self.objectives = {
            'session': self.session,
            'session_w_noise': self.session,
            'shuffle_session': self.shuffle_session,
            'shuffle_session_w_noise': self.shuffle_session
        }

        assert self.pre_training_objective in self.objectives.keys()
        self.generate_item = self.objectives[self.pre_training_objective]

        if self.args.input_representation == 'ids':
            self.data, self.tokenizer = toItemIDFormat(self.data, self.item_map, self.tokenizer)
            item_special_tokens = {
                'additional_special_tokens': list(self.item_map.values())
            }
            self.tokenizer.add_special_tokens(item_special_tokens)

        # do some preprocessing of the input data
        self.data['session'] = self.data. \
            apply(lambda r: r['query'].split(" [SEP] ") + [r['relevant_doc']], axis=1)
        self.sessions = self.data['session'].to_numpy()
        self.candidate_lists = self.data[self.data.columns[2:(self.args.num_candidate_docs_train) + 1]].to_numpy()

        self.noise = False
        if self.pre_training_objective == 'session_w_noise' or\
            self.pre_training_objective == 'shuffle_session_w_noise':
            self.noise = True

    def session(self, index):
        #shuffle the session by items.
        query = self.sessions[index]

        # 15% of items are removed, inspired by BERT
        if self.noise:
            idx_to_keep = random.sample([i for i in range(len(query))],
                                    k=int((len(query) * 0.85)))
            query = [query[i] for i in sorted(idx_to_keep)]

        # select a random candidate list
        candidates = random.choice(self.candidate_lists)

        #the relevant document is the original one (last in the query)
        candidates_list = [query[-1]] + candidates.tolist()
        query = query[0:-1] #remove relevant from candidate
        labels = [1] + ([0] * (self.args.num_candidate_docs_train-1))

        #shuffle candidates
        aux = list(zip(candidates_list, labels))
        random.shuffle(aux)
        candidates_list, labels = zip(*aux)

        join_token = " [ITEM_SEP] "
        if self.args.input_representation == 'ids':
            join_token = " "
        q_str = join_token.join(query)
        doc_str = " {} ".format(self.tokenizer.sep_token).join(candidates_list)

        tokenized_input = self.tokenizer.encode_plus(q_str, doc_str,
                                                     add_special_tokens=True,
                                                     max_length=self.tokenizer.max_len,
                                                     only_first=True)["input_ids"]
        padding_length = self.tokenizer.max_len - len(tokenized_input)
        tokenized_input = tokenized_input + ([self.tokenizer.pad_token_id] * padding_length)

        if self.idx < 5:
            logging.info("Instance {} original\n\n{}\n".format(self.idx, self.data.iloc[[index]].values[0][0:-1]))
            logging.info("Instance {} query string\n\n{}\n".format(self.idx, q_str))
            logging.info("Instance {} doc string ({} candidates)\n\n{}\n".
                         format(self.idx, self.args.num_candidate_docs_train, doc_str))
            logging.info(
                "Instance {} tokenized input \n\n{}\n".format(self.idx, tokenized_input))
            logging.info("Instance {} reconstructed input \n\n{}\n".format(self.idx,
                                                                          self.tokenizer.convert_ids_to_tokens(
                                                                              tokenized_input)))
            logging.info("Instance {} labels \n\n{}\n".format(self.idx, labels))
        return (torch.LongTensor(tokenized_input), torch.LongTensor(labels))

    def shuffle_session(self, index):
        #shuffle the session by items.
        query = self.sessions[index]

        if self.noise:
            # 15% inspired by BERT
            query = random.sample(query, int((len(query) * 0.85)))
        else:
            random.shuffle(query)

        #select a random candidate list
        candidates = random.choice(self.candidate_lists)

        #the relevant document is a random item from the session
        candidates_list = [query[-1]] + candidates.tolist()
        query = query[0:-1] #remove relevant from candidate
        labels = [1] + ([0] * (self.args.num_candidate_docs_train-1))

        #shuffle candidates
        aux = list(zip(candidates_list, labels))
        random.shuffle(aux)
        candidates_list, labels = zip(*aux)

        join_token = " [ITEM_SEP] "
        if self.args.input_representation == 'ids':
            join_token = " "
        q_str = join_token.join(query)
        doc_str = " {} ".format(self.tokenizer.sep_token).join(candidates_list)

        tokenized_input = self.tokenizer.encode_plus(q_str, doc_str,
                                                     add_special_tokens=True,
                                                     max_length=self.tokenizer.max_len,
                                                     only_first=True)["input_ids"]
        padding_length = self.tokenizer.max_len - len(tokenized_input)
        tokenized_input = tokenized_input + ([self.tokenizer.pad_token_id] * padding_length)

        if self.idx < 5:
            logging.info("Instance {} original\n\n{}\n".format(self.idx, self.data.iloc[[index]].values[0][0:-1]))
            logging.info("Instance {} query string\n\n{}\n".format(self.idx, q_str))
            logging.info("Instance {} doc string ({} candidates)\n\n{}\n".
                         format(self.idx, self.args.num_candidate_docs_train, doc_str))
            logging.info(
                "Instance {} tokenized input \n\n{}\n".format(self.idx, tokenized_input))
            logging.info("Instance {} reconstructed input \n\n{}\n".format(self.idx,
                                                                          self.tokenizer.convert_ids_to_tokens(
                                                                              tokenized_input)))
            logging.info("Instance {} labels \n\n{}\n".format(self.idx, labels))
        return (torch.LongTensor(tokenized_input), torch.LongTensor(labels))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        self.idx+=1
        return self.generate_item(index)