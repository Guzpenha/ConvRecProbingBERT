from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, DataLoader
from IPython import embed
from tqdm import tqdm

import torch
import random
import logging
import functools
import operator

class TokenSimilarityProbe():
    def __init__(self, number_candidates, input_data, 
                number_queries_per_user, batch_size, 
                probe_type, bert_model, probe_technique):
        self.seed = 42
        random.seed(self.seed)        
        torch.manual_seed(self.seed)

        self.probe_technique = probe_technique
        self.warmup_steps = 0
        self.number_candidates = number_candidates
        self.data = input_data
        self.number_queries_per_user = number_queries_per_user
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)
        self.batch_size = self.batch_size * max(1, self.n_gpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BertModel.\
            from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.\
            from_pretrained(bert_model)
        
        if probe_type == "recommendation":
            self.sentences_generator = self.get_sentences_rec
        elif probe_type == "search":
            self.sentences_generator = self.get_sentences_review

        self._generate_probe_data()

    def get_sentences_review(self, row):
        review = row[0].replace("[ITEM_NAME]", "[UNK]")
        relevant_doc = row[1]
        candidate_docs = row[2:(2+self.number_candidates)]
        sentences = []
        raw_queries = []
        
        raw_query = [review, relevant_doc]
        sentence_pos = ("{}".format(relevant_doc),
                        "{}".format(review))

        sentences_neg = []
        for non_relevant_item in candidate_docs:
            raw_query.append(non_relevant_item)
            sentences_neg.append(("{}".format(non_relevant_item),
                                  "{}".format(review)))
        sentences.append((sentence_pos, sentences_neg))
        raw_queries.append(raw_query)

        return sentences, raw_queries

    def get_sentences_rec(self, row):
        user_session = row[0].split(" [SEP] ")
        relevant_doc = row[1]
        candidate_docs = row[2:(2+self.number_candidates)]
        
        random.shuffle(user_session)
        sentences = []
        raw_queries = []
        for item in user_session[0:self.number_queries_per_user]:
            raw_query = [item, relevant_doc]
            sentence_pos = ("{}".format(item),
                            "{}".format(relevant_doc))            
            sentences_neg = []
            for non_relevant_item in candidate_docs:
                raw_query.append(non_relevant_item)
                sentences_neg.append(("{}".format(item),
                                     "{}".format(non_relevant_item)))
            sentences.append((sentence_pos, sentences_neg))
            raw_queries.append(raw_query)
        return sentences, raw_queries

    def _encode_sentence(self, sentence, max_length=50):
        pad_token=0
        pad_token_segment_id=0
        input_ids = self.tokenizer.encode(sentence,
                                          add_special_tokens=True,
                                          max_length=max_length)        
        attention_mask = [1] * len(input_ids)
        
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = [0] * max_length                 

        return input_ids, attention_mask, token_type_ids

    def _generate_probe_data(self):
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []        
        self.all_raw_queries = []
        query_idx = 0

        logging.info("Generating probe dataset.")
        for idx, row in enumerate(tqdm(self.data.itertuples(index=False), 
                                  desc="Generating probe dataset.")):
            sentences, raw_queries = self.sentences_generator(row)
            for q in raw_queries:
                self.all_raw_queries += raw_queries
                query_idx+=1

            for pos, negatives in sentences:
                sentence_a, sentence_b = pos
                input_ids, attention_masks, token_type_ids = \
                    self._encode_sentence(sentence_a)
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_masks)
                all_token_type_ids.append(token_type_ids)

                input_ids, attention_masks, token_type_ids = \
                    self._encode_sentence(sentence_b)
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_masks)
                all_token_type_ids.append(token_type_ids)

                for neg in negatives:
                    sentence_a, sentence_b = neg
                    input_ids, attention_masks, token_type_ids = \
                        self._encode_sentence(sentence_a)
                    all_input_ids.append(input_ids)
                    all_attention_masks.append(attention_masks)
                    all_token_type_ids.append(token_type_ids)

                    input_ids, attention_masks, token_type_ids = \
                        self._encode_sentence(sentence_b)
                    all_input_ids.append(input_ids)
                    all_attention_masks.append(attention_masks)
                    all_token_type_ids.append(token_type_ids)

            if idx < 5:
                logging.info("Probing negative example %d" % idx)
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("attention_mask: %s" % " ".join([str(x) for x in attention_masks]))
                logging.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))                
                logging.info("raw: %s --> %s" % (str(neg[0]), str(neg[1])))
                logging.info("reconstructed: %s" % str(self.tokenizer.decode(input_ids)))

        self.dataset = TensorDataset(torch.tensor(all_input_ids, dtype = torch.long), 
                                     torch.tensor(all_attention_masks, dtype = torch.long),
                                     torch.tensor(all_token_type_ids, dtype = torch.long))

        self.data_loader = DataLoader(self.dataset, 
            batch_size=self.batch_size, 
            pin_memory=True)

    def run_probe(self):
        all_scores = []        

        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.model.eval()

        logging.info("Running BERT predictions for calculating probe results")
        for batch_idx, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            batch = tuple(t.to(self.device) for t in batch)            
            inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2]}
            
            if self.probe_technique == 'cls-sim':
                pooled_rep = self.model(**inputs)[0][:, 0]
            elif self.probe_technique == 'mean-sim':
                pooled_rep = self.model(**inputs)[0].mean(dim=1)

            sentences_count = 0
            for i in range(len(pooled_rep)):
                if i % 2 == 0:
                    all_scores.append(torch.dot(pooled_rep[i],
                                                pooled_rep[i+1]).
                                                detach().cpu().numpy().tolist())

        results = []
        query_scores, query_labels = [], []
        logging.info("Aggregating predications per query.")
        query_id = 0
        for batch_idx, score in enumerate(all_scores):
            query_scores.append(score)
            if (batch_idx+1) % (self.number_candidates + 1) == 0:
                results.append([query_scores, [1]+[0]*self.number_candidates,
                    self.all_raw_queries[query_id]])
                query_scores, query_labels = [], []
                query_id+=1        
        return results

    def pre_train_using_probe(self, num_epochs):
        # TODO
        pass        