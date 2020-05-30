from transformers import RobertaForMaskedLM, RobertaTokenizer, BertForMaskedLM, BertTokenizer
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
import fasttext

def filter_categories_df(df, bert_model, 
                        fast_text_language_detector_path='./data/lid.176.ftz'):
    replace_rules = {        
        "Children's Music": "children",
        "Children's" : "children",
        "Album-Oriented Rock (AOR)": "rock",
        " & " : "|"
    }
    language_pred_model = fasttext.load_model(fast_text_language_detector_path)
    
    tqdm.pandas()

    logging.info("Renaming categories.")
    for k, v in replace_rules.items():
        df["genres"] = df.progress_apply(lambda r,key=k,value=v: r["genres"].replace(key, value), axis=1)    

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    df["split_genres"] = df.apply(lambda r: r["genres"].lower().split("|"), axis=1)
    count = df.shape[0]
    filtered_df = df

    filtered_df = filtered_df.drop_duplicates(["title"])
    logging.info("Filtered {} duplicated titles.".
            format(count-filtered_df.shape[0]))
    count = filtered_df.shape[0]

    logging.info("Predicting item`s language.")
    filtered_df["is_english_item"] = df.\
        progress_apply(lambda r,f=language_pred_model.predict:
             f(r["title"], k=1)[0][0] == '__label__en', axis=1)
    filtered_df = filtered_df[filtered_df["is_english_item"]]
    logging.info("Filtered {} non-english items.".
            format(count-filtered_df.shape[0]))
    count = filtered_df.shape[0]

    logging.info("Checking if labels are tokens in tokenizer.")
    filtered_df["split_genres"] = filtered_df.progress_apply(lambda r,t=tokenizer:
         [l for l in r["split_genres"] if l in t.get_vocab()], axis=1)    
    filtered_df["no_valid_labels_in_vocab"] = filtered_df.\
        apply(lambda r: len(r["split_genres"])==0, axis=1)
    filtered_df = filtered_df[~ filtered_df["no_valid_labels_in_vocab"]]
    logging.info("Filtered {} with no valid labels.".
            format(count-filtered_df.shape[0]))
    count = filtered_df.shape[0]
    filtered_df["genres"] = filtered_df.apply(lambda r: "|".join(r["split_genres"]), axis=1)    
    print(filtered_df["genres"].unique())
    return filtered_df[["item", "title", "genres"]]

class MaskedLanguageModelProbe():
    def __init__(self, input_data, batch_size, bert_model, item_domain, sentence_type):
        self.seed = 42
        random.seed(self.seed)        
        torch.manual_seed(self.seed)

        self.item_domain = item_domain
        self.warmup_steps = 0
        self.data = input_data        
        self.batch_size = batch_size
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)
        self.batch_size = self.batch_size * max(1, self.n_gpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "roberta" in bert_model:
            self.model = RobertaForMaskedLM.\
                from_pretrained(bert_model)
            self.tokenizer = RobertaTokenizer.\
                from_pretrained(bert_model)
        else:
            self.model = BertForMaskedLM.\
                from_pretrained(bert_model)
            self.tokenizer = BertTokenizer.\
                from_pretrained(bert_model)

        if sentence_type == "type-I":
            self.sentences_generator = self.sentences_generator_1
        elif sentence_type == "type-II":
            self.sentences_generator = self.sentences_generator_2
        elif sentence_type == "no-item":
            self.sentences_generator = self.sentences_generator_3

        self._generate_probe_data()

    def sentences_generator_1(self, row):
        item_title = row[1]
        categories = [c.lower() for c in row[2].split("|")]
        #Example: Pulp Fiction is a [MASK] movie. label = drama and etc
        sentence = "{} is a {} {}.".format(item_title, self.tokenizer.mask_token, self.item_domain)
        return sentence, categories

    def sentences_generator_2(self, row):
        item_title = row[1]
        categories = [c.lower() for c in row[2].split("|")]
        #Example: Pulp Fiction is a [MASK] movie. label = drama and etc
        sentence = "{} is a {} of the {} genre.".format(item_title, self.item_domain, self.tokenizer.mask_token)
        return sentence, categories

    def sentences_generator_3(self, row):        
        categories = [c.lower() for c in row[2].split("|")]
        #Example: It is a [MASK] movie. label = drama and etc
        sentence = "It is a {} of the {} genre.".format(self.item_domain, self.tokenizer.mask_token)
        return sentence, categories

    def _encode_sentence(self, sentence, labels, max_length=50):
        pad_token = self.tokenizer.pad_token_id
        input_ids = self.tokenizer.encode(sentence,
                                          add_special_tokens=True,
                                          max_length=max_length)        
        attention_mask = [1] * len(input_ids)
        
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = [0] * max_length 
        
        encoded_label_tokens = [self.tokenizer.encode(t,
                add_special_tokens=False)[0] for t in labels]
        labels = []        
        for encoded_label in encoded_label_tokens:
            sequence_labels = [-100] * len(input_ids)
            mask_idx = [idx for (idx, i) in enumerate(input_ids) if i == \
                self.tokenizer.mask_token_id][0]
            sequence_labels[mask_idx] = encoded_label
            labels.append(sequence_labels)
        training_label = labels[random.choice([i for i in range(len(encoded_label_tokens))])]
        return input_ids, attention_mask, token_type_ids, \
            training_label , encoded_label_tokens + ([0] * (max_length - len(encoded_label_tokens)))

    def _generate_probe_data(self):
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []
        all_labels_training = []
        all_labels = []
        sentences_ids = []

        logging.info("Generating probe dataset.")
        for idx, row in enumerate(tqdm(self.data.itertuples(index=False), 
                                  desc="Generating probe dataset.")):
            sentence, labels = self.sentences_generator(row)
            # long sentences due to long titles "have no mask" token because of the 
            # 50 maxlen cut threshold.
            if len(self.tokenizer.encode(sentence, add_special_tokens=True)) > 40:
                continue

            input_ids, attention_masks, token_type_ids, label_training, labels = \
                    self._encode_sentence(sentence, labels)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_masks)
            all_token_type_ids.append(token_type_ids)
            all_labels_training.append(label_training)
            all_labels.append(labels)
            sentences_ids.append(idx)            

            if idx < 5:
                logging.info("Probing example %d" % idx)
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("attention_mask: %s" % " ".join([str(x) for x in attention_masks]))                
                logging.info("labels: %s" % " ".join([str(x) for x in labels if x != 0]))
                logging.info("raw: %s --> %s" % (str(sentence), str(self.tokenizer.decode([str(x) for x in labels if x != 0]))))
                logging.info("reconstructed: %s" % str(self.tokenizer.decode(input_ids)))

        self.dataset = TensorDataset(torch.tensor(all_input_ids, dtype = torch.long), 
                                     torch.tensor(all_attention_masks, dtype = torch.long),
                                     torch.tensor(all_token_type_ids, dtype = torch.long),
                                     torch.tensor(all_labels_training, dtype = torch.long),
                                     torch.tensor(all_labels, dtype = torch.long),
                                     torch.tensor(sentences_ids, dtype = torch.long))

        self.data_loader = DataLoader(self.dataset, 
            batch_size=self.batch_size, 
            pin_memory=True)

    def run_probe(self):
        all_scores = []
        all_labels = []

        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.model.eval()

        logging.info("Running BERT predictions for calculating probe results")
        results = []
        for batch_idx, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            batch = tuple(t.to(self.device) for t in batch)
            labels = batch[4]            
            inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2]}
            tokens_prediction_scores = self.model(**inputs)[0]

            for i in range(len(batch[0])):
                input_ids = batch[0][i]                
                raw_query_idx = batch[5][i].detach().cpu().numpy().tolist()
                masked_idx = (input_ids == self.tokenizer.mask_token_id).nonzero().item()
                token_predictions = tokens_prediction_scores[i, masked_idx, :]
                probs = token_predictions.softmax(dim=0)
                values, predictions = probs.topk(50)
                preds = predictions.detach().cpu().numpy()
                l = labels[i][labels[i].nonzero()][:,0].\
                    detach().cpu().numpy().tolist()

                results.append([self.tokenizer.decode(preds),
                                self.data.iloc[raw_query_idx]["genres"].replace("|", " "),
                                self.data.iloc[raw_query_idx]["title"],
                                values.detach().cpu().numpy().tolist()])
        return results

    def pre_train_using_probe(self, num_epochs):
        self.model.train()
        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.zero_grad()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=num_epochs*len(self.data_loader)
        )

        logging.info("Pre-training BERT for probe.")
        for epoch in range(num_epochs):
            logging.info("Starting epoch {}".format(epoch+1))
            for batch_idx, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2],
                            "masked_lm_labels": batch[3]}

                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        return self.model