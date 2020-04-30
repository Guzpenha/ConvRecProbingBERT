import torch
import random
import pickle
from tqdm import tqdm
import numpy as np
import logging
import os
from scipy.special import softmax
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features\
    as convert_examples_to_features

from list_wise_reformer.models.utils import \
    ConversationResponseRankingProcessor as CRRProcessor

from IPython import embed

class BERTRanker():

    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.model = BertForSequenceClassification.from_pretrained('bert-large-cased')

        self.batch_size = 5
        self.max_seq_length = 300
        self.num_train_epochs = 1
        self.learning_rate = 5e-5
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.gradient_accumulation_steps=1
        self.max_grad_norm=1.0
        self.logging_steps=50
        self.seed = args.seed
        self.args = args

        self.processor = CRRProcessor()
        self.n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, sessions):
        examples = self.processor.\
            get_examples_from_sessions(sessions,
                                       2)        
        if os.path.exists(self.args.data_folder+self.args.task+"/train_examples_bert.pk"):
            logging.info("Loading instances from file.")
            f = open(self.args.data_folder+self.args.task+"/train_examples_bert.pk", "rb")
            features = pickle.load(f)
        else:
            logging.info("Generating train instances")
            features = convert_examples_to_features(examples,
                                                self.tokenizer,
                                                label_list=self.processor.get_labels(),
                                                max_length=self.max_seq_length,
                                                output_mode="classification",
                                                pad_on_left=False,
                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=0)
            f = open(self.args.data_folder+self.args.task+"/train_examples_bert.pk", "wb")
            pickle.dump(features, f)
                
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        self.model.to(self.device)
        train_batch_size = self.batch_size * max(1, self.n_gpu)
        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=train_batch_size)

        t_total = len(train_dataloader) // self.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
        )

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        global_step = 0
        self.model.zero_grad()
        train_iterator = range(int(self.num_train_epochs))

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "token_type_ids": batch[2],
                          "labels": batch[3]}

                outputs = self.model(**inputs)
                loss = outputs[0] 

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

    def predict(self, sessions, prediction_cols):        
        examples = self.processor.\
            get_examples_from_sessions(sessions,
                                       len(prediction_cols))
        if os.path.exists(self.args.data_folder+self.args.task+"/test_examples_bert.pk"):
            logging.info("Loading instances from file.")
            f = open(self.args.data_folder+self.args.task+"/test_examples_bert.pk", "rb")
            features = pickle.load(f)
        else:
            logging.info("Generating test instances")
            features = convert_examples_to_features(examples,
                                                self.tokenizer,
                                                label_list=self.processor.get_labels(),
                                                max_length=self.max_seq_length,
                                                output_mode="classification",
                                                pad_on_left=False,
                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=0)
            f = open(self.args.data_folder+self.args.task+"/test_examples_bert.pk", "wb")
            pickle.dump(features, f)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        eval_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        #One batch = one query.
        eval_batch_size = len(prediction_cols)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)        

        logging.info("***** Running evaluation *****")
        nb_eval_steps = 0
        preds = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
                outputs = self.model(**inputs)
                _, logits = outputs[:2]
                
            nb_eval_steps += 1
            instance_predictions = softmax(logits.detach().cpu().numpy())
            preds.append(list(instance_predictions[:, 1]))

        return preds