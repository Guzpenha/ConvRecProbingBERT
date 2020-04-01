from list_wise_reformer.eval.evaluation import evaluate_models
from tqdm import tqdm

import logging
import torch
import torch.nn as nn
import torch.optim as optim

class LWRTrainer():
    def __init__(self, args, model, train_loader, val_loader, test_loader):
        self.args = args
        self.validate_epochs = args.validate_epochs
        self.num_validation_instances = args.num_validation_instances
        self.num_epochs = args.num_epochs

        self.num_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device {}".format(self.device))
        logging.info("Num GPU {}".format(self.num_gpu))

        self.model = model.to(self.device)
        if self.num_gpu > 1:
            self.model = nn.DataParallel(self.model)

        self.metrics = ['recip_rank', 'ndcg_cut_10']
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.lr)

        if self.args.loss == "cross-entropy":
            self.loss_function = nn.CrossEntropyLoss()
        self.max_grad_norm = 0.5

    def fit(self):
        logging.info("Total steps per epoch : {}".format(len(self.train_loader)))
        logging.info("Validating every {} epoch.".format(self.validate_epochs))
        tqdm_dataloader = tqdm(range(self.num_epochs))
        tqdm_dataloader.set_description('Epoch 0, train loss _, val nDCG@10 _')
        total_steps=0
        ndcg=0
        for epoch in tqdm_dataloader:
            for batch in self.train_loader:
                batch = [x.to(self.device) for x in batch]
                input, labels = batch

                logits = self.model(input)

                if self.args.loss == "cross-entropy":
                    _, max_indexes = torch.topk(logits, 1)
                    max_indexes = max_indexes.flatten()
                    loss = self.loss_function(logits, max_indexes)
                else:
                    raise Exception("Unsupported loss function")
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                tqdm_dataloader.\
                    set_description('Epoch {}, train loss {:.3f}, '
                                    'val nDCG@10 {:.3f}'.format(epoch + 1, loss, ndcg))
                total_steps+=1
            if self.validate_epochs > 0 and total_steps % self.validate_epochs == 0:
                res = self.validate(loader = self.val_loader)
                ndcg = res['ndcg_cut_10']

    def validate(self, loader):
        self.model.eval()
        all_logits = []
        all_labels = []
        for idx, batch in enumerate(loader):
            batch = [x.to(self.device) for x in batch]
            input, labels = batch

            with torch.no_grad():
                logits = self.model(input)
                for p in logits:
                    all_logits.append(p.tolist())
                for l in labels:
                    all_labels.append(l.tolist())
            if self.num_validation_instances!=0 and idx > self.num_validation_instances:
                break
        return self.evaluate(all_logits, all_labels)

    def test(self):
        logging.info("Starting evaluation on test.")
        self.num_validation_instances = 0
        res = self.validate(self.test_loader)
        for metric, v in res.items():
            logging.info("Test {} : {:4f}".format(metric, v))

    def evaluate(self, preds, labels):
        qrels = {}
        qrels['model'] = {}
        qrels['model']['preds'] = preds
        qrels['model']['labels'] = labels

        results = evaluate_models(qrels)
        agg_results = {}
        for metric in self.metrics:
            res = 0
            per_q_values = []
            for q in results['model']['eval'].keys():
                per_q_values.append(results['model']['eval'][q][metric])
                res += results['model']['eval'][q][metric]
            res /= len(results['model']['eval'].keys())
            agg_results[metric] = res

        return agg_results