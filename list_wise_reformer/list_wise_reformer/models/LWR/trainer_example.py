from list_wise_reformer.models.LWR.model import ListWiseReformer
from list_wise_reformer.models.LWR.trainer import LWRTrainer
from list_wise_reformer.models.LWR.dataset import LWRFineTuningDataLoader

from transformers import BertTokenizer
from IPython import embed
import pandas as pd
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Setting minimal parameters
parser = argparse.ArgumentParser()
args = parser.parse_args()

args.num_epochs=1
args.validate_epochs=1
args.num_validation_instances = 0
args.num_candidate_docs_train = 2
args.train_batch_size = 2
args.val_batch_size = 1
args.lr = 5e-5
args.loss="cross-entropy"

MAX_SEQ_LEN = 1024
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.max_len = MAX_SEQ_LEN

# Creating input data
train = pd.DataFrame([['What is the meaning of life?', '42', '41'],
                      ['What is the meaning of life?', '42', '41']],
                     columns = ['query', 'relevant_doc', 'non_relevant_1'])

valid = pd.DataFrame([['I want to listen to a good music album',
                        '2012-2017 by Against All Logic',
                       'Elephant Stone by Elephant Stone']],
                     columns = ['query', 'relevant_doc', 'non_relevant_1'])

test = pd.DataFrame([['expensive noise cancelling headphones',
                      'WH-1000XM3', 'JBL Tune 500BT']],
                    columns = ['query', 'relevant_doc', 'non_relevant_1'])

# Instantiating components for training a ListWiseReformer
model = ListWiseReformer(
    num_tokens= tokenizer.vocab_size,
    dim = 1048,
    depth = 12,
    max_seq_len = MAX_SEQ_LEN,
    num_candidate_docs=2)

dataloader = LWRFineTuningDataLoader(args=args, train_df=train,
                                     val_df=valid, test_df=test,
                                     tokenizer=tokenizer)

train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

trainer = LWRTrainer(args, model, train_loader, val_loader, test_loader)

trainer.fit()
trainer.test()
