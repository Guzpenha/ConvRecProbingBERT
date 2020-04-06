from list_wise_reformer.models.LWR.model import ListWiseReformer
from list_wise_reformer.models.LWR.trainer import LWRTrainer
from list_wise_reformer.models.LWR.dataset import LWRFineTuningDataLoader
from list_wise_reformer.models.LWR.loss import custom_losses

from transformers import BertTokenizer
from sacred.observers import FileStorageObserver
from sacred import Experiment
from IPython import embed

import torch
import pandas as pd
import argparse
import logging

ex = Experiment('Listwise Reformer experiment')

@ex.main
def run_experiment(args):
    args.run_id = str(ex.current_run._id)
    train = pd.read_csv(args.data_folder+args.task+"/train.csv")
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv")

    if args.sample_data !=-1:
        train = train[0:args.sample_data]
        valid = valid[0:args.sample_data]

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer.add_tokens(['[UTTERANCE_SEP]', '[ITEM_SEP]'])
    tokenizer.max_len = args.max_seq_len

    dataloader = LWRFineTuningDataLoader(args=args, train_df=train,
                                         val_df=valid, test_df=valid,
                                         tokenizer=tokenizer)
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

    model = ListWiseReformer(num_tokens=len(dataloader.tokenizer),
                            dim = args.hidden_dim, depth = args.depth,
                            max_seq_len = args.max_seq_len,
                            num_doc_predictions=args.num_candidate_docs_train,
                            seed=args.seed, heads=args.num_heads)
    if args.load_model != "":
        logging.info("Loading weights from {}".format(args.load_model))
        model.load_state_dict(torch.load(args.load_model))

    trainer = LWRTrainer(args, model, train_loader, val_loader, test_loader)

    model_name = model.__class__.__name__
    logging.info("Fitting {} for {}{}".format(model_name, args.data_folder, args.task))
    trainer.fit()
    logging.info("Predicting")
    preds = trainer.test()

    #Saving predictions to a file
    preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(len(preds[0]))])
    preds_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions.csv", index=False)

    #Saving model to a file
    if args.save_model:
        torch.save(model.state_dict(), args.output_dir+"/"+args.run_id+"/model")

    return trainer.best_ndcg

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run listwise reformer for")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output predictions")
    parser.add_argument("--load_model", default="", type=str, required=False,
                        help="Path with model weights to load before training.")
    parser.add_argument("--save_model", default=False, type=str, required=False,
                        help="Save trained model at the end of training.")

    #Training procedure
    parser.add_argument("--seed", default=42, type=str, required=True,
                        help="random seed")
    parser.add_argument("--num_epochs", default=100, type=int, required=False,
                        help="Number of epochs for training.")
    parser.add_argument("--max_gpu", default=-1, type=int, required=False,
                        help="max gpu used")
    parser.add_argument("--validate_epochs", default=2, type=int, required=False,
                        help="Run validation every <validate_epochs> epochs.")
    parser.add_argument("--num_validation_instances", default=-1, type=int, required=False,
                        help="Run validation for a sample of <num_validation_instances>. To run on all instances use -1.")
    parser.add_argument("--train_batch_size", default=32, type=int, required=False,
                        help="Training batch size.")
    parser.add_argument("--val_batch_size", default=32, type=int, required=False,
                        help="Validation and test batch size.")
    parser.add_argument("--num_candidate_docs_train", default=51, type=int, required=False,
                        help="Number of documents to use during training")
    parser.add_argument("--sample_data", default=-1, type=int, required=False,
                         help="Amount of data to sample for training and eval. If no sampling required use -1.")
    parser.add_argument("--input_representation", default="text", type=str, required=False,
                        help="Represent the input as 'text' or 'item_ids' (available only for rec)")

    #Model hyperparameters
    parser.add_argument("--num_heads", default=2, type=int, required=False,
                        help="Number of attention heads.")
    parser.add_argument("--lr", default=5e-5, type=float, required=False,
                        help="Learning rate.")
    parser.add_argument("--max_seq_len", default=1024, type=int, required=False,
                        help="Maximum sequence length for the inputs.")
    parser.add_argument("--hidden_dim", default=256, type=int, required=False,
                        help="Hidden dimension size.")
    parser.add_argument("--depth", default=2, type=int, required=False,
                        help="Depth of reformer.")
    parser.add_argument("--loss", default="cross-entropy", type=str, required=False,
                        help="Loss function to use [cross-entropy, "+",".join(custom_losses.keys())+"].")

    args = parser.parse_args()
    args.sacred_ex = ex

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    ex.observers.append(FileStorageObserver(args.output_dir))
    ex.add_config({'args': args})
    return ex.run()

if __name__ == "__main__":
    main()