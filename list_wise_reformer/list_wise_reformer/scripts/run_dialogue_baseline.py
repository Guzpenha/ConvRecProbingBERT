# Due to the java error this doesnt run on HPC \/
# from list_wise_reformer.models.dialogue_baselines import BM25, RM3, QL
from list_wise_reformer.models.rec_baselines import RandomRecommender
from list_wise_reformer.models.BERTRanker import BERTRanker

from list_wise_reformer.eval.evaluation import evaluate_models
import pandas as pd
import argparse
import logging

from sacred import Experiment
from sacred.observers import FileStorageObserver

from IPython import embed

ex = Experiment('Response ranking system experiment.')

model_classes = {
    'random': RandomRecommender,
    # 'bm25': BM25,
    # 'rm3': RM3,
    # 'ql': QL,
    'bert': BERTRanker
}

@ex.main
def run_experiment(args):
    if "pre_trained" in args.ranker:
        args.ranker = "bert"

    args.run_id = str(ex.current_run._id)
    if args.multi_task_for == "":
        train_file="/train.csv"
    else:
        train_file = "/train_mt_{}.csv".format(args.multi_task_for)
    train = pd.read_csv(args.data_folder+args.task+train_file, 
        lineterminator= "\n").fillna(' ')
    if args.nrows != -1:
        valid = pd.read_csv(args.data_folder+args.task+"/valid.csv", 
            lineterminator= "\n", nrows=args.nrows).fillna(' ')
    else:
        valid = pd.read_csv(args.data_folder+args.task+"/valid.csv", 
            lineterminator= "\n").fillna(' ')

    if args.ranker in ['bm25', 'ql', 'rm3']:
        model = model_classes[args.ranker](args.data_folder+
                                           args.task+"/anserini_index")
    else:
        model = model_classes[args.ranker](args, ex)

    model_name = model.__class__.__name__
    logging.info("Fitting {} for {}".format(model_name, args.task))
    if args.ranker == 'bert':
        model.fit(train, valid)
    else:
        model.fit(train)

    del(train)

    if args.task == "redial":
        logging.info("Predicting for adversarial")
        results = {}
        valid = pd.read_csv(args.data_folder+args.task+"/valid_adv.csv", 
            lineterminator= "\n").fillna(' ')
        preds = model.predict(valid, valid.columns[1:], training_eval=False, adv_eval=True)

        results[model_name] = {}
        results[model_name]['preds'] = preds
        # only first doc is relevant -> [1, 0, 0, ..., 0]
        labels = [[1] + ([0] *  (len(valid.columns[1:])))
                            for _ in range(valid.shape[0])]
        results[model_name]['labels'] = labels

        #Saving predictions to a file
        preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(len(preds[0]))])
        preds_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions_adv.csv", index=False)

        results = evaluate_models(results)

        model_name = model.__class__.__name__
        logging.info("Evaluating {} for adv ".format(model_name))
        for metric in ['recip_rank', 'ndcg_cut_10']:
            res = 0
            for q in results[model_name]['eval'].keys():
                res += results[model_name]['eval'][q][metric]
            res /= len(results[model_name]['eval'].keys())
            logging.info("%s: %.4f" % (metric, res))

    logging.info("Predicting")
    results = {}
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv", 
            lineterminator= "\n").fillna(' ')
    preds = model.predict(valid, valid.columns[1:])

    results[model_name] = {}
    results[model_name]['preds'] = preds
    # only first doc is relevant -> [1, 0, 0, ..., 0]
    labels = [[1] + ([0] *  (len(valid.columns[1:])))
                        for _ in range(valid.shape[0])]
    results[model_name]['labels'] = labels

    #Saving predictions to a file
    preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(len(preds[0]))])
    preds_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions.csv", index=False)

    results = evaluate_models(results)

    model_name = model.__class__.__name__
    logging.info("Evaluating {}".format(model_name))
    for metric in ['recip_rank', 'ndcg_cut_10']:
        res = 0
        for q in results[model_name]['eval'].keys():
            res += results[model_name]['eval'][q][metric]
        res /= len(results[model_name]['eval'].keys())
        logging.info("%s: %.4f" % (metric, res))

    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run baselines for ['music', 'books', 'movies']")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--seed", default=42, type=int, required=False,
                        help="random seed")
    parser.add_argument("--batch_size", default=5, type=int, required=False,
                        help="batch_size")
    parser.add_argument("--bert_model", default='bert-base-cased', type=str, required=False,
                        help="default is bert-base-cased.")
    parser.add_argument("--nrows", default=-1, type=int, required=False,
                        help="-1 if all rows, otherwise the number of training and test instances to use.")
    parser.add_argument("--num_epochs", default=5, type=int, required=False,
                        help="Number of epochs for models that do optimization.")                
    parser.add_argument("--learning_rate", default=5e-5, type=float, required=False,
                        help="Learning rate")
    parser.add_argument("--logging_steps", default=100, type=int, required=False,
                        help="Number of steps to log evaluation metric.")
    parser.add_argument("--early_stopping_steps", default=-1, type=int, required=False,
                        help="Whether to stop early (<number of steps>) or not (<-1>).")
    parser.add_argument("--ranker", type=str, required=True,
                        help="ranker to use : "+",".join(model_classes.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output predictions")
    parser.add_argument("--multi_task_for", default="", type=str, required=False, 
                        help="whether to use file with multiple train_instances or not. Empty str if using train, otherwise train_<multi_task_for>.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    #Store pretrained name on config.
    if "pre_trained_on" in args.bert_model:
        args.ranker = 'pre_trained_on' + args.bert_model.split("pre_trained_on")[-1]
    ex.observers.append(FileStorageObserver(args.output_dir))
    ex.add_config({'args': args})
    return ex.run()

if __name__ == "__main__":
    main()