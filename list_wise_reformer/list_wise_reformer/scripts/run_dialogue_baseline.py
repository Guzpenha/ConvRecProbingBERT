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
    args.run_id = str(ex.current_run._id)
    train = pd.read_csv(args.data_folder+args.task+"/train.csv", lineterminator= "\n").fillna(' ')
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv", lineterminator= "\n").fillna(' ')

    if args.ranker in ['bm25', 'ql', 'rm3']:
        model = model_classes[args.ranker](args.data_folder+
                                           args.task+"/anserini_index")
    else:
        model = model_classes[args.ranker](args)

    results = {}
    model_name = model.__class__.__name__
    logging.info("Fitting {} for {}".format(model_name, args.task))
    model.fit(train)
    del(train)
    logging.info("Predicting")
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
    parser.add_argument("--num_epochs", default=2, type=int, required=False,
                        help="Number of epochs for recommenders that do optimization.")
    parser.add_argument("--ranker", type=str, required=True,
                        help="ranker to use : "+",".join(model_classes.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output predictions")
    args = parser.parse_args()

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