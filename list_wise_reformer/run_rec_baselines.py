from IPython import embed

from list_wise_reformer.models.rec_baselines import PopularityRecommender,\
    RandomRecommender, SASRecommender
from list_wise_reformer.eval.evaluation import evaluate_models

import pandas as pd
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run baselines for ['ml25m', 'gr']")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    args = parser.parse_args()

    train = pd.read_csv(args.data_folder+args.task+"/train.csv")
    valid = pd.read_csv(args.data_folder+args.task+"/valid.csv")

    baselines = [RandomRecommender(),
                 PopularityRecommender()]

    results = {}
    for model in baselines:
        model_name = model.__class__.__name__
        logging.info("Fitting model {}".format(model_name))
        model.fit(train)
        logging.info("Predicting")
        preds = model.predict(valid, valid.columns[1:])
        results[model_name] = {}
        results[model_name]['preds'] = preds
        # only first doc is relevant -> [1, 0, 0, ..., 0]
        labels = [[1] + ([0] *  (len(valid.columns[1:])))
                            for _ in range(valid.shape[0])]
        results[model_name]['labels'] = labels

    results = evaluate_models(results)

    for model in baselines:
        model_name = model.__class__.__name__
        logging.info("Evaluating {}".format(model_name))
        for metric in ['recip_rank', 'ndcg_cut_10']:
            res = 0
            for q in results[model_name]['eval'].keys():
                res += results[model_name]['eval'][q][metric]
            res /= len(results[model_name]['eval'].keys())
            logging.info("%s: %.4f" % (metric, res))

if __name__ == "__main__":
    main()