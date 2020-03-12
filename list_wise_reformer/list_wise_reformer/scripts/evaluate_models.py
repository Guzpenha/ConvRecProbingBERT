from list_wise_reformer.eval.evaluation import evaluate_models
from IPython import embed

import pandas as pd
import numpy as np
import scipy.stats
import argparse
import logging
import json

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

METRICS = ['recip_rank', 'ndcg_cut_10']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_folders", default=None,
                        type=str, required=True,
                        help="the folders separated by , containing files 'config.json'"
                             " and 'predictions.csv'")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="the folder to write results")
    args = parser.parse_args()

    run_folders = args.predictions_folders.split(",")
    all_res = []
    for run_folder in run_folders:
        with open(run_folder+"config.json") as f:
            config = json.load(f)['args']
            predictions_df = pd.read_csv(run_folder+"predictions.csv")
            qrels = {}
            qrels['model'] = {}
            qrels['model']['preds'] = predictions_df.values
            # only first doc is relevant -> [1, 0, 0, ..., 0]
            labels = [[1] + ([0] * (len(predictions_df.columns[1:])))
                      for _ in range(predictions_df.shape[0])]
            qrels['model']['labels'] = labels

            results = evaluate_models(qrels)
            logging.info("Model %s" % config['recommender'])
            logging.info("Seed %s" % config['seed'])

            metrics_results = []
            metrics_cols = []
            for metric in METRICS:
                res = 0
                per_q_values = []
                for q in results['model']['eval'].keys():
                    per_q_values.append(results['model']['eval'][q][metric])
                    res += results['model']['eval'][q][metric]
                res /= len(results['model']['eval'].keys())
                metrics_results+= [res, per_q_values]
                metrics_cols+= [metric, metric+'_per_query']
                logging.info("%s: %.4f" % (metric, res))

            all_res.append([config['task'],
                            config['recommender'],
                            config['seed'],
                            run_folder] + metrics_results)


    metrics_results = pd.DataFrame(all_res,
                                   columns=['dataset', 'model',
                                            'seed', 'run'] + metrics_cols)

    agg_df = metrics_results.groupby(["model", "dataset"]). \
        agg(['mean', 'std', 'count', 'max']). \
        reset_index().round(4)
    col_names = ["model", "dataset"]
    for metric in METRICS:
        col_names+=[metric+"_mean", metric+"_std",
                    metric+"_count", metric+"_max"]
    agg_df.columns =  col_names
    agg_df.to_csv(args.output_folder+"aggregated_results.csv", index=False)

if __name__ == "__main__":
    main()