from list_wise_reformer.eval.evaluation import evaluate_models
from IPython import embed

import pandas as pd
import numpy as np
import scipy.stats
import argparse
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

METRICS = ['recip_rank', 'ndcg_cut_10']

pd.set_option('display.max_columns', None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_folders", default=None,
                        type=str, required=False,
                        help="the folders separated by , containing files 'config.json'"
                             " and 'predictions.csv'")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="the folder to write results")
    args = parser.parse_args()

    run_folders = ["/ssd/home/gustavo/recsys2020penha/data/output_data/mt_bert4dialogue/{}/".format(i) for i in [32,33,34,35,36,37,38,39,40,41,42,43,44,45]]
    valid_adv = pd.read_csv("/ssd/home/gustavo/recsys2020penha/data/dialogue/redial/valid_adv.csv", lineterminator= "\n").fillna(' ')
    valid = pd.read_csv("/ssd/home/gustavo/recsys2020penha/data/dialogue/redial/valid.csv", lineterminator= "\n").fillna(' ')
    path_movies = "/ssd/home/gustavo/recsys2020penha/data/movies_with_mentions.csv"
    df_movies = pd.read_csv(path_movies)
    movies = set(df_movies["movieName"].tolist()) 

    def in_movies_stupidly_slow(utterance, movies):
        in_movies=""
        for m in movies: 
            if m in utterance: 
                if len(m)> len(in_movies):
                    in_movies=m
        return in_movies
    valid_adv["is_recommendation"] = valid_adv.apply(lambda r,m=movies,f=in_movies_stupidly_slow: f(r["relevant_doc"],m),axis=1)

    valid["num_turns"] = valid.apply(lambda r: len(r["query"].split("[UTTERANCE_SEP]")),axis=1)
    valid_adv["num_turns"] = valid_adv.apply(lambda r: len(r["query"].split("[UTTERANCE_SEP]")),axis=1)    

    turns_dfs = []
    # run_folders = args.predictions_folders.split(",")
    all_res = []
    for run_folder in run_folders:
        with open(run_folder+"config.json") as f:
            config = json.load(f)['args']
            config['seed'] = str(config['seed'])
            predictions_df = pd.read_csv(run_folder+"predictions.csv")
            qrels = {}
            qrels['model'] = {}
            qrels['model']['preds'] = predictions_df.values
            # only first doc is relevant -> [1, 0, 0, ..., 0]
            labels = [[1] + ([0] * (len(predictions_df.columns[1:])))
                      for _ in range(predictions_df.shape[0])]
            qrels['model']['labels'] = labels

            results = evaluate_models(qrels)
            logging.info("Task %s" % (config['task']))
            logging.info("Model %s (%s)" % (config['ranker'], run_folder))
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
            
            df_turns = pd.DataFrame([ _ for _ in zip(list(valid["num_turns"]), per_q_values)], columns=["turns", "nDCG@10"])
            df_turns["run"] = config['seed']
            df_turns["dataset"] = "ReDial-BM25"
            turns_dfs.append(df_turns)

            all_res.append([config['task'],
                            config['ranker']+"_"+config["multi_task_for"],
                            config['seed'],
                            run_folder] + metrics_results)

            predictions_df = pd.read_csv(run_folder+"predictions_adv.csv")
            predictions_df= predictions_df.join(valid_adv)            
            print(predictions_df.shape)
            predictions_df = predictions_df[predictions_df["is_recommendation"] != ""][["prediction_{}".format(i) for i in range(51)]]
            print(predictions_df.shape)
            qrels = {}
            qrels['model'] = {}
            qrels['model']['preds'] = predictions_df.values
            # only first doc is relevant -> [1, 0, 0, ..., 0]
            labels = [[1] + ([0] * (len(predictions_df.columns[1:])))
                      for _ in range(predictions_df.shape[0])]
            qrels['model']['labels'] = labels

            results = evaluate_models(qrels)
            logging.info("Task %s" % (config['task']))
            logging.info("Model %s (%s)" % (config['ranker'], run_folder))
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

            df_turns = pd.DataFrame([ _ for _ in zip(list(valid_adv["num_turns"]), per_q_values)], columns=["turns", "nDCG@10"])
            df_turns["run"] = config['seed']
            df_turns["dataset"] = "ReDial-Adv"
            turns_dfs.append(df_turns)

            all_res.append([config['task']+"_adv",
                            config['ranker']+"_"+config["multi_task_for"],
                            config['seed'],
                            run_folder] + metrics_results)


    metrics_results = pd.DataFrame(all_res,
                                   columns=['dataset', 'model',
                                            'seed', 'run'] + metrics_cols)

    turns_dfs = pd.concat(turns_dfs)
    turns_dfs.groupby(["dataset","turns"])["nDCG@10"].agg(["mean","count"]).to_csv(args.output_folder+'ranker'+"_turns_df.csv",
                                              index=True, sep="\t")       

    #Calculate statistics of metrics
    agg_df = metrics_results.groupby(["dataset", "model"]). \
        agg(['mean', 'std', 'count', 'max']). \
        reset_index().round(4)
    col_names = ["dataset", "model"]
    for metric in METRICS:
        col_names+=[metric+"_mean", metric+"_std",
                    metric+"_count", metric+"_max"]

    agg_df.columns =  col_names
    agg_df.sort_values(metric+"_mean").to_csv(args.output_folder+'ranker'+"_aggregated_results.csv",
                                              index=False, sep="\t")    
    #run statistical tests between maximum runs
    arg_max = metrics_results. \
        sort_values(metric, ascending=False). \
        drop_duplicates(['dataset','model']). \
        reset_index()

    arg_max = arg_max.sort_values(metric, ascending=True)

    per_dataset_df = []
    # In the end the tests we want is Ours_(rec+dialogue+search) vs all baselines,
    # which is equal to N_metrics * N_baselines * N_tasks tests. When displaying the table,
    # only the last model can have the superscripts with statistical significance.
    n_tests = len(METRICS) * (len(arg_max["model"].unique()) -1) * \
              len(arg_max["dataset"].unique())
    logging.info("n_tests : {}".format(n_tests))

    for dataset in arg_max["dataset"].unique():
        seen_models = []
        filtered_df = arg_max[arg_max["dataset"] == dataset]
        for metric in METRICS:
            filtered_df[metric+"_pvalues"] = ""
            filtered_df[metric+"_statistical_tests"] = ""
        for idx, r in filtered_df.iterrows():
            model_print_idx=0
            for model_idx in seen_models:
                model_print_idx+=1
                for metric in METRICS:
                    baseline_values = filtered_df.loc[model_idx, metric+"_per_query"]
                    current_model_values = filtered_df.loc[idx, metric+"_per_query"]
                    statistic, pvalue = scipy.stats.ttest_rel(baseline_values,
                                                              current_model_values)
                    filtered_df.loc[idx, metric+"_pvalues"] = filtered_df.loc[idx, metric+"_pvalues"] + \
                                                              "," + str(pvalue)
                    if pvalue <= (0.05/n_tests):
                        filtered_df.loc[idx, metric+"_statistical_tests"] = \
                            filtered_df.loc[idx, metric+"_statistical_tests"]+ \
                            str(model_print_idx)
            seen_models.append(idx)
        per_dataset_df.append(filtered_df)
    arg_max = pd.concat(per_dataset_df)
    arg_max = arg_max[[c for c in arg_max.columns if "per_query" not in c]]
    arg_max.to_csv(args.output_folder+'ranker'+"_max_results.csv",
                   index=False, sep="\t")
    embed()
if __name__ == "__main__":
    main()