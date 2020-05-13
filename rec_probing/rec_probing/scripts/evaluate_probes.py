# from rec_probing.probes.mlm_probe import *
from rec_probing.probes.nsp_probe import *

from IPython import embed
import argparse
import logging
import ast
import pandas as pd
import scipy.stats
import random

random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[ logging.StreamHandler() ]
)    

def recall_at_with_cand(preds, labels, candidates, at=1):
    preds = preds[0:candidates]
    labels = labels[0:candidates]
    zipped = [ _ for _ in zip(preds, labels)]
    random.shuffle(zipped)
    preds, labels  = zip(*sorted(zipped, reverse=True))
    return sum(labels[:at])/sum(labels)

final_order = [ 
    ('recommendation',    'gr', 'R_2@1'),
    ('recommendation',    'gr', 'R_5@1'),
    ('recommendation', 'ml25m', 'R_2@1'),
    ('recommendation', 'ml25m', 'R_5@1'),
    ('recommendation', 'music', 'R_2@1'),
    ('recommendation', 'music', 'R_5@1'),
    (        'search',    'gr', 'R_2@1'),
    (        'search',    'gr', 'R_5@1'),
    (        'search', 'ml25m', 'R_2@1'),
    (        'search', 'ml25m', 'R_5@1'),
    (        'search', 'music', 'R_2@1'),
    (        'search', 'music', 'R_5@1')
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default=None, type=str, required=True,
                        help="path for folder with probes output files")
    parser.add_argument("--number_candidates", default=5, type=int, required=False,
                        help="number of candidates for the nsp probes.")
    parser.add_argument("--number_queries", default=100000, type=int, required=False,
                        help="number of total probe queries.")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="path for folder to write results")    
    parser.add_argument("--probe_technique", default='nsp', type=str, required=False,
                        help="Which technique to use when probing.")
    
    args = parser.parse_args()

    dfs = []
    for task in ['ml25m', 'gr', 'music']:        
        for probe in ['recommendation', 'search']:
            for model in ['bert-base-cased', 'bert-large-cased']:                
                # file_signature = "probe_type_{}_task_{}_num_candidates_{}_num_queries_{}_model_{}_technique_{}.csv".format(
                #     probe, task, args.number_candidates, args.number_queries, model, args.probe_technique
                # )
                file_signature = "probe_type_{}_task_{}_num_candidates_{}_num_queries_{}_model_{}.csv".format(
                    probe, task, args.number_candidates, args.number_queries, model
                )
                df = pd.read_csv(args.input_folder+file_signature)
                df["model"] = model
                df["task"] = task
                df["probe"] = probe                
                df["labels"] = df["labels"].apply(ast.literal_eval)
                df["query_scores"] = df["query_scores"].apply(ast.literal_eval)
                # df["equal_scores"] = df.apply(lambda r: r["query_scores"][0] == r["query_scores"][1], axis=1)
                # logging.info("Percentage of equal 0 and 1 idx scores: {}".format(sum(df["equal_scores"])/df.shape[0]))
                dfs.append(df)
    dfs_raw = pd.concat(dfs)
    logging.info("Calculating R_2@1")
    dfs_raw["R_2@1"] = dfs_raw.apply(lambda r, f=recall_at_with_cand:
            f(r["query_scores"], r["labels"], 2), axis=1)
    logging.info("Calculating R_5@1")
    dfs_raw["R_5@1"] = dfs_raw.apply(lambda r, f=recall_at_with_cand:
            f(r["query_scores"], r["labels"], 5), axis=1)

    logging.info("Calculating statistical tests.")
    df_lists = dfs_raw.groupby(["probe", "task", "model"])[["R_2@1", "R_5@1"]].\
        agg(list).reset_index()
    statistical_tests = []
    for probe, task, metric in final_order:
        bert_base = df_lists[(df_lists["probe"]==probe) &
                        (df_lists["task"]==task) & 
                        (df_lists["model"]=="bert-base-cased")][metric].values[0]
        bert_large = df_lists[(df_lists["probe"]==probe) &
                        (df_lists["task"]==task) & 
                        (df_lists["model"]=="bert-large-cased")][metric].values[0]
        statistic, pvalue = scipy.stats.ttest_rel(bert_base, bert_large)
        statistical_tests.append(pvalue<0.01)
    df_tests = pd.DataFrame([statistical_tests], columns=final_order)
    df_tests.to_csv(args.output_folder+"statistical_tests_probe_results_{}.csv".format(args.probe_technique), sep="\t")

    logging.info("Aggregating and pivoting results table")
    agg_df = dfs_raw.groupby(["probe", "task", "model"])[["R_2@1", "R_5@1"]].\
        agg("mean").unstack(0).unstack(0)
    agg_df.columns = agg_df.columns.swaplevel(0, 2)
    agg_df.columns = agg_df.columns.swaplevel(0, 1)
    agg_df[final_order].\
        to_csv(args.output_folder+"aggregate_probe_results_{}.csv".format(args.probe_technique), sep="\t")

if __name__ == "__main__":
    main()