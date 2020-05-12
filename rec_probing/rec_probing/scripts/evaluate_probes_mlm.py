from IPython import embed
import argparse
import logging
import ast
import pandas as pd
import scipy.stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[ logging.StreamHandler() ]
)    

final_order = [ 
    (   'gr', 'R@10'),
    (   'gr', 'R@50'),
    ('ml25m', 'R@10'),
    ('ml25m', 'R@50'),
    ('music', 'R@10'),
    ('music', 'R@50')]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default=None, type=str, required=True,
                        help="path for folder with probes output files")
    parser.add_argument("--number_queries", default=100000, type=int, required=False,
                        help="number of total probe queries.")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="path for folder to write results")    
    parser.add_argument("--popular_only", default=False, type=bool, required=False,
                        help="whether to use only popular items in eval")

    args = parser.parse_args()

    dfs = []
    for task in ['ml25m', 'gr', 'music']:
        df_pop = pd.read_csv(args.input_folder.split("/output_data")[0]+"/recommendation/{}/popularity.csv".format(task))
        df_pop = df_pop[0:1000]
        df_pop.columns = ["item", "popularity"]
        for probe in ['categories']:
            for sentence_type in ['type-I', 'type-II']:
                for model in ['bert-base-cased', 'bert-large-cased']:
                    file_signature = "probe_type_{}_task_{}_num_queries_{}_model_{}_sentence_type_{}.csv".format(
                        probe, task, args.number_queries, model, sentence_type
                    )
                    df = pd.read_csv(args.input_folder+file_signature)
                    df["item"] = df["raw_queries"]
                    if args.popular_only:
                        df = df.merge(df_pop, on=["item"])
                    df["model"] = model
                    df["task"] = task
                    df["probe"] = probe
                    df["sentence_type"] = sentence_type
                    dfs.append(df)
    dfs_raw = pd.concat(dfs)    
    logging.info("Calculating statistical tests.")
    df_lists = dfs_raw.groupby(["task", "model","sentence_type"])[["R@10", "R@50"]].\
        agg(list).reset_index()
    statistical_tests_all = []
    for sentence_type in ['type-I', 'type-II']:
        statistical_tests = []
        aux_df = df_lists[df_lists["sentence_type"] == sentence_type]
        for task, metric in final_order:
            bert_base = aux_df[(aux_df["task"]==task) & 
                            (aux_df["model"]=="bert-base-cased")][metric].values[0]
            bert_large = aux_df[(aux_df["task"]==task) & 
                            (aux_df["model"]=="bert-large-cased")][metric].values[0]
            statistic, pvalue = scipy.stats.ttest_rel(bert_base, bert_large)
            statistical_tests.append(pvalue<0.01)
        statistical_tests_all.append(statistical_tests)

    df_tests = pd.DataFrame(statistical_tests_all, columns=final_order)
    df_tests.to_csv(args.output_folder+"statistical_tests_probe_results_mlm_popular_only_{}.csv".format(args.popular_only), sep="\t")

    logging.info("Aggregating and pivoting results table")
    agg_df = dfs_raw.groupby(["task", "model", "sentence_type"])[["R@10", "R@50"]].\
        agg("mean").unstack(0)
    agg_df.columns = agg_df.columns.swaplevel(0, 1)
    agg_df[final_order].\
        to_csv(args.output_folder+"aggregate_probe_results_mlm_popular_only_{}.csv".format(args.popular_only), sep="\t")

if __name__ == "__main__":
    main()