# from rec_probing.probes.mlm_probe import *
from rec_probing.probes.nsp_probe import *

from IPython import embed
import argparse
import logging
import ast
import pandas as pd
import scipy.stats
import random
import jellyfish
from tqdm import tqdm
import numpy as np
import gensim.downloader as api

w2vec_model = api.load('word2vec-google-news-300')
# w2vec_model = api.load('glove-wiki-gigaword-300')

distances = [
    jellyfish.levenshtein_distance
]
tqdm.pandas()
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

    tasks = ['ml25m', 'gr', 'music']
    dfs = []
    logging.info("Opening files.")
    for task in tasks:
        for probe in ['recommendation', 'search']:
            for model in ['bert-base-cased', 'bert-large-cased']:                
                file_signature = "probe_type_{}_task_{}_num_candidates_{}_num_queries_{}_model_{}_technique_{}.csv".format(
                    probe, task, args.number_candidates, args.number_queries, model, args.probe_technique
                )
                df = pd.read_csv(args.input_folder+file_signature)
                df["model"] = model
                df["task"] = task
                df["probe"] = probe                
                df["labels"] = df["labels"].apply(ast.literal_eval)
                df["query_scores"] = df["query_scores"].apply(ast.literal_eval)
                df["raw_queries"] = df["raw_queries"].apply(ast.literal_eval)
                dfs.append(df)
    dfs_raw = pd.concat(dfs)
    dfs_raw["prompt_item"] = dfs_raw.apply(lambda r: r["raw_queries"][0], axis=1)
    dfs_raw["relevant_item"] = dfs_raw.apply(lambda r: r["raw_queries"][1], axis=1)
    
    logging.info("Calculating R_2@1")
    dfs_raw["R_2@1"] = dfs_raw.apply(lambda r, f=recall_at_with_cand:
            f(r["query_scores"], r["labels"], 2), axis=1)
    logging.info("Calculating R_5@1")
    dfs_raw["R_5@1"] = dfs_raw.apply(lambda r, f=recall_at_with_cand:
            f(r["query_scores"], r["labels"], 5), axis=1)    

    logging.info("Joining with year information")    
    data_path = args.input_folder.split("/output_data")[0]
    df_years_all = []
    for task in tasks:
        if task != "music":
            df_years = pd.read_csv("{}/recommendation/{}/item_years.csv".format(data_path, task))
            df_years["task"] = task
            df_years_all.append(df_years)
    df_years_all = pd.concat(df_years_all)[["title", "year", "task"]]
    df_years_all["relevant_item"] = df_years_all["title"]
    df_years_all["year"] = pd.to_numeric(df_years_all["year"], errors='coerce')
    df_years_all = df_years_all[~df_years_all["year"].isnull()]
    df_years_all = df_years_all[df_years_all["year"] >1500]
    df_years_all = df_years_all[df_years_all["year"] <2100]
    year_agg = dfs_raw[dfs_raw["model"] == "bert-large-cased"].\
            merge(df_years_all, on=["task", "relevant_item"]).\
            groupby(["probe", "task", "model", "year"])[["R_2@1", "R_5@1"]].\
            agg(["mean", "count"])    
    year_agg = year_agg.reset_index()
    year_agg.columns = ["probe", "task", "model", "year", "R_2@1", "count", "R_5@1", "count_1"]
    year_agg.reset_index().\
        to_csv(args.output_folder+"agg_years_probe_results_{}.csv".format(args.probe_technique), sep="\t")

    dfs_raw_corr = dfs_raw[dfs_raw["model"] == 'bert-large-cased']

    logging.info("Calculating standard token %")
    tokenizer = BertTokenizer.\
            from_pretrained('bert-large-cased')
    vocab = tokenizer.get_vocab()
    dfs_raw_corr["prompt_item_std_token_percentage"] = dfs_raw_corr.apply(lambda r, v=vocab: 
            sum([w in v for w in str(r["prompt_item"]).split(" ")])/len(str(r["prompt_item"]).split(" ")),axis=1)
    dfs_raw_corr["relevant_item_std_token_percentage"] = dfs_raw_corr.apply(lambda r, v=vocab: 
            sum([w in v for w in str(r["relevant_item"]).split(" ")])/len(str(r["relevant_item"]).split(" ")),axis=1)

    logging.info("Merging with in_wiki")    
    dfs_wiki = []
    for task in tasks:
        df = pd.read_csv("{}/recommendation/{}/in_wiki.csv".format(data_path, task))
        #for join purposes:
        df["prompt_item"] = df["title"]
        df["relevant_item"] = df["title"]
        df["task"] = task
        dfs_wiki.append(df)

    df_wiki_all = pd.concat(dfs_wiki).drop_duplicates(["title", "task"])
    df_wiki_all["res_wiki_search"] = df_wiki_all["res_wiki_search"].apply(ast.literal_eval)
    df_wiki_all["top_1_res_search"] = df_wiki_all.apply(lambda r: r["res_wiki_search"][0] if r["in_wiki"] else "", axis=1)    
    df_wiki_all["str_sim_to_top_1"] = df_wiki_all.progress_apply(lambda r, f=jellyfish.levenshtein_distance: f(r["top_1_res_search"], r["title"]) if r["top_1_res_search"] != "" else 10000, axis=1)
    df_wiki_all["in_wiki"] = df_wiki_all["str_sim_to_top_1"] < 40

    dfs_raw_corr = dfs_raw_corr.merge(df_wiki_all[["task","prompt_item","in_wiki"]],
                                 on=["task", "prompt_item"], how="left").fillna(False)
    dfs_raw_corr["prompt_in_wiki"] = dfs_raw_corr["in_wiki"]
    dfs_raw_corr = dfs_raw_corr.drop(columns=["in_wiki"])
    dfs_raw_corr = dfs_raw_corr.merge(df_wiki_all[["task","relevant_item","in_wiki"]],
                                 on=["task", "relevant_item"], how="left").fillna(False)
    dfs_raw_corr["relevant_item_in_wiki"] = dfs_raw_corr["in_wiki"]
    dfs_raw_corr = dfs_raw_corr.drop(columns=["in_wiki"])

    logging.info("Merging with item_popularity")
    dfs_pop = []
    for task in tasks:
        df = pd.read_csv("{}/recommendation/{}/popularity.csv".format(data_path, task))
        df.columns = ["title", "pop"]
        df["task"] = task
        dfs_pop.append(df)
    df_pop_all = pd.concat(dfs_pop)
    df_pop_all.drop_duplicates(["task", "title"])
    df_pop_all["log_pop"] = np.log(df_pop_all["pop"])
    df_pop_all["relevant_item"] = df_pop_all["title"] # for merge purposes
    dfs_raw_corr = dfs_raw_corr.merge(df_pop_all, on=["task", "relevant_item"], how="left").fillna(0.0)

    logging.info("Calculating string similarities")
    for f in distances:
        dfs_raw_corr[f.__name__] = dfs_raw_corr.progress_apply(lambda r, f=f: f(r["raw_queries"][0], r["raw_queries"][1]), axis=1)

    logging.info("Calculating WMD.")
    def preprocess(sentence):
        return [w for w in sentence.lower().split()]
    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)    
    dfs_raw_corr["word_movers_distance"] = dfs_raw_corr.progress_apply(lambda r, f=w2vec_model.wmdistance, p=preprocess:
                                            f(p(r["raw_queries"][0]), p(r["raw_queries"][1])), axis=1).replace([np.inf, -np.inf], 10)
    logger.propagate = True
    logger.setLevel(logging.INFO)

    logging.info("Calculating correlations")
    corrs = []
    col_names = []
    for f in distances:
        col_names.append("cor_string_sim_"+f.__name__)
        col_names.append("pvalue_string_sim_"+f.__name__)
    col_names = col_names + ["cor_prompt_in_wiki", "pvalue_prompt_in_wiki", 
                             "cor_relevant_in_wiki", "pvalue_relevant_in_wiki", 
                             "cor_pop", "pvalue_pop",
                             "cor_log_pop", "pvalue_log_pop",
                             "cor_prompt_item_std_token_percentage", "pvalue_prompt_item_std_token_percentage",
                             "cor_relevant_item_std_token_percentage", "pvalue_relevant_item_std_token_percentage",
                             "cor_word_movers_distance", "pvalue_word_movers_distance"]

    for probe in ['recommendation', 'search']:
        for task in tasks:
            df_filtered = dfs_raw_corr[(dfs_raw_corr["task"] == task) & (dfs_raw_corr["probe"]==probe)]
            cor_and_pvalues = []
            for f in distances:
                cor, pvalue = scipy.stats.pearsonr(df_filtered["R_5@1"], df_filtered[f.__name__])
                cor_and_pvalues.append(cor)
                cor_and_pvalues.append(pvalue)
            for wiki_col in ["prompt_in_wiki", "relevant_item_in_wiki"]:
                cor, pvalue = scipy.stats.pearsonr(df_filtered["R_5@1"], df_filtered[wiki_col])
                cor_and_pvalues.append(cor)
                cor_and_pvalues.append(pvalue)
            for pop_col in ["pop", "log_pop"]:
                cor, pvalue = scipy.stats.pearsonr(df_filtered["R_5@1"], df_filtered[pop_col])
                cor_and_pvalues.append(cor)
                cor_and_pvalues.append(pvalue)
            for token_col in ["prompt_item_std_token_percentage", "relevant_item_std_token_percentage"]:
                cor, pvalue = scipy.stats.pearsonr(df_filtered["R_5@1"], df_filtered[token_col])
                cor_and_pvalues.append(cor)
                cor_and_pvalues.append(pvalue)
            cor, pvalue = scipy.stats.pearsonr(df_filtered["R_5@1"], df_filtered['word_movers_distance'])
            cor_and_pvalues.append(cor)
            cor_and_pvalues.append(pvalue)
            corrs.append([probe, task] + cor_and_pvalues)

    df_corrs = pd.DataFrame(corrs, columns=["probe", "task"] + col_names)
    df_corrs.to_csv(args.output_folder+"corrs_probe_results_{}.csv".format(args.probe_technique), sep="\t")

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