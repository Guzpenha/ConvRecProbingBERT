from IPython import embed
import argparse
import logging
import ast
import numpy as np
import pandas as pd
import scipy.stats
import jellyfish
from tqdm import tqdm
from transformers import BertTokenizer
import gensim.downloader as api

# w2vec_model = api.load('word2vec-google-news-300')
w2vec_model = api.load('glove-wiki-gigaword-300')

tqdm.pandas()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[ logging.StreamHandler() ]
)    

final_order = [ 
    (   'gr', 'R@1'),
    (   'gr', 'R@5'),
    ('ml25m', 'R@1'),
    ('ml25m', 'R@5'),
    ('music', 'R@1'),
    ('music', 'R@5')]

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

    tasks = ['ml25m', 'gr', 'music']    
    # sentence_types = ['no-item', 'type-I', 'type-II']
    sentence_types = ['type-I', 'type-II']
    models = ['bert-base-cased', 'bert-large-cased']
    dfs = []
    logging.info("Reading files")
    for task in tasks:
        df_pop = pd.read_csv(args.input_folder.split("/output_data")[0]+"/recommendation/{}/popularity.csv".format(task))
        df_pop = df_pop[0:int(len(df_pop)*0.01)]
        df_pop.columns = ["item", "popularity"]
        for probe in ['categories']:
            for sentence_type in sentence_types:
                for model in models:
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
                    df["preds_scores"] = df["preds_scores"].apply(ast.literal_eval)
                    dfs.append(df)
    dfs_raw = pd.concat(dfs)
    logging.info("Calculating R@5 and R@1")
    dfs_raw["preds"] = dfs_raw.apply(lambda r: r["preds"].lower(), axis=1)
    dfs_raw["labels"] = dfs_raw.apply(lambda r: r["labels"].lower(), axis=1)
    dfs_raw["intersection_5"] = dfs_raw.apply(lambda r: set(r["preds"].split(" ")[0:5]).intersection(set(r["labels"].split(" "))),axis=1)
    dfs_raw["R@5"] = dfs_raw.apply(lambda r: len(r["intersection_5"])/len(r["labels"].split(" ")),axis=1)
    dfs_raw["intersection_1"] = dfs_raw.apply(lambda r: set(r["preds"].split(" ")[0:1]).intersection(set(r["labels"].split(" "))),axis=1)
    dfs_raw["R@1"] = dfs_raw.apply(lambda r: len(r["intersection_1"])/len(r["labels"].split(" ")),axis=1)

    calculate_corr=True
    if calculate_corr:
        dfs_raw_corr = dfs_raw[dfs_raw["model"] == 'bert-large-cased']

        logging.info("Joining with year information")    
        data_path = args.input_folder.split("/output_data")[0]
        df_years_all = []
        for task in tasks:
            if task != "music":
                df_years = pd.read_csv("{}/recommendation/{}/item_years.csv".format(data_path, task))
                df_years["task"] = task
                df_years_all.append(df_years)
        df_years_all = pd.concat(df_years_all)[["title", "year", "task"]]
        df_years_all["item"] = df_years_all["title"]
        df_years_all["year"] = pd.to_numeric(df_years_all["year"], errors='coerce')
        df_years_all = df_years_all[~df_years_all["year"].isnull()]
        df_years_all = df_years_all[df_years_all["year"] >1500]
        df_years_all = df_years_all[df_years_all["year"] <2100]
        year_agg = dfs_raw[dfs_raw["model"] == "bert-large-cased"].\
                merge(df_years_all, on=["task", "item"]).\
                groupby(["probe", "sentence_type", "task", "model", "year"])[["R@1", "R@5"]].\
                agg(["mean", "count"])    
        year_agg = year_agg.reset_index()
        year_agg.columns = ["probe", "sentence_type", "task", "model", "year", "R@1", "count", "R@5", "count_1"]
        year_agg.reset_index().\
            to_csv(args.output_folder+"_agg_years_probe_results_mlm.csv", sep="\t")

        logging.info("Calculating WMD.")
        def preprocess(sentence):
            return [w for w in sentence.lower().split()]
        logger = logging.getLogger()
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)    
        dfs_raw_corr["word_movers_distance"] = dfs_raw_corr.progress_apply(lambda r, f=w2vec_model.wmdistance, p=preprocess:
                                                f(p(r["item"]), p(r["labels"])), axis=1).replace([np.inf, -np.inf], 10)
        logger.propagate = True
        logger.setLevel(logging.INFO)

        logging.info("Merging with in_wiki")
        data_path = args.input_folder.split("/output_data")[0]
        dfs_wiki = []
        for task in tasks:
            df = pd.read_csv("{}/recommendation/{}/in_wiki.csv".format(data_path, task))
            #for join purposes:
            df["item"] = df["title"]        
            df["task"] = task
            dfs_wiki.append(df)
        df_wiki_all = pd.concat(dfs_wiki).drop_duplicates(["title", "task"])
        df_wiki_all["res_wiki_search"] = df_wiki_all.apply(lambda r: "[]" if r["res_wiki_search"] == "exception" else r["res_wiki_search"],axis=1)
        df_wiki_all["res_wiki_search"] = df_wiki_all["res_wiki_search"].apply(ast.literal_eval)
        df_wiki_all["top_1_res_search"] = df_wiki_all.apply(lambda r: r["res_wiki_search"][0] if r["in_wiki"] else "", axis=1)    
        df_wiki_all["str_sim_to_top_1"] = df_wiki_all.progress_apply(lambda r, f=jellyfish.levenshtein_distance: f(r["top_1_res_search"], r["title"]) if r["top_1_res_search"] != "" else 10000, axis=1)
        df_wiki_all["in_wiki"] = df_wiki_all["str_sim_to_top_1"] < 40
        df_wiki_all["wiki_page_length"] = df_wiki_all.apply(lambda r: 0 if not r["in_wiki"] else r["wiki_page_length"],axis=1)

        dfs_raw_corr = dfs_raw_corr.merge(df_wiki_all[["task","item","in_wiki", "wiki_page_length"]],
                                    on=["task", "item"], how="left").fillna(False)
        dfs_raw_corr["item_in_wiki"] = dfs_raw_corr["in_wiki"]
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
        df_pop_all["item"] = df_pop_all["title"] # for merge purposes
        dfs_raw_corr = dfs_raw_corr.merge(df_pop_all, on=["task", "item"], how="left").fillna(0.0)


        logging.info("Calculating standard token %")
        tokenizer = BertTokenizer.\
                from_pretrained('bert-large-cased')
        vocab = tokenizer.get_vocab()
        dfs_raw_corr["title_std_token_percentage"] = dfs_raw_corr.apply(lambda r, v=vocab: 
                sum([w in v for w in str(r["title"]).split(" ")])/len(str(r["title"]).split(" ")),axis=1)

        dfs_raw_corr["BERT_score_avg_10"] = dfs_raw_corr.apply(lambda r: (sum(r["preds_scores"][0:10])/10.0), axis=1)
        dfs_raw_corr["BERT_score_avg_5"] = dfs_raw_corr.apply(lambda r: (sum(r["preds_scores"][0:5])/5.0), axis=1)
        logging.info("Calculating correlations")
        corrs = []
        col_names =  [ "cor_item_in_wiki", "pvalue_item_in_wiki",
                        "cor_wiki_page_length", "pvalue_wiki_page_length",
                        "cor_pop", "pvalue_pop",
                        "cor_log_pop", "pvalue_log_pop",
                        "cor_BERT_score_avg_5", "pvalue_BERT_score_avg_5",
                        "cor_BERT_score_avg_10", "pvalue_BERT_score_avg_10",
                        "cor_title_std_token_percentage", "pvalue_title_std_token_percentage",
                        "cor_word_movers_distance", "pvalue_word_movers_distance"]

        for sentence_type in sentence_types:
            for task in tasks:
                df_filtered = dfs_raw_corr[(dfs_raw_corr["task"] == task) & (dfs_raw_corr["sentence_type"] == sentence_type)]
                cor_and_pvalues = []
                for wiki_col in ["item_in_wiki", "wiki_page_length"]:
                    cor, pvalue = scipy.stats.pearsonr(df_filtered["R@1"], df_filtered[wiki_col])
                    cor_and_pvalues.append(cor)
                    cor_and_pvalues.append(pvalue)
                for pop_col in ["pop", "log_pop"]:
                    cor, pvalue = scipy.stats.pearsonr(df_filtered["R@1"], df_filtered[pop_col])
                    cor_and_pvalues.append(cor)
                    cor_and_pvalues.append(pvalue)
                for bert_score_col in ["BERT_score_avg_5", "BERT_score_avg_10"]:
                    cor, pvalue = scipy.stats.pearsonr(df_filtered["R@1"], df_filtered[bert_score_col])
                    cor_and_pvalues.append(cor)
                    cor_and_pvalues.append(pvalue)
                cor, pvalue = scipy.stats.pearsonr(df_filtered["R@1"], df_filtered["title_std_token_percentage"])
                cor_and_pvalues.append(cor)
                cor_and_pvalues.append(pvalue)
                cor, pvalue = scipy.stats.pearsonr(df_filtered["R@1"], df_filtered["word_movers_distance"])
                cor_and_pvalues.append(cor)
                cor_and_pvalues.append(pvalue)
                corrs.append([task, sentence_type] + cor_and_pvalues)

        df_corrs = pd.DataFrame(corrs, columns=["task", "sentence_type"] + col_names)
        df_corrs.to_csv(args.output_folder+"corrs_probe_results_mlm.csv", sep="\t")

    logging.info("Calculating statistical tests.")
    df_lists = dfs_raw.groupby(["task", "model","sentence_type"])[["R@1", "R@5"]].\
        agg(list).reset_index()
    statistical_tests_all = []
    for sentence_type in sentence_types:
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
    agg_df = dfs_raw.groupby(["task", "model", "sentence_type"])[["R@1", "R@5"]].\
        agg("mean").unstack(0)
    agg_df.columns = agg_df.columns.swaplevel(0, 1)
    agg_df[final_order].\
        to_csv(args.output_folder+"aggregate_probe_results_mlm_popular_only_{}.csv".format(args.popular_only), sep="\t")

if __name__ == "__main__":
    main()