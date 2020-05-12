from IPython import embed
import argparse
import logging
import ast
import pandas as pd
import numpy as np
import scipy.stats
from ast import literal_eval
import random

random.seed(42)
np.random.seed(42)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[ logging.StreamHandler() ]
)    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default=None, type=str, required=True,
                        help="path for folder with probes output files")
    parser.add_argument("--number_queries", default=100000, type=int, required=False,
                        help="number of total probe queries.")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="path for folder to write results")

    args = parser.parse_args()

    samples_df = []
    examples = []
    for task in ['ml25m', 'gr', 'music']:
        df_pop = pd.read_csv(args.input_folder.split("/output_data")[0]+"/recommendation/{}/popularity.csv".format(task))
        df_pop = df_pop[0:1000]
        df_pop.columns = ["item", "popularity"]
        logging.info("Sampling from {}".format(task))
        domain = {'ml25m':'movie', 'gr':'book', 'music':'music album'}[task]
        for probe in ['categories']:
            sampled_items = None
            sample_idx=None
            for sentence_type in ['type-I', 'type-II']:
                for model in ['bert-large-cased']:#['bert-base-cased', 'bert-large-cased']:
                    file_signature = "probe_type_{}_task_{}_num_queries_{}_model_{}_sentence_type_{}.csv".format(
                        probe, task, args.number_queries, model, sentence_type
                    )
                    df = pd.read_csv(args.input_folder+file_signature)
                    df["item"] = df["raw_queries"]                    
                    df = df.merge(df_pop, on=["item"])
                    if sentence_type == "type-I":
                        df["raw_queries"] = df.apply(lambda r: r["item"] + " is a _____ " + domain + ".",axis=1)
                    elif sentence_type == "type-II":
                        df["raw_queries"] = df.apply(lambda r: r["item"] + " is a " + domain + " of the genre _____.",axis=1)
                    # df = df.loc[df["raw_queries"].str.len() < 60]
                    # df = df.loc[df["labels"].str.len() < 40]
                    
                    if sample_idx is None:
                        sample_idx = random.sample(range(0, df.shape[0]), 10)

                    df["model"] = model
                    df["task"] = task
                    df["probe"] = probe
                    df["preds"] = df.apply(lambda r: r["preds"].lower().split(" ")[0:3], axis=1)
                    df["preds_scores"] = df["preds_scores"].apply(literal_eval)
                    df["preds_scores"] = df.apply(lambda r: r["preds_scores"][0:3], axis=1)
                    df["preds"] = df.apply(lambda r: ", ".join([p + " [{:.3f}]".format(s).lstrip('0') for p,s in zip(r["preds"], r["preds_scores"])]), axis=1)                    
                    if sampled_items is None:
                        sample = df.iloc[sample_idx][["task","raw_queries", "labels", "preds", "R@10", "item"]]
                        sampled_items = df.iloc[sample_idx]["item"].values
                    else:
                        sample = df[df["item"].isin(sampled_items)][["task","raw_queries", "labels", "preds", "R@10", "item"]]
                    samples_df.append(sample)

    samples_df = pd.concat(samples_df).sort_values(["task", "item"])    

    samples_df.to_csv(args.output_folder+"samples_table_mlm.csv", sep="\t", index=False)

if __name__ == "__main__":
    main()