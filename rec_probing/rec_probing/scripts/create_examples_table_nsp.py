from IPython import embed
import argparse
import logging
import ast
import pandas as pd
import numpy as np
import scipy.stats
from ast import literal_eval
import random
from scipy.special import softmax

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
    parser.add_argument("--number_candidates", default=5, type=int, required=False,
                        help="number of candidates for the nsp probes.")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="path for folder to write results")    
    parser.add_argument("--popular_only", default=False, type=bool, required=False,
                        help="whether to use only popular items in eval")

    args = parser.parse_args()

    samples_df = []
    examples = []
    for task in ['ml25m', 'gr', 'music']:
        df_pop = pd.read_csv(args.input_folder.split("/output_data")[0]+"/recommendation/{}/popularity.csv".format(task))
        df_pop = df_pop[0:1000]
        df_pop.columns = ["item", "popularity"]
        logging.info("Sampling from {}".format(task))        
        for probe in ['search', 'recommendation']:
            sampled_items = None
            sample_idx=None            
            for model in ['bert-large-cased']:#['bert-base-cased', 'bert-large-cased']:
                file_signature = "probe_type_{}_task_{}_num_candidates_{}_num_queries_{}_model_{}.csv".format(
                    probe, task, args.number_candidates, args.number_queries, model
                )
                df = pd.read_csv(args.input_folder+file_signature)
                df["raw_queries"] = df["raw_queries"].apply(literal_eval)
                df["item"] = df.apply(lambda r: r["raw_queries"][1], axis=1)
                df = df.merge(df_pop, on=["item"])
                df["query_scores"] = df["query_scores"].apply(literal_eval)
                df["labels"] = df["labels"].apply(literal_eval)
                df["task"] = task
                if probe == "search":
                    df["joined_queries"] = df.apply(lambda r: r["raw_queries"][1] + " || "+
                            " ".join(r["raw_queries"][0].split(" ")[0:10]) + " [...]" + "\n" +
                            r["raw_queries"][2] + " || "+
                            " ".join(r["raw_queries"][0].split(" ")[0:10]) + " [...]", axis=1)
                elif probe == "recommendation":
                    df["joined_queries"] = df.apply(lambda r: "If you liked "+ r["raw_queries"][0] +
                            ", you will also like " + r["raw_queries"][1] + "\n" +
                            "If you liked "+ r["raw_queries"][0] +
                            ", you will also like " + r["raw_queries"][2], axis=1)                
                df["sent_len"] = df["joined_queries"].str.len()
                df["model"] = model
                df["probe"] = probe                                
                df["labels"] = df.apply(lambda r: r["labels"][0:2], axis=1)
                df["query_scores"] = df.apply(lambda r: r["query_scores"][0:2], axis=1)
                df["query_scores"] = df.apply(lambda r: [1, 0] if r["query_scores"][0] > r["query_scores"][1] else [0, 1], axis=1)
                df["query_scores"] = df.apply(lambda r: "["+", ".join(["{}".format(s) for s in r["query_scores"]]) + "]", axis=1)                
                df = df.sort_values("sent_len")[0:100]
                sample = df.sample(2)[["probe", "task", "joined_queries", "labels", "query_scores", "relevant>non_relevant_1", "item", "raw_queries"]]
                samples_df.append(sample)

    samples_df = pd.concat(samples_df).sort_values(["probe", "task"])

    samples_df.to_csv(args.output_folder+"samples_table_nsp.csv", sep="\t", index=False)

if __name__ == "__main__":
    main()