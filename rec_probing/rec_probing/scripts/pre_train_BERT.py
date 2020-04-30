from rec_probing.probes.nsp_probe import *
from IPython import embed

import os
import argparse
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[ logging.StreamHandler() ]
)    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run probes for ['ml25m', 'gr', 'music']")
    parser.add_argument("--probe_type", default=None, type=str, required=True,
                        help="the probe to be used ['recommendation', 'search']")
    parser.add_argument("--input_folder", default=None, type=str, required=True,
                        help="path for folder with /<task>/train.csv file")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="path for folder to write results")
    parser.add_argument("--number_candidates", default=1, type=int, required=False,
                        help="number of candidates for the nsp probes.")
    parser.add_argument("--number_queries", default=10000, type=int, required=False,
                        help="number of total probe queries.")
    parser.add_argument("--batch_size", default=1, type=int, required=False,
                        help="batch_size")
    parser.add_argument("--num_epochs", default=1, type=int, required=False,
                        help="num_epochs")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str, required=False,
                        help="bert model name ['bert-base-cased' or 'bert-large-cased']")

    args = parser.parse_args()

    path = "{}/{}/train.csv".format(args.input_folder, args.task)
    df = pd.read_csv(path, lineterminator="\n", nrows=args.number_queries)

    probe = NextSentencePredictionProbe(number_candidates = args.number_candidates, 
                                            input_data = df,
                                            number_queries_per_user=1,
                                            batch_size = args.batch_size,
                                            probe_type = args.probe_type,
                                            bert_model = args.bert_model)
    results = probe.run_probe()
    results_df = pd.DataFrame(results,\
         columns = ["query_scores", "labels", "raw_queries"])
    results_df["relevant>non_relevant_1"] = results_df.\
        apply(lambda r: r['query_scores'][0]> r['query_scores'][1], axis=1)
    logging.info("Percentage correct before pre-training: %f" % (100 * results_df["relevant>non_relevant_1"].sum()/results_df.shape[0]))

    pre_trained_model = probe.pre_train_using_probe(args.num_epochs)

    results = probe.run_probe()
    results_df = pd.DataFrame(results,\
         columns = ["query_scores", "labels", "raw_queries"])
    results_df["relevant>non_relevant_1"] = results_df.\
        apply(lambda r: r['query_scores'][0]> r['query_scores'][1], axis=1)
    logging.info("Percentage correct after pre-training: %f" % (100 * results_df["relevant>non_relevant_1"].sum()/results_df.shape[0]))

    model_signature = "pre_trained_on_probe_type_{}_task_{}_num_candidates_{}_num_queries_{}_model_{}".\
        format(args.probe_type, args.task, args.number_candidates, args.number_queries, args.bert_model)

    if not os.path.exists(args.output_folder+model_signature):
        os.makedirs(args.output_folder+model_signature)
    pre_trained_model.save_pretrained(args.output_folder+model_signature)

if __name__ == "__main__":
    main()