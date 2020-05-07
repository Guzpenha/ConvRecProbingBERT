from rec_probing.probes.mlm_probe import *

from IPython import embed
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
    parser.add_argument("--input_folder", default=None, type=str, required=True,
                        help="path for folder with /<task>/categories.csv file")
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="path for folder to write results")
    parser.add_argument("--number_queries", default=-1, type=int, required=False,
                        help="number of total probe queries.")
    parser.add_argument("--batch_size", default=1, type=int, required=False,
                        help="batch_size")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str, required=False,
                        help="bert model name ['bert-base-cased' or 'bert-large-cased']")
    parser.add_argument("--sentence_type", default="type-I", type=str, required=False,
                        help="prompt sentence type ['type-I' or 'type-II']")

    args = parser.parse_args()

    path = "{}/{}/categories.csv".format(args.input_folder, args.task)
    if args.number_queries != -1:
        df = pd.read_csv(path, lineterminator="\n", nrows=args.number_queries)
    else:
        df = pd.read_csv(path, lineterminator="\n")

    domain = {'ml25m':'movie', 'gr':'book', 'music':'music album'} [args.task]

    probe = MaskedLanguageModelProbe(input_data = df,
                                    batch_size = args.batch_size,
                                    bert_model = args.bert_model,
                                    item_domain = domain,
                                    sentence_type = args.sentence_type)
    results = probe.run_probe()
    results_df = pd.DataFrame(results,\
         columns = ["preds", "labels", "raw_queries"])
    results_df["intersection_50"] = results_df.apply(lambda r: set(r["preds"].split(" ")[0:50]).intersection(set(r["labels"].split(" "))),axis=1)
    results_df["R@50"] = results_df.apply(lambda r: len(r["intersection_50"])/len(r["labels"].split(" ")),axis=1)
    results_df["intersection_10"] = results_df.apply(lambda r: set(r["preds"].split(" ")[0:10]).intersection(set(r["labels"].split(" "))),axis=1)
    results_df["R@10"] = results_df.apply(lambda r: len(r["intersection_10"])/len(r["labels"].split(" ")),axis=1)
    logging.info("Average R@50: {}".format(results_df["R@50"].mean()))
    logging.info("Average R@10: {}".format(results_df["R@10"].mean()))
    file_signature = "probe_type_categories_task_{}_num_queries_{}_model_{}_sentence_type_{}".\
        format(args.task, args.number_queries, args.bert_model, args.sentence_type)
    results_df.to_csv(args.output_folder+file_signature+".csv", index=False)

if __name__ == "__main__":
    main()