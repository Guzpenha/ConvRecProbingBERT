from rec_probing.probes.nsp_probe import *
from rec_probing.probes.sim_probe import *

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
    parser.add_argument("--bert_model", default="bert-base-cased", type=str, required=False,
                        help="bert model name ['bert-base-cased' or 'bert-large-cased']")

    args = parser.parse_args()

    data_path = args.output_folder.split("/output_data")[0]
    df_pop = pd.read_csv("{}/recommendation/{}/popularity.csv".format(data_path, args.task))
    df_pop.columns = ["title", "pop"]
    pop = df_pop.set_index("title").to_dict()["pop"]

    dialogue_task={'ml25m':'movies', 'gr': 'books', 'music': 'music'}
    df_dialogues = pd.read_csv("{}/dialogue/{}/train.csv".format(data_path, dialogue_task[args.task]), 
            lineterminator= "\n")
    df_dialogues = df_dialogues[~df_dialogues.isnull().any(axis=1)] 

    path = "{}/{}/train.csv".format(args.input_folder, args.task)
    df = pd.read_csv(path, lineterminator="\n", nrows=df_dialogues.shape[0])    
    # df = pd.read_csv(path, lineterminator="\n", nrows=args.number_queries) 

    probe = NextSentencePredictionProbe(number_candidates = args.number_candidates, 
                        input_data = df,
                        number_queries_per_user=1,
                        batch_size = args.batch_size,
                        probe_type = args.probe_type,
                        bert_model = args.bert_model,
                        probe_technique = 'nsp',
                        items_popularity=pop)

    df_conv_format = probe.get_probes_in_dialogue_format()
    logging.info("Probe data size: {}".format(df_conv_format.shape[0]))
    logging.info("Dialogues size:  {}".format(df_dialogues.shape[0]))
    df_mt = pd.concat([df_dialogues, df_conv_format]).sample(frac=1).reset_index(drop=True)
    logging.info("MT data size:    {}".format(df_mt.shape[0]))

    file_signature = "probe_type_{}".\
        format(args.probe_type)
    df_mt.to_csv("{}/dialogue/{}/train_mt_{}.csv".format(data_path, dialogue_task[args.task], file_signature), index=False)

if __name__ == "__main__":
    main()
