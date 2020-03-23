import pandas as pd
from tqdm import tqdm
from transformers import DataProcessor, InputExample

def toBPRMFFormat(train_sessions_df, test_session_df):
    num_user = train_sessions_df.shape[0]
    item_map = {}
    i = 0
    for data in [train_sessions_df, test_session_df]:
        for _, r in data.iterrows():
            for item in r['query'].split(" [SEP] "):
                if item not in item_map:
                    item_map[item] = i
                    i += 1
            for col in data.columns[1:]:
                if r[col] not in item_map:
                    item_map[r[col]] = i
                    i += 1
    return num_user, item_map

def toSASRecFormat(train_sessions_df, test_session_df):
    u_ids = {}
    u_count = 1
    i_ids = {}
    i_count = 1

    transformed_train = []
    for user, r in train_sessions_df.iterrows():
        if user not in u_ids:
            u_ids[user]=u_count
            u_count+=1
        for item in r['query'].split(" [SEP] "):
            if item not in i_ids:
                i_ids[item]=i_count
                i_count+=1
            transformed_train.append([u_ids[user], i_ids[item]])

    transformed_test = []
    for user, r in test_session_df.iterrows():
        if r['relevant_doc'] not in i_ids:
            i_ids[r['relevant_doc']] = i_count
            i_count+=1
        rel_item = i_ids[r['relevant_doc']]
        seq = []
        for item in r['query'].split(" [SEP] "):
            if item not in i_ids:
                i_ids[item]=i_count
                i_count+=1
            seq.append(i_ids[item])
        test_items = []
        for c in test_session_df.columns:
            item = r[c]
            if item not in i_ids:
                i_ids[item]=i_count
                i_count+=1
            if "non_relevant_" in c:
                test_items.append(i_ids[item])
        transformed_test.append([u_ids[user],
                                seq,
                                [rel_item] + test_items])
    transformed_test = pd.DataFrame(transformed_test, columns=['user_id',
                                                           'seq',
                                                           'test_items'])
    return transformed_train, transformed_test

def toBERT4RecFormat(train_sessions_df, test_session_df):
    u_ids = {}
    u_count = 0
    i_ids = {}
    i_count = 0

    transformed_train = []
    for user, r in train_sessions_df.iterrows():
        if user not in u_ids:
            u_ids[user]=u_count
            u_count+=1
        for item in r['query'].split(" [SEP] "):
            if item not in i_ids:
                i_ids[item]=i_count
                i_count+=1
            transformed_train.append([u_ids[user], i_ids[item]])

    transformed_test = []
    for user, r in test_session_df.iterrows():
        if r['relevant_doc'] not in i_ids:
            i_ids[r['relevant_doc']] = i_count
            i_count+=1
        rel_item = i_ids[r['relevant_doc']]
        seq = []
        for item in r['query'].split(" [SEP] "):
            if item not in i_ids:
                i_ids[item]=i_count
                i_count+=1
            seq.append(i_ids[item])
        test_items = []
        for c in test_session_df.columns:
            item = r[c]
            if item not in i_ids:
                i_ids[item]=i_count
                i_count+=1
            if "non_relevant_" in c:
                test_items.append(i_ids[item])

        transformed_test.append([seq, [rel_item] + test_items])
    transformed_test = pd.DataFrame(transformed_test, columns=['seq',
                                                               'test_items'])
    return transformed_train, transformed_test

def generate_anserini_json_collection(all_responses):
    documents = []
    doc_set = set()
    doc_id = 0
    for _, r in tqdm(all_responses.iterrows()):
        for column in [c for c in all_responses.columns
                       if "non_relevant_" in c] + ["relevant_doc"]:
            if r[column] not in doc_set:
                documents.append({'id': doc_id,
                                  'contents': r[column]})
                doc_id+=1
                doc_set.add(r[column])
    return documents

class ConversationResponseRankingProcessor(DataProcessor):
    """Processor for the Conversation Response Ranking datasets,
    such as MSDialog, UDC and MANtIS."""

    def get_train_examples(self, data_dir):
        """See base class."""
        pass

    def get_dev_examples(self, data_dir):
        """See base class."""
        pass

    def get_test_examples(self, data_dir):
        """See base class."""
        pass

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_examples_from_sessions(self, sessions, num_neg_samples):
        examples = []
        for idx, row in sessions.iterrows():
            #add relevant:
            examples.append(
                InputExample(guid=str(idx),
                             text_a=row['query'],
                             text_b=row['relevant_doc'],
                             label="1")
            )
            #add non relevant
            i=0
            for col in [c for c in sessions.columns if "non_relevant" in c]:
                if i> num_neg_samples:
                    break
                i+=1
                examples.append(
                    InputExample(guid=str(idx),
                                 text_a=row["query"],
                                 text_b=row[col],
                                 label="0")
                )
        return examples