from IPython import embed
import pandas as pd
from tqdm import tqdm
from transformers import DataProcessor, InputExample
from nltk.tokenize import TweetTokenizer
import numpy as np

def toBPRMFFormat(train_sessions_df, valid_session_df):
    num_user = train_sessions_df.shape[0]
    item_map = {}
    i = 0
    for data in [train_sessions_df, valid_session_df]:
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

def toSASRecFormat(train_sessions_df, valid_session_df):
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
    for user, r in valid_session_df.iterrows():
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
        for c in valid_session_df.columns:
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

def toBERT4RecFormat(train_sessions_df, valid_session_df):
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
    for user, r in valid_session_df.iterrows():
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
        for c in valid_session_df.columns:
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

def toBERT4RecPytorchFormat(train_sessions_df, valid_session_df):
    train = {}
    val = {}
    u_ids = {}
    u_count = 0
    i_ids = {}
    i_count = 0

    for user, r in tqdm(train_sessions_df.iterrows()):
        user=user+1
        if user not in u_ids:
            u_ids[user]=u_count
            u_count+=1
        train[u_ids[user]] = []
        for item in r['query'].split(" [SEP] "):
            if item not in i_ids:
                i_ids[item]=i_count
                i_count+=1
            train[u_ids[user]].append(i_ids[item])

    for user, r in tqdm(valid_session_df.iterrows()):
        user = user + 1
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
        for c in valid_session_df.columns:
            item = r[c]
            if item not in i_ids:
                i_ids[item]=i_count
                i_count+=1
            if "non_relevant_" in c:
                test_items.append(i_ids[item])
        val[u_ids[user]]= [rel_item] + test_items

    smap = { i+1 : i for i in range(len(i_ids))}
    dataset = {
        'train': train,
        'val': val,
        'test': val,
        'umap': u_ids,
        'smap': smap
    }
    return dataset

def add_sentence_to_vocab(sentence, vocab, id, tknzr):
    for word in tknzr.tokenize(sentence):
        if word not in vocab:
            vocab[word] = id
            id += 1
    return id

def word_to_ids(sentence, vocab, tknzr):
    return [vocab[word] for word in tknzr.tokenize(sentence)]

def add_sentence_to_char_vocab(sentence, vocab, id):
    for c in sentence:
        if c not in vocab:
            vocab[c] = id
            id+=1
    return id

def toU2UIMNFormat(train_sessions_df, valid_session_df):
    # Create vocab
    char_vocab = {}
    char_id = 0
    id=1
    vocab = {'<PAD>': 0, 'UNKNOWN' : 1, '__eou__' : 2, '__eot__': 3}
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    for df in [train_sessions_df, valid_session_df]:
        for _, r in tqdm(df.iterrows()):
            query = r['query']
            id = add_sentence_to_vocab(query, vocab, id, tknzr)
            for column in [c for c in train_sessions_df.columns
                               if "non_relevant_" in c] + ["relevant_doc"]:
                id = add_sentence_to_vocab(r[column], vocab, id, tknzr)

    # Create pool of responses
    doc_id = 0
    responses = {}
    for df in [train_sessions_df, valid_session_df]:
        for idx, r in df.iterrows():
            for doc_col in df.columns[1:]:
                tokenized_doc = ' '.join(tknzr.tokenize(r[doc_col]))
                if tokenized_doc not in responses:
                    responses[tokenized_doc] = str(doc_id)
                    doc_id+=1


    train = []
    valid = []
    for df, output in [(train_sessions_df, train),
                       (valid_session_df, valid)]:
        for idx, r in df.iterrows():
            query = r['query'].replace("[UTTERANCE_SEP]", "__eou__")
            tokenized_q = ' '.join(tknzr.tokenize(query))
            char_id = add_sentence_to_char_vocab(tokenized_q, char_vocab, char_id)

            tokenized_doc = ' '.join(tknzr.tokenize(r['relevant_doc']))
            char_id = add_sentence_to_char_vocab(tokenized_doc, char_vocab, char_id)
            positive_doc_id = responses[tokenized_doc]

            neg_ids = []
            for neg_col in [c for c in train_sessions_df.columns
                           if "non_relevant_" in c]:
                tokenized_doc = ' '.join(tknzr.tokenize(r[neg_col]))
                char_id = add_sentence_to_char_vocab(tokenized_doc, char_vocab, char_id)
                neg_ids.append(responses[tokenized_doc])

            instance = [idx,
                        tokenized_q,
                        positive_doc_id,
                        '|'.join(neg_ids)]
            output.append(instance)

    return train, valid, responses, vocab, char_vocab

def toDAMFormat(train_sessions_df, valid_session_df, number_ns_train=1):
    '''
    Pickle data = {'y': [[1, 0]],
    'c':[[word_id1, word_id2]],
    'r': [[word_id1, word_id2]],}
    '''
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

    #create vocabulary
    word_to_id = {'UTTERANCE_SEP' :  0 }
    id=1
    for df in [train_sessions_df, valid_session_df]:
        for _, r in tqdm(df.iterrows()):
            query = r['query']
            id = add_sentence_to_vocab(query, word_to_id, id, tknzr)
            for column in [c for c in train_sessions_df.columns
                               if "non_relevant_" in c] + ["relevant_doc"]:
                id = add_sentence_to_vocab(r[column], word_to_id, id, tknzr)


    #create data to pickle
    train_dict = { 'y' : [], 'c' : [], 'r' : [] }
    valid_dict = { 'y' : [], 'c' : [], 'r' : [] }

    for set_dict, df, limit_ns in [(train_dict, train_sessions_df, True),
                         (valid_dict, valid_session_df, False)]:
        for _, r in tqdm(df.iterrows()):
            #adding relevant instance
            set_dict['y'].append(1)
            set_dict['c'].append(word_to_ids(r['query'], word_to_id, tknzr))
            set_dict['r'].append(word_to_ids(r['relevant_doc'], word_to_id, tknzr))

            for i, column in enumerate([c for c in df.columns
                           if "non_relevant_" in c]):
                if i == number_ns_train and limit_ns:
                    break
                set_dict['y'].append(0)
                set_dict['c'].append(word_to_ids(r['query'], word_to_id, tknzr))
                set_dict['r'].append(word_to_ids(r[column], word_to_id, tknzr))

    return (train_dict, valid_dict, valid_dict), word_to_id

def toItemIDFormat(df, item_map, tokenizer):
    i_count = len(item_map)

    new_df = []
    for _, r in tqdm(df.iterrows()):
        q_items = []
        for item in r['query'].split(" [SEP] "):
            if item not in item_map:
                item_map[item] = "[ITEM_"+str(i_count)+"]"
                i_count+=1
            q_items.append(str(item_map[item]))
        query=" [SEP] ".join(q_items)

        if r['relevant_doc'] not in item_map:
            item_map[r['relevant_doc']] = "[ITEM_"+str(i_count)+"]"
            i_count+=1
        rel_doc = item_map[r['relevant_doc']]

        non_rel_items = []
        for c in df.columns:
            if "non_relevant_" in c:
                item = r[c]
                if item not in item_map:
                    item_map[item] = "[ITEM_"+str(i_count)+"]"
                    i_count += 1
                non_rel_items.append(item_map[item])

        new_df.append([query, rel_doc] + non_rel_items)
    return pd.DataFrame(new_df, columns=df.columns), tokenizer

def toMSNFormat(train_sessions_df, valid_session_df, number_ns_train=1):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

    #create vocabulary
    word_to_id = {'<PAD>' :  0 }
    id=1
    for df in [train_sessions_df, valid_session_df]:
        for _, r in tqdm(df.iterrows()):
            query = r['query']
            id = add_sentence_to_vocab(query, word_to_id, id, tknzr)
            for column in [c for c in train_sessions_df.columns
                               if "non_relevant_" in c] + ["relevant_doc"]:
                id = add_sentence_to_vocab(r[column], word_to_id, id, tknzr)

    train = ([], [], [])
    valid = ([], [], [])
    vocab_embed = (word_to_id, [np.random.uniform(-1,1,200)
                           for _ in range(len(word_to_id))])
    max_words = 50
    max_utterances = 10

    def split_and_pad_query(query, max_utterances, max_words, tknzr, word_to_id):
        splitted_and_padded_c = np.zeros((max_utterances, max_words), dtype=int).tolist()
        for idx, utterance in enumerate(query.split("[UTTERANCE_SEP]")):
            if idx > max_utterances:
                break
            for j, id in enumerate(word_to_ids(utterance, word_to_id, tknzr)):
                if j > max_words:
                    break
                splitted_and_padded_c[max_utterances - 1 - idx][max_words - 1 - j] = id
        return splitted_and_padded_c

    def pad_document(doc, max_words, tknzr, word_to_id):
        padded_r = np.zeros(max_words, dtype=int).tolist()
        for j, id in enumerate(word_to_ids(doc, word_to_id, tknzr)):
            if j > max_words:
                break
            padded_r[max_words - 1 - j] = id
        return padded_r

    for set_triplet, df, limit_ns in [(train, train_sessions_df, True),
                                    (valid, valid_session_df, False)]:
        for _, r in tqdm(df.iterrows()):
            #adding relevant instance
            splitted_and_padded_c = split_and_pad_query(r['query'], max_utterances, max_words, tknzr, word_to_id)
            padded_r = pad_document(r['relevant_doc'], max_words, tknzr, word_to_id)
            set_triplet[0].append(splitted_and_padded_c)
            set_triplet[1].append(padded_r)
            set_triplet[2].append(1)

            for i, column in enumerate([c for c in df.columns
                                        if "non_relevant_" in c]):
                if i == number_ns_train and limit_ns:
                    break
                padded_r = pad_document(r[column], max_words, tknzr, word_to_id)
                set_triplet[0].append(splitted_and_padded_c)
                set_triplet[1].append(padded_r)
                set_triplet[2].append(0)

    return train, valid, vocab_embed

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


def acumulate_lists(l1, l2, acum_step):
    acum_l1 = []
    acum_l2 = []
    current_l1 = []
    current_l2 = []
    for i in range(len(l2)):
        current_l1.append(l1[i][0])
        current_l2.append(l2[i][0])
        if (i + 1) % acum_step == 0 and i != 0:
            acum_l1.append(current_l1)
            acum_l2.append(current_l2)
            current_l1 = []
            current_l2 = []
    return acum_l1, acum_l2