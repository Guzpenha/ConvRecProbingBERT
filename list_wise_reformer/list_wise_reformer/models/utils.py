import pandas as pd

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