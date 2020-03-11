import pandas as pd

def toSASRecFormat(train_sessions_df, validation_session_df):
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

    transformed_val = []
    for user, r in validation_session_df.iterrows():
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
        for c in validation_session_df.columns:
            item = r[c]
            if item not in i_ids:
                i_ids[item]=i_count
                i_count+=1
            if "non_relevant_" in c:
                test_items.append(i_ids[item])
        transformed_val.append([u_ids[user],
                                seq,
                                [rel_item] + test_items])
    transformed_val = pd.DataFrame(transformed_val, columns=['user_id',
                                                           'seq',
                                                           'test_items'])
    return transformed_train, transformed_val