import pandas as pd
import json
from IPython import embed
from collections import defaultdict

def make_n_turn_instances(n_minus_one_dialogues):
    existing_ids = set(n_minus_one_dialogues['last_response_id'].unique())
    posts_dict = n_minus_one_dialogues.set_index("last_response_id").to_dict()

    posts_responses_from_submitter = defaultdict(lambda : [])
    i=0
    with open('comments.json', 'r') as f:
        for l in f:
            i+=1
            response = json.loads(l)
            parent_id = response['parent_id'].split("_")[1]
            if parent_id in existing_ids and response['is_submitter']:
                posts_responses_from_submitter[parent_id].append(response)
            # if i>1000:
            #     break

    utterances_instances = []
    for k,v in posts_responses_from_submitter.items():
        for response in v:
            utterances_instances.append([
                posts_dict['query'][k] + " [UTTERANCE_SEP] " +
                posts_dict['relevant_response'][k] + " [UTTERANCE_SEP] "+ response['text'],
                response['response_id']
            ])
    df = pd.DataFrame(utterances_instances, columns=['query', 'last_response_id'])
    existing_ids = set(df['last_response_id'].unique())
    posts_dict = df.set_index("last_response_id").to_dict()

    posts_responses_to_submitter = defaultdict(lambda: [])
    i=0
    with open('comments.json', 'r') as f:
        for l in f:
            i+=1
            response = json.loads(l)
            parent_id = response['parent_id'].split("_")[1]
            if parent_id in existing_ids:
                posts_responses_to_submitter[parent_id].append(response)
            # if i>10000:
            #     break

    utterances_instances_final = []
    for k,v in posts_responses_to_submitter.items():
        for response in v:
            utterances_instances_final.append([
                posts_dict['query'][k],
                response['text'],
                response['score'],
                response['response_id'],
                response['subreddit']
            ])
    df = pd.DataFrame(utterances_instances_final, columns=['query',
                                                   'relevant_response',
                                                   'relevance_score',
                                                   'last_response_id',
                                                   'subreddit'])
    return df

def join_post_comments():
    posts = []
    with open('posts.json', 'r') as f:
        for l in f:
            post = json.loads(l)
            if post['text'] != "" and post['text']!= "[deleted]":
                posts.append(v for v in post.values())
    posts = pd.DataFrame(posts, columns = [k for k in post.keys()])
    posts['query'] = posts.apply(lambda r: r['title'] + " " + r['text'], axis=1)
    existing_ids = set(posts['submission_id'].unique())
    posts_dict = posts.set_index("submission_id").to_dict()

    # making 1-turn conversations
    posts_responses = defaultdict(lambda : [])
    i=0
    with open('comments.json', 'r') as f:
        for l in f:
            i+=1
            response = json.loads(l)
            parent_id = response['parent_id'].split("_")[1]
            if parent_id in existing_ids:
                posts_responses[parent_id].append(response)
            # if i>1000:
            #     break
    two_utterances_instances = []
    for k,v in posts_responses.items():
        for response in v:
            two_utterances_instances.append([
                posts_dict['query'][k],
                response['text'],
                response['score'],
                response['response_id'],
                response['subreddit']
            ])
    df_two_utt = pd.DataFrame(
        two_utterances_instances, columns=['query',
                                           'relevant_response',
                                           'relevance_score',
                                           'last_response_id',
                                           'subreddit'])

    all_turns = [df_two_utt]

    # making > 2 turn  conversations
    max_turns = 5
    for i in range(max_turns):
        all_turns.append(make_n_turn_instances(all_turns[-1]))
    return pd.concat(all_turns)

def main():
    joined_dialogues = join_post_comments()
    joined_dialogues['query'] = joined_dialogues.apply(lambda r: r['query'].
                                                       replace("\n", ' ').
                                                       replace("\t", ' ').
                                                       replace("\r", ' '), axis=1)
    joined_dialogues[(joined_dialogues['relevant_response'] != "[deleted]")
            & (joined_dialogues['relevance_score'] > 1)].to_csv("dialogues.csv", index=False)

if __name__ == '__main__':
    main()