import zipfile
import json
from IPython import embed
import pandas as pd
import random
from tqdm import tqdm

random.seed(42)
with zipfile.ZipFile('./data/website/redial_dataset.zip', 'r') as z:
    z.extractall()

train_data = []
for line in open("./data/train_data.jsonl", "r"):
    train_data.append(json.loads(line))
print("Loaded {} train conversations".format(len(train_data)))

test_data = []
for line in open("./data/test_data.jsonl", "r"):
    test_data.append(json.loads(line))
print("Loaded {} test conversations".format(len(test_data)))


all_condensed_messages = []
all_mentioned_movies = set()
print("Condensing dialogues' utterances.")
for fold in [train_data, test_data]:
    for dialogue in tqdm(fold):
        messages = dialogue['messages']
        if type(dialogue['movieMentions']) == dict:
            for movie in dialogue['movieMentions'].values():
                all_mentioned_movies.add(movie)

        condensed_messages = []
        curr_utterance = ""
        current_worker_id = messages[0]['senderWorkerId']
        for i, utterance in enumerate(messages):
            utterance_text = utterance['text']            
            if utterance['senderWorkerId'] == current_worker_id:
                if i!=0:
                    curr_utterance+=". {}".format(utterance_text)
                else:
                    curr_utterance+="{}".format(utterance_text)
            else:
                if type(dialogue['movieMentions']) == dict:
                    for movie_id, title in dialogue['movieMentions'].items():
                        if title is not None:                    
                            curr_utterance = curr_utterance.replace("@{}".format(movie_id), title)
                condensed_messages.append(curr_utterance)
                curr_utterance = "{}".format(utterance_text)
                current_worker_id = utterance['senderWorkerId']
        all_condensed_messages.append(condensed_messages)

random.shuffle(all_condensed_messages)
all_conv = []
print("Creating training instances")
for condensed_messages in tqdm(all_condensed_messages):
    context = ""    
    for i in range(0, len(condensed_messages) - (len(condensed_messages)%2), 2):
        query = context + condensed_messages[i]
        relevant = condensed_messages[i+1]
        all_conv.append([query, relevant])
        context+=condensed_messages[i] + " [UTTERANCE_SEP] " + relevant + " [UTTERANCE_SEP] "    

all_conv = pd.DataFrame(all_conv, columns = ["query", "relevant_response"])
all_conv["subreddit"] = "all"
all_conv.to_csv("./data/dialogue/redial_dialogues.csv")