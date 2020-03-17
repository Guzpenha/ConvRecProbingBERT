from IPython import embed
import pandas as pd
import json

def main():
    posts = []
    with open('../reddit_crawling/posts.json', 'r') as f:
        for l in f:
            posts.append([v for v in json.loads(l).values()])
    df_posts = pd.DataFrame(posts, columns=[k for k in json.loads(l).keys()])
    print(df_posts.shape)
    print("Number of posts from each subreddit: ")
    print(df_posts.groupby("subreddit").count()['submission_id'])

    responses = []
    with open('../reddit_crawling/comments.json', 'r') as f:
        for l in f:
            responses.append([v for v in json.loads(l).values()])
    df_responses = pd.DataFrame(responses, columns=[k for k in json.loads(l).keys()])
    print(df_responses.shape)
    print("Number of responses from each subreddit: ")
    print(df_responses.groupby("subreddit").count()['response_id'])

    df_context_response = pd.read_csv("../reddit_crawling/dialogues.csv", lineterminator= "\n")
    df_context_response['turns'] = df_context_response.apply(lambda r:
                                                             (len(r['query'].split("[UTTERANCE_SEP]"))+1)//2, axis=1)
    df_context_response['words_context'] = df_context_response.apply(lambda r:
                                                                     len(r['query'].split(" ")), axis=1)
    df_context_response['words_response'] = df_context_response.apply(lambda r:
                                                                      len(str(r['relevant_response']).split(" ")), axis=1)

    print(df_context_response.shape)
    print("Number of context-response pairs from each subreddit: ")
    print(df_context_response.groupby("subreddit").count()["query"])
    print("Statistics: ")
    print(df_context_response[['subreddit', 'turns',
                               'relevance_score',
                               'words_context', 'words_response']].describe())
    for subreddit in df_context_response['subreddit'].unique():
        print(subreddit)
        print(df_context_response[df_context_response["subreddit"] == subreddit]
                                    [['subreddit', 'turns','relevance_score',
                                   'words_context', 'words_response']].describe())

    df_context_response[['subreddit', 'turns',
                               'relevance_score',
                               'words_context', 'words_response']].\
        to_csv("./conv_data_statistics.csv", index=False)

if __name__ == '__main__':
    main()