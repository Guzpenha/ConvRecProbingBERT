from datetime import datetime
from IPython import embed
import requests
import traceback
import json

url = "https://api.pushshift.io/reddit/{}/search?limit=1000&subreddit=booksuggestions,moviesuggestions&sort=desc&before="

start_time = datetime.utcnow()

def downloadFromUrl(filename, object_type, nest_level = 1):
    print(f"Saving {object_type}s to {filename}")

    count = 0
    handle = open(filename, 'a+')
    # previous_epoch = int(start_time.timestamp())
    previous_epoch = 1509432386
    while True:
        new_url = url.format(object_type) + str(previous_epoch)
        json_data = requests.get(new_url,
                                 headers={'User-Agent': "Post downloader based on script by /u/Watchful1"}).json()
        if 'data' not in json_data:
            break
        objects = json_data['data']
        if len(objects) == 0:
            break

        for object in objects:
            previous_epoch = object['created_utc'] - 1
            count += 1
            if object_type == 'comment':
                try:
                    response = json.dumps({
                        "response_id": object['id'],
                        "created_utc": object['created_utc'],
                        "created_date": datetime.fromtimestamp(object['created_utc']).strftime("%Y-%m-%d"),
                        "permalink": object['permalink'] if 'permalink' in object else 'not_available',
                        "parent_id": object['parent_id'],
                        "text": object['body'].encode(encoding='ascii', errors='ignore').decode().replace("\n", ' '),
                        "is_submitter": object['is_submitter'] if 'is_submitter' in object else 'not_available',
                        "score": object['score'],
                        "subreddit": object['subreddit']
                    })
                    handle.write(response+"\n")
                except Exception as err:
                    print(f"Couldn't print post from : {new_url}")
                    print(traceback.format_exc())
            elif object_type == 'submission':
                if object['is_self']:
                    if 'selftext' not in object or object['selftext'] == '[removed]':
                        continue
                    try:
                        if object['num_comments'] > 0:
                            submission = json.dumps({
                                "submission_id": object['id'],
                                "created_utc": object['created_utc'],
                                "created_date": datetime.fromtimestamp(object['created_utc']).strftime("%Y-%m-%d"),
                                "full_link": object['full_link'],
                                "title": object['title'],
                                "text": object['selftext'].encode(encoding='ascii', errors='ignore').decode().replace("\n", ' '),
                                "subreddit": object['subreddit']
                            })
                            handle.write(submission+"\n")
                    except Exception as err:
                        print(f"Couldn't print post from : {new_url}")
                        print(traceback.format_exc())

        print("Saved {} {}s through {}".format(count, object_type,
                                               datetime.fromtimestamp(previous_epoch).strftime("%Y-%m-%d")))
    print(f"Saved {count} {object_type}s")
    handle.close()


def main():

    # downloadFromUrl("posts.json", "submission")
    downloadFromUrl("comments.json", "comment")

if __name__ == "__main__":
    main()
