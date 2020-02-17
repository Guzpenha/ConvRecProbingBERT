from IPython import embed
import json
import pandas as pd

author_names = {}
with open("goodreads_book_authors.json", 'r') as f:
    for l in f:
        author = json.loads(l)
        author_names[author['author_id']] = author['name']

with open("goodreads_books.json", 'r') as f:
    books = []
    for l in f:
        book = json.loads(l)
        if len(book['authors']) > 0:
            first_author = book['authors'][0]["author_id"]
            if first_author in author_names:
                books.append([book['book_id'],
                              book['title'] + " by "+
                              author_names[first_author]])
books = pd.DataFrame(books, columns=['bookId', 'title'])
books.to_csv("books_names.csv", index=False)