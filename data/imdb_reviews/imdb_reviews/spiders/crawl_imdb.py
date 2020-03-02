import scrapy
import pandas as pd
from IPython import embed

class IMDBReviews(scrapy.Spider):
    name = "imdb_reviews"

    custom_settings = {
        'DOWNLOAD_DELAY': 0.5
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path_links, path_names = "../movies_links.csv", "../movies_names.csv"
        movie_links = pd.read_csv(path_links)
        movie_titles = pd.read_csv(path_names)

        movie_titles['movieId'] = movie_titles['movieId'].astype(str)
        id_to_name = movie_titles.set_index('movieId').to_dict()['title']
        movie_links['movieId'] = movie_titles['movieId'].astype(str)
        id_to_imdb = movie_links.set_index('movieId').to_dict()['imdbId']

        imbd_id_to_name = {}
        for k, v in id_to_name.items():
            imbd_id_to_name[id_to_imdb[k]] = v

        df = pd.read_csv("crawled.csv", sep='\t')
        crawled = set(df.name.unique())
        self.urls = []
        self.url_to_name = {}
        for k,v in imbd_id_to_name.items():
            if v not in crawled:
                url = ("https://www.imdb.com/title/tt" +
                        ("0" * (7 - len(str(k)))) + str(k)+
                        "/reviews")
                self.urls.append(url)
                self.url_to_name[url] = v

    def start_requests(self):
        for url in self.urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        with open("crawled.csv", "a+") as f:
            f.write(self.url_to_name[response.url]+"\n")

        reviews = response.css('div.text::text').getall()

        with open("reviews.csv","a+") as f:
            for review in reviews[0:20]:
                review = review.replace("\n", '').replace("\t", ' ')
                if len(review) > 50:
                    f.write(self.url_to_name[response.url]+"\t"+review+"\n")