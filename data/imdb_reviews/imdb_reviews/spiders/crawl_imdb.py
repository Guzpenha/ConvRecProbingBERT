import scrapy
import pandas as pd

class IMDBReviews(scrapy.Spider):
    name = "imdb_reviews"

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

        df = pd.read_csv("reviews.csv", sep="\t", names=['name', 'review'])
        crawled = set(df.name.unique())
        self.urls = []
        self.url_to_name = {}
        for k,v in imbd_id_to_name.items():
            if v not in crawled:
                url = ("https://www.imdb.com/title/tt" +
                        ("0" * (9 - len(str(k)))) + str(k)+
                        "/reviews")
                self.urls.append(url)
                self.url_to_name[url] = v

    def start_requests(self):
        for url in self.urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        reviews = response.css('div.text::text').getall()
        # id = response.url.split('/reviews')[0].split('/tt')[1]
        # id = str(int(id))
        with open("reviews.csv","a+") as f:
            for review in reviews[0:20]:
                review = review.replace("\n", '').replace("\t", ' ')
                if len(review) > 50:
                    f.write(self.url_to_name[response.url]+"\t"+review+"\n")