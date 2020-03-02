from IPython import embed
import pandas as pd

def main():
    movie_titles = pd.read_csv("../data/movies_names.csv")
    movie_titles['movieId'] = movie_titles['movieId'].astype(str)

    df = pd.read_csv("../data/imdb_reviews/reviews.csv", sep= "\t", names =['name', 'review'])
    print("Movies with reviews: ", len(df.name.unique()))
    print("Total number of items: ", movie_titles.shape[0])

    print("Reviews ",df.shape)
    print("Average reviews per movie ", df.groupby("name").count()['review'].mean())

if __name__ == '__main__':
    main()