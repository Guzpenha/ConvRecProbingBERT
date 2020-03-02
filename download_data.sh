mkdir data/search
mkdir data/recommendation
mkdir data/dialogue

#================================================#
#Download datasets for creating product search data
#================================================#
cd data/search

#Download reviews from IMDB
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1quaTeELrdydQ9uFJi_EfruuoNPJ0TnhF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1quaTeELrdydQ9uFJi_EfruuoNPJ0TnhF" -O imdb_reviews.csv.zip && rm -rf /tmp/cookies.txt
unzip imdb_reviews.csv.zip
rm imdb_reviews.csv.zip

#Download reviews from GoodReads
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pQnXa7DWLdeUpvUFsKusYzwbA5CAAZx7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pQnXa7DWLdeUpvUFsKusYzwbA5CAAZx7" -O goodreads_reviews_dedup.json.gz && rm -rf /tmp/cookies.txt
gunzip goodreads_reviews_dedup.json.gz
rm goodreads_reviews_dedup.json.gz

#================================================#
#Download datasets for creating recommendation data
#================================================#
cd ../recommendation

#Download interactions from MovieLens 25m
wget "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
unzip ml-25m.zip
mv ml-25m/* ./
rm -rf ml-25m
rm tags.csv ml-25m.zip genome-scores.csv genome-tags.csv README.txt links.csv
mv movies.csv movies_names.csv
mv ratings.csv ml25m_ratings.csv

#Download interactions from GoodReads
#movie_names.csv
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lco8w7HO0oz202fqenGkmtSU7cWVZJEh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lco8w7HO0oz202fqenGkmtSU7cWVZJEh" -O books_names.csv && rm -rf /tmp/cookies.txt
#all interactions file (11G compressed)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n0vRIZM7uDRPqjXP8K6QSniRliE4Buy4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n0vRIZM7uDRPqjXP8K6QSniRliE4Buy4" -O goodreads_interactions_dedup.json.gz  && rm -rf /tmp/cookies.txt
gunzip goodreads_interactions_dedup.json.gz
rm goodreads_interactions_dedup.json.gz


#================================================#
#Download dataset for conversational rec data
#================================================#
cd ../dialogue

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1naj_8j0MeCH2ZZ8u8WKLJqKDEoc5kggN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1naj_8j0MeCH2ZZ8u8WKLJqKDEoc5kggN" -O dialogues.csv.zip && rm -rf /tmp/cookies.txt
unzip dialogues.csv.zip
rm dialogues.csv.zip