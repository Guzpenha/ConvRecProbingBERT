mkdir ./data/search
mkdir ./data/recommendation
mkdir ./data/dialogue

#================================================#
#Download datasets for creating product search data
#================================================#
cd ./data/search

# Download reviews from Amazon Music
wget "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/CDs_and_Vinyl.json.gz"
gunzip CDs_and_Vinyl.json.gz
rm CDs_and_Vinyl.json.gz
wget "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_CDs_and_Vinyl.json.gz"
gunzip meta_CDs_and_Vinyl.json.gz
rm meta_CDs_and_Vinyl.json.gz

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
#categories file
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ah0_KpUterVi-AHxJ03iKD6O0NfbK0md' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ah0_KpUterVi-AHxJ03iKD6O0NfbK0md" -O gooreads_book_genres_initial.json.gz && rm -rf /tmp/cookies.txt
gunzip gooreads_book_genres_initial.json.gz
rm gooreads_book_genres_initial.json.gz
#file with other information of books (we use the publication year)
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK" -O goodreads_books.json.gz && rm -rf /tmp/cookies.txt
gunzip goodreads_books.json.gz
rm goodreads_books.json.gz

#================================================#
#Download dataset for conversational rec data
#================================================#
cd ../dialogue

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1naj_8j0MeCH2ZZ8u8WKLJqKDEoc5kggN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1naj_8j0MeCH2ZZ8u8WKLJqKDEoc5kggN" -O dialogues.csv.zip && rm -rf /tmp/cookies.txt
unzip dialogues.csv.zip
rm dialogues.csv.zip

git clone https://github.com/ReDialData/website.git
cd website
git checkout data