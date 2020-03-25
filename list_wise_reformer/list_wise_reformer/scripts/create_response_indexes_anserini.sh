wANSERINI_FOLDER=/Users/gustavopenha/personal/recsys20/data/anserini
DATA_FOLDER=/Users/gustavopenha/personal/recsys20/data/dialogue/

for TASK in books
do
  mkdir $DATA_FOLDER/${TASK}/anserini_index/
  mkdir $DATA_FOLDER/${TASK}/anserini_json/

  python create_anserini_data.py \
    --task ${TASK} \
    --data_folder $DATA_FOLDER \
    --output_folder $DATA_FOLDER/${TASK}/anserini_json/

  sh $ANSERINI_FOLDER/target/appassembler/bin/IndexCollection -collection JsonCollection \
    -generator LuceneDocumentGenerator -threads 9 -input $DATA_FOLDER/${TASK}/anserini_json/ \
    -index $DATA_FOLDER/${TASK}/anserini_index -storePositions -storeDocvectors -storeRawDocs
done