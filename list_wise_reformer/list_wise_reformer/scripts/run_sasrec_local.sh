#cd ../models/SASRec
#
#virtualenv --python=/usr/bin/python2.7 env
#source env/bin/activate
#
#pip install tqdm
#pip install -Iv tensorflow==1.12
#pip install pandas
#
#cd ../model/scripts
#python create_sasrec_data.py  \
#    --task ml25m \
#    --data_folder /Users/gustavopenha/personal/recsys20/data/recommendation/ \
#    --sasrec_folder /Users/gustavopenha/personal/recsys20/list_wise_reformer/list_wise_reformer/models

cd ../models/SASRec

python main.py --dataset=train_ml25m \
  --train_dir=default \
  --num_epochs=10 \
  --eval_epochs=10000 \
  --maxlen=200 \
  --dataset_list_valid valid_ml25m.csv \
  --output_predictions_folder /Users/gustavopenha/personal/recsys20/data/output_data/sasrec/1/
