import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *
import pandas as pd
import json

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--dataset_list_valid', default='')
parser.add_argument('--output_predictions_folder', default='')
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_candidates', default=51, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--eval_epochs', default=20, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

tf.set_random_seed(args.seed)

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print 'average sequence length: %.2f' % (cc / len(user_train))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())

T = 0.0
t0 = time.time()

try:
    for epoch in range(1, args.num_epochs + 1):

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})

        if epoch % args.eval_epochs == 0:
            t1 = time.time() - t0
            T += t1
            print 'Evaluating',
            t_test = evaluate(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print ''
            print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
except:
    sampler.close()
    f.close()
    exit(1)

if args.dataset_list_valid != "":
    if not os.path.isdir(args.output_predictions_folder):
        os.makedirs(args.output_predictions_folder)
    with open(os.path.join(args.output_predictions_folder, 'config.json'), 'w') as f:
        args.task = args.dataset.split("_")[1]
        args.recommender = "SASRec"
        args.seed = str(args.seed)
        args_dict = {}
        args_dict['args'] = vars(args)

        f.write(json.dumps(args_dict, indent=4, sort_keys=True))
    f.close()
    predictions = pred_custom_lists(model, dataset, args, sess)
    df = pd.DataFrame(predictions, columns=['prediction_'+str(i)
                                            for i in range(len(predictions[0]))])
    df.to_csv(args.output_predictions_folder+"/predictions.csv",
              index=False)
f.close()
sampler.close()
print("Done")
