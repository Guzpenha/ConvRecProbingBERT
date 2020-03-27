import time
import argparse
import pickle
import torch
from MSN import MSN

task_dic = {
    'ubuntu':'./dataset/ubuntu_data/',
    'douban':'./dataset/DoubanConversaionCorpus/',
    'alime':'./dataset/E_commerce/',
    'music':'./dataset/music/',
    'movies':'./dataset/movies/',
    'books':'./dataset/books/',
}
data_batch_size = {
    "ubuntu": 200,
    "douban": 150,
    "alime":  200,
    "music": 200,
    "movies": 200,
    "books": 200,
}

## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='music',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--is_training",
                    default=True,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--max_utterances",
                    default=10,
                    type=int,
                    help="The maximum number of utterances.")
parser.add_argument("--max_words",
                    default=50,
                    type=int,
                    help="The maximum number of words for each utterance.")
parser.add_argument("--batch_size",
                    default=0,
                    type=int,
                    help="The batch size.")
parser.add_argument("--gru_hidden",
                    default=300,
                    type=int,
                    help="The hidden size of GRU in layer 1")
parser.add_argument("--learning_rate",
                    default=1e-3,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2_reg",
                    default=0.0,
                    type=float,
                    help="The l2 regularization.")
parser.add_argument("--epochs",
                    default=1,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--output_predictions_folder",
                    required=True,
                    type=str,
                    help="The path to output results")
parser.add_argument("--seed",
                    default=42,
                    type=int,
                    help="Random Seed")

args = parser.parse_args()
args.batch_size = data_batch_size[args.task]
args.save_path += args.task + '.' + MSN.__name__ + ".pt"
args.score_file_path = task_dic[args.task] + args.score_file_path

print(args)
print("Task: ", args.task)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def train_model():
    path = task_dic[args.task]
    X_train_utterances, X_train_responses, y_train = pickle.load(file=open(path+"train.pkl", 'rb'))
    X_dev_utterances, X_dev_responses, y_dev = pickle.load(file=open(path+"test.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))
    model = MSN(word_embeddings, args=args)
    model.fit(
        X_train_utterances, X_train_responses, y_train,
        X_dev_utterances, X_dev_responses, y_dev
    )


def test_model():
    path = task_dic[args.task]
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))

    model = MSN(word_embeddings, args=args)
    model.load_model(args.save_path)
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)

def test_adversarial():
    path = task_dic[args.task]
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))
    model = MSN(word_embeddings, args=args)
    model.load_model(args.save_path)
    print("adversarial test set (k=1): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_1.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=2): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_2.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=3): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_3.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    if args.is_training:
        train_model()
        test_model()
    else:
        test_model()
        # test_adversarial()
    end = time.time()
    print("use time: ", (end-start)/60, " min")




