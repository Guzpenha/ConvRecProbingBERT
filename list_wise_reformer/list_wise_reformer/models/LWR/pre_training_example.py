from list_wise_reformer.models.LWR.model import ListWiseReformer
from list_wise_reformer.models.LWR.trainer import LWRTrainer
from list_wise_reformer.models.LWR.dataset import LWRRecommenderPretrainingDataLoader
from list_wise_reformer.models.LWR.loss import custom_losses
from transformers import BertTokenizer
from IPython import embed
import pandas as pd
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Setting minimal parameters
parser = argparse.ArgumentParser()

# Input and output configs
parser.add_argument("--load_model", default="", type=str, required=False,
                    help="Path with model weights to load before training.")
parser.add_argument("--save_model", default=False, type=str, required=False,
                    help="Save trained model at the end of training.")

# Training procedure
parser.add_argument("--seed", default=42, type=str, required=False,
                    help="random seed")
parser.add_argument("--num_epochs", default=100, type=int, required=False,
                    help="Number of epochs for training.")
parser.add_argument("--max_gpu", default=-1, type=int, required=False,
                    help="max gpu used")
parser.add_argument("--validate_epochs", default=2, type=int, required=False,
                    help="Run validation every <validate_epochs> epochs.")
parser.add_argument("--num_validation_instances", default=-1, type=int, required=False,
                    help="Run validation for a sample of <num_validation_instances>. To run on all instances use -1.")
parser.add_argument("--train_batch_size", default=1, type=int, required=False,
                    help="Training batch size.")
parser.add_argument("--val_batch_size", default=32, type=int, required=False,
                    help="Validation and test batch size.")
parser.add_argument("--num_candidate_docs_train", default=4, type=int, required=False,
                    help="Number of documents to use during training (the input must already have the cand. docs)")
parser.add_argument("--num_candidate_docs_eval", default=4, type=int, required=False,
                    help="Number of documents to use during evaluation (the input must already have the cand. docs)")
parser.add_argument("--sample_data", default=-1, type=int, required=False,
                    help="Amount of data to sample for training and eval. If no sampling required use -1.")

# Model hyperparameters
parser.add_argument("--input_representation", default="text", type=str, required=False,
                    help="Represent the input as 'text' or 'item_ids' (available only for rec)")
parser.add_argument("--num_heads", default=2, type=int, required=False,
                    help="Number of attention heads.")
parser.add_argument("--lr", default=5e-5, type=float, required=False,
                    help="Learning rate.")
parser.add_argument("--max_seq_len", default=1024, type=int, required=False,
                    help="Maximum sequence length for the inputs.")
parser.add_argument("--hidden_dim", default=256, type=int, required=False,
                    help="Hidden dimension size.")
parser.add_argument("--depth", default=2, type=int, required=False,
                    help="Depth of reformer.")
parser.add_argument("--loss", default="PointwiseRMSE", type=str, required=False,
                    help="Loss function to use [cross-entropy, " + ",".join(custom_losses.keys()) + "].")
parser.add_argument("--pre_training_objective", default="shuffle_session_w_noise", type=str, required=False,
                    help="Pre training objective ['shuffle_session', 'shuffle_session_w_noise'].")
MAX_SEQ_LEN = 1024
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.max_len = MAX_SEQ_LEN

# Creating input data
train = pd.DataFrame([["Lord of the Rings: The Two Towers, The (2002) [SEP] Back to the Future Part III (1990)"
                       " [SEP] Back to the Future Part II (1989) [SEP] Gattaca (1997) [SEP]"
                       " Bridge on the River Kwai, The (1957) [SEP] Pirates of the Caribbean: The Curse of the Black Pearl (2003) [SEP]"
                       " Finding Nemo (2003) [SEP] Good Morning, Vietnam (1987) [SEP] Dirty Dancing (1987) [SEP] Singin' in the Rain (1952) [SEP] "
                       "Moulin Rouge (2001) [SEP] NeverEnding Story, The (1984) [SEP] Lost in Translation (2003) [SEP] Requiem for a Dream (2000) "
                       "[SEP] Shrek 2 (2004) [SEP] Talk to Her (Hable con Ella) (2002) [SEP] Three Colors: Red (Trois couleurs: Rouge) (1994) "
                       "[SEP] Delicatessen (1991) [SEP] Three Colors: Blue (Trois couleurs: Bleu) (1993) [SEP] "
                       "Seventh Seal, The (Sjunde inseglet, Det) (1957) [SEP] Persona (1966) [SEP] Dolce Vita, La (1960) "
                       "[SEP] Strada, La (1954) [SEP] Black Cat, White Cat (Crna macka, beli macor) (1998) [SEP] "
                       "In the Mood For Love (Fa yeung nin wa) (2000) [SEP] Noi the Albino (Nói albinói) (2003)"
                       " [SEP] Fanny and Alexander (Fanny och Alexander) (1982) [SEP] Cries and Whispers (Viskningar och rop) (1972) "
                       "[SEP] Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001) [SEP] City of God (Cidade de Deus) (2002)",
                       'doc_1', 'doc_2', 'doc_3', 'doc10'],
                      ["Shawshank Redemption, The (1994) [SEP] Usual Suspects, The (1995) [SEP] Godfather, The (1972) [SEP]"
                       "Schindler's List (1993) [SEP] Godfather: Part II, The (1974) [SEP] One Flew Over the Cuckoo's Nest (1975) [SEP] "
                       "Goodfellas (1990) [SEP] Silence of the Lambs, The (1991) [SEP] City of God (Cidade de Deus) (2002) [SEP] "
                       "Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) [SEP] Fight Club (1999) [SEP] "
                       "Casablanca (1942) [SEP] Matrix, The (1999) [SEP] Pulp Fiction (1994) [SEP] Interstellar (2014) [SEP] "
                       "Gone Girl (2014) [SEP] Napoleon Dynamite (2004) [SEP] War of the Worlds (2005) [SEP] Walk the Line (2005)"
                       " [SEP] Rear Window (1954) [SEP] Seven Samurai (Shichinin no samurai) (1954) [SEP] "
                       "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) [SEP] "
                       "American Beauty (1999) [SEP] Lives of Others, The (Das leben der Anderen) (2006) [SEP]"
                       " Life Is Beautiful (La Vita è bella) (1997) [SEP] Lord of the Rings: The Return of the King, The (2003) [SEP] "
                       "Lord of the Rings: The Fellowship of the Ring, The (2001) [SEP] 12 Angry Men (1957) [SEP] Memento (2000) [SEP] "
                       "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001) [SEP] Inception (2010) [SEP] Hotel Rwanda (2004) [SEP] "
                       "Incendies (2010) [SEP] Dark Knight, The (2008) [SEP] American History X (1998) [SEP] King's Speech, The (2010) "
                       "[SEP] Cinema Paradiso (Nuovo cinema Paradiso) (1989) [SEP] Departures (Okuribito) (2008) [SEP] North by Northwest "
                       "(1959) [SEP] Looper (2012) [SEP] Prestige, The (2006) [SEP] Minority Report (2002) [SEP] Lord of the Rings: "
                       "The Two Towers, The (2002) [SEP] Secret in Their Eyes, The (El secreto de sus ojos) (2009) [SEP] Pianist,"
                       " The (2002) [SEP] Color Purple, The (1985) [SEP] Predestination (2014) [SEP] Eternal Sunshine of the Spotless Mind (2004) "
                       "[SEP] Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001) "
                       "[SEP] Lost in Translation (2003) [SEP] Apocalypse Now (1979) [SEP] Before Sunrise (1995) [SEP] "
                       "Saving Private Ryan (1998) [SEP] Before Sunset (2004) [SEP] Boot, Das (Boat, The) (1981) [SEP] "
                       "No Country for Old Men (2007) [SEP] Lawrence of Arabia (1962) [SEP] "
                       "Harry Potter and the Prisoner of Azkaban (2004) [SEP] Harry Potter and the Chamber of Secrets (2002) "
                       "[SEP] Requiem for a Dream (2000) [SEP] Starred Up (2013)",
                       'doc_2', 'doc_1', 'doc_6', 'doc_10']],
                     columns = ['query', 'relevant_doc', 'non_relevant_1', 'non_relevant_2', 'non_relevant_3'])
args = parser.parse_args()
args.sacred_ex = None

test, valid = train, train

dataloader = LWRRecommenderPretrainingDataLoader(args=args, train_df=train,
                                     val_df=valid, test_df=test,
                                     tokenizer=tokenizer)
# Instantiating components for training a ListWiseReformer
model = ListWiseReformer(
    num_tokens=len(dataloader.tokenizer),
    dim = 1048,
    depth = 12,
    max_seq_len = MAX_SEQ_LEN,
    num_doc_predictions=args.num_candidate_docs_train)


train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

trainer = LWRTrainer(args, model, train_loader, val_loader, test_loader)

trainer.fit()
trainer.test()
