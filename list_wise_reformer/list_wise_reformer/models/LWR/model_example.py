from list_wise_reformer.models.LWR.model import ListWiseReformer
from transformers import BertTokenizer
import torch

MAX_SEQ_LEN = 4096
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.max_len = MAX_SEQ_LEN

model = ListWiseReformer(
    num_tokens= tokenizer.vocab_size,
    dim = 1048,
    depth = 12,
    max_seq_len = MAX_SEQ_LEN,
    num_doc_predictions=3,
    heads = 8,
    lsh_dropout = 0.1,
    layer_dropout = 0.1,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
    emb_dim = 128,        # embedding factorization for further memory savings
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    num_mem_kv = 128,       # persistent learned memory key values, from all-attention paper
    twin_attention = False, # both branches of the reversible network will be attention
    use_full_attn = False,  # use full self attention, for comparison
    full_attn_thres = 1024, # use full attention if context length is less than set value
    use_scale_norm = False # use scale norm from 'Transformers without tears' paper,
)

x = torch.randint(0, tokenizer.vocab_size, (2, MAX_SEQ_LEN)).long() #B X SEQ_LEN
y = model(x) # B X NUM_DOCS
print(y.shape)