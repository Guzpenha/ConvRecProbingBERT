from reformer_pytorch.reformer_pytorch import FixedPositionEmbedding, \
    Reformer, default, identity

import torch
import torch.nn as nn

class ListWiseReformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, num_candidate_docs,
                 heads = 8, bucket_size = 64, n_hashes = 4,
                 ff_chunks = 100, attn_chunks = None,
                 weight_tie = False, lsh_dropout = 0.,
                 layer_dropout = 0., random_rotations_per_head = False,
                 twin_attention = False, use_scale_norm = False,
                 use_full_attn = False, full_attn_thres = 0,
                 num_mem_kv = 0, emb_dim = None,
                 fixed_position_emb = False):

        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = FixedPositionEmbedding(emb_dim) if fixed_position_emb else nn.Embedding(max_seq_len, emb_dim)
        self.to_model_dim = identity if emb_dim == dim else nn.Linear(emb_dim, dim)

        self.reformer = Reformer(dim, depth, max_seq_len, heads = heads, bucket_size = bucket_size,
                                 n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks,
                                  causal = False, weight_tie = weight_tie, lsh_dropout = lsh_dropout,
                                 layer_dropout = layer_dropout,
                                 random_rotations_per_head = random_rotations_per_head,
                                 twin_attention = twin_attention, use_scale_norm = use_scale_norm,
                                 use_full_attn = use_full_attn, full_attn_thres = full_attn_thres,
                                 num_mem_kv = num_mem_kv)
        self.pooler_dense = nn.Linear(dim, dim)
        self.pooler_activation = nn.Tanh()

        self.to_logits = nn.Linear(dim, num_candidate_docs)

    def forward(self, x, **kwargs):
        t = torch.arange(x.shape[1], device=x.device)
        x = self.token_emb(x)
        x = x + self.pos_emb(t).type(x.type())

        x = self.to_model_dim(x)
        x = self.reformer(x, **kwargs)

        first_token = x[:, 0] # "pool" first token ([CLS])
        pooled_x = self.pooler_dense(first_token)
        pooled_x = self.pooler_activation(pooled_x)

        return self.to_logits(pooled_x)