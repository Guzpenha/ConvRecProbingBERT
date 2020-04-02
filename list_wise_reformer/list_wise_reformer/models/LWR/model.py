from reformer_pytorch.reformer_pytorch import AxialPositionalEncoding, \
    FixedPositionalEmbedding, AbsolutePositionalEmbedding,\
    Reformer, default, Identity
import torch
import torch.nn as nn

class ListWiseReformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, num_doc_predictions, seed=42, heads=8, bucket_size=64, n_hashes=4,
                 add_local_attn_hash=False, ff_chunks=100, attn_chunks=1, causal=False, weight_tie=False,
                 lsh_dropout=0., ff_dropout=0., ff_mult=4, ff_activation=None, post_attn_dropout=0., layer_dropout=0.,
                 random_rotations_per_head=False, twin_attention=False, use_scale_norm=False, use_full_attn=False,
                 full_attn_thres=0, reverse_thres=0, num_mem_kv=0, one_value_head=False, emb_dim=None,
                 fixed_position_emb=False, axial_position_emb=False, axial_position_shape=(), axial_position_dims=()):
        super().__init__()
        torch.random.manual_seed(seed)
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.token_emb.weight.data.uniform_(-0.01, 0.01)

        self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        if axial_position_emb:
            self.pos_emb = AxialPositionalEncoding(emb_dim, max_seq_len, axial_position_shape, axial_position_dims)
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)

        self.reformer = Reformer(dim, depth, max_seq_len, heads=heads, bucket_size=bucket_size, n_hashes=n_hashes,
                                 add_local_attn_hash=add_local_attn_hash, ff_chunks=ff_chunks, attn_chunks=attn_chunks,
                                 causal=causal, weight_tie=weight_tie, lsh_dropout=lsh_dropout, ff_mult=ff_mult,
                                 ff_activation=ff_activation, ff_dropout=ff_dropout, post_attn_dropout=post_attn_dropout,
                                 layer_dropout=layer_dropout, random_rotations_per_head=random_rotations_per_head,
                                 twin_attention=twin_attention, use_scale_norm=use_scale_norm,
                                 use_full_attn=use_full_attn, full_attn_thres=full_attn_thres,
                                 reverse_thres=reverse_thres, num_mem_kv=num_mem_kv, one_value_head=one_value_head)

        self.pooler_dense = nn.Linear(dim, dim)
        self.pooler_activation = nn.Tanh()

        self.to_logits = nn.Linear(dim, num_doc_predictions)

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.pos_emb(x).type(x.type())

        x = self.to_model_dim(x)
        x = self.reformer(x, **kwargs)

        first_token = x[:, 0] # "pool" first token ([CLS])
        pooled_x = self.pooler_dense(first_token)
        pooled_x = self.pooler_activation(pooled_x)

        return self.to_logits(pooled_x)