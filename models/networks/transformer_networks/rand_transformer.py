# Code for Transformer module
import math
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch import nn, einsum

from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm

from einops import rearrange, repeat

from .pos_embedding import PEPixelTransformer


class RandTransformer(nn.Module):
    def __init__(self, tf_conf, vq_conf=None):
        """init method"""
        super().__init__()

        # vqvae related params
        if vq_conf is not None:
            ntokens_vqvae = vq_conf.model.params.n_embed  # 512 -> num of patches and each has 256 dimension
            embed_dim_vqvae = vq_conf.model.params.embed_dim  # 256
        else:
            ntokens_vqvae = tf_conf.model.params.ntokens
            embed_dim_vqvae = tf_conf.model.params.embed_dim

        # pe
        pe_conf = tf_conf.pe
        pos_embed_dim = pe_conf.pos_embed_dim

        # tf
        mparam = tf_conf.model.params
        ntokens = mparam.ntokens
        d_tf = mparam.embed_dim  # 768
        nhead = mparam.nhead
        num_encoder_layers = mparam.nlayers_enc
        dim_feedforward = mparam.d_hid
        dropout = mparam.dropout
        self.ntokens_vqvae = ntokens_vqvae  # 512

        # Use the codebook embedding dim. Weights will be replaced by the learned codebook from vqvae.
        self.embedding_start = nn.Embedding(1, embed_dim_vqvae)  # shape (1, 256)
        self.embedding_encoder = nn.Embedding(ntokens_vqvae, embed_dim_vqvae)  # shape (512, 256)

        # position embedding
        self.pos_embedding = PEPixelTransformer(pe_conf=pe_conf)
        self.fuse_linear = nn.Linear(embed_dim_vqvae + pos_embed_dim + pos_embed_dim + 512, d_tf)  # + 512
        # cross attention
        self.cross_attention = CrossAttention(query_dim=512, context_dim=768)  # 512  768

        # transformer
        encoder_layer = TransformerEncoderLayer(d_tf, nhead, dim_feedforward, dropout, activation='relu')
        encoder_norm = LayerNorm(d_tf)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.dec_linear = nn.Linear(d_tf, ntokens_vqvae)  # 768 -> 512

        self.d_tf = d_tf

        self._init_weights()

    def _init_weights(self) -> None:
        """initialize the weights of params."""

        _init_range = 0.1

        self.embedding_start.weight.data.uniform_(-1.0 / self.ntokens_vqvae, 1.0 / self.ntokens_vqvae)
        self.embedding_encoder.weight.data.uniform_(-1.0 / self.ntokens_vqvae, 1.0 / self.ntokens_vqvae)

        self.fuse_linear.bias.data.normal_(0, 0.02)
        self.fuse_linear.weight.data.normal_(0, 0.02)

        self.dec_linear.bias.data.normal_(0, 0.02)
        self.dec_linear.weight.data.normal_(0, 0.02)


    def generate_square_subsequent_mask(self, sz, device):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

    def generate_square_id_mask(self, sz, device):
        mask = torch.eye(sz)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        return mask

    def forward_transformer(self, src, src_mask=None):
        output = self.encoder(src, mask=src_mask)
        # output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return output

    def forward(self, inp, inp_posn, tgt_posn, img_encoding=None):
        """ Here we will have the full sequence of inp """
        device = inp.get_device()
        seq_len, bs = inp.shape[:2]
        tgt_len = tgt_posn.shape[0]

        # token embedding
        sos = inp[:1, :]
        inp_tokens = inp[1:, :]
        inp_val = torch.cat([self.embedding_start(sos), self.embedding_encoder(inp_tokens)], dim=0) * math.sqrt(
            self.d_tf)  # from index to embeddings [513, 256] self.embedding_start(sos), here we use prepending img_encoding.permute(1, 0, 2)
        inp_posn = repeat(self.pos_embedding(inp_posn), 't pos_d -> t bs pos_d',
                          bs=bs)  # from coordinates to embeddings [512, pos_dim] excluding the last
        tgt_posn = repeat(self.pos_embedding(tgt_posn), 't pos_d -> t bs pos_d',
                          bs=bs)  # from coordinates to embeddings [512, pos_dim] excluding the start
        inp = torch.cat([inp_val, inp_posn, tgt_posn], dim=-1)
        if img_encoding is not None:
            cca_feature = self.cross_attention(x=inp.permute(1, 0, 2), context=img_encoding, mask=None)
            inp = torch.cat([inp, cca_feature.permute(1, 0, 2)], dim=-1) # normal

        # fusion
        inp = rearrange(inp, 't bs d -> (t bs) d')
        inp = rearrange(self.fuse_linear(inp), '(t bs) d -> t bs d', t=seq_len, bs=bs)

        src_mask = self.generate_square_subsequent_mask(seq_len, device)

        outp = self.forward_transformer(inp, src_mask=src_mask)

        outp = self.dec_linear(outp)  # torch.Size([512, 10, 768 -> 512])

        return outp


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, dropout=None, to_original_dimension=True):
        h = self.heads

        q = self.to_q(x)

        if context is not None:
            if torch.isnan(context).any():
                import pdb;
                pdb.set_trace()

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if torch.isnan(sim).any():
            import pdb;
            pdb.set_trace()

        if mask:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        if dropout:
            attn = F.dropout(attn, p=dropout, training=True)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if to_original_dimension:
            return self.to_out(out)
        else:
            return out
