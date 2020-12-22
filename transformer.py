import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
#from utils import *

import torch
import torch.nn as nn
import numpy as np

import torch
np.random.seed(1337)
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ResidualNorm (nn.Module):
    def __init__ (self, size, dropout):
        super(ResidualNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward (self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MLP (nn.Module):
    def __init__(self, model_depth, ff_depth, dropout):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(model_depth, ff_depth)
        self.w2 = nn.Linear(ff_depth, model_depth)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

################################################################
# attention

class MultiHeadAttention (nn.Module):
    def __init__ (self, n_heads, model_depth, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dk = model_depth//n_heads
        self.WQ = nn.Linear(model_depth, model_depth, bias=bias)
        self.WK = nn.Linear(model_depth, model_depth, bias=bias)
        self.WV = nn.Linear(model_depth, model_depth, bias=bias)
        self.WO = nn.Linear(model_depth, model_depth, bias=bias)

    def forward (self, x, kv, mask):
        batch_size = x.size(0)
        Q = self.WQ(x ).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)
        K = self.WK(kv).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)
        V = self.WV(kv).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)

        x = attention(Q, K, V, mask=mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads*self.dk)
        return self.WO(x)

def attention (Q,K,V, mask=None):
    dk = Q.size(-1)
    T = (Q @ K.transpose(-2, -1))/math.sqrt(dk)
    if mask is not None:
        T = T.masked_fill_(mask.unsqueeze(1)==0, -1e9)
    T = F.softmax(T, dim=-1)
    return T @ V


################################################################
# encoder

class Encoder (nn.Module):
    def __init__ (self, n_layers, n_heads, model_depth, ff_depth, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_heads, model_depth, ff_depth, dropout) for i in range(n_layers)])
        self.lnorm = LayerNorm(model_depth)

    def forward (self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.lnorm(x)

class EncoderLayer (nn.Module):
    def __init__ (self, n_heads, model_depth, ff_depth, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, model_depth)
        self.resnorm1 = ResidualNorm(model_depth, dropout)
        self.ff = MLP(model_depth, ff_depth, dropout)
        self.resnorm2 = ResidualNorm(model_depth, dropout)

    def forward (self, x, mask):
        x = self.resnorm1(x, lambda arg: self.self_attn(arg,arg,mask))
        x = self.resnorm2(x, self.ff)
        return x

################################################################
# decoder

class Decoder (nn.Module):
    def __init__ (self, n_layers, n_heads, model_depth, ff_depth, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(n_heads, model_depth, ff_depth, dropout) for i in range(n_layers)])
        self.lnorm = LayerNorm(model_depth)

    def forward (self, x, src_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, src_out, src_mask, tgt_mask)
        return self.lnorm(x)

class DecoderLayer (nn.Module):
    def __init__ (self, n_heads, model_depth, ff_depth, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, model_depth)
        self.resnorm1 = ResidualNorm(model_depth, dropout)
        self.enc_attn = MultiHeadAttention(n_heads, model_depth)
        self.resnorm2 = ResidualNorm(model_depth, dropout)
        self.ff = MLP(model_depth, ff_depth, dropout)
        self.resnorm3 = ResidualNorm(model_depth, dropout)

    def forward (self, x, src_out, src_mask, tgt_mask):
        x = self.resnorm1(x, lambda arg: self.self_attn(arg,arg, tgt_mask))
        x = self.resnorm2(x, lambda arg: self.enc_attn(arg,src_out, src_mask))
        x = self.resnorm3(x, self.ff)
        return x

################################################################
# embedder

class Embedding(nn.Module):
    def __init__(self, vocab_size, model_depth):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, model_depth)
        self.model_depth = model_depth
        self.positional = PositionalEncoding(model_depth)

    def forward(self, x):
        emb = self.lut(x) * math.sqrt(self.model_depth)
        return self.positional(emb)

class PositionalEncoding(nn.Module):
    def __init__(self, model_depth, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, model_depth)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, model_depth, 2) *
                             -(math.log(10000.0) / model_depth))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

################################################################
# transformer

class Generator (nn.Module):
    def __init__(self, model_depth, vocab_size):
        super(Generator, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.ff(x), dim=-1)

class Transformer (nn.Module):
    def __init__ (self, vocab_size, n_layers, n_heads, model_depth, ff_depth, dropout):
        super(Transformer, self).__init__()
        self.model_depth = model_depth
        self.encoder   = Encoder(n_layers, n_heads, model_depth, ff_depth, dropout)
        self.decoder   = Decoder(n_layers, n_heads, model_depth, ff_depth, dropout)
        if vocab_size is not None:
            if type(vocab_size) is int:
                self.set_vocab_size(vocab_size)
            else:
                self.set_vocab_size(vocab_size[0], vocab_size[1])

    def set_vocab_size (self, src_vocab_size, tgt_vocab_size=None):
        if tgt_vocab_size is None:
            self.src_embedder = Embedding(src_vocab_size, self.model_depth)
            self.tgt_embedder = self.src_embedder
            self.generator = Generator(self.model_depth, src_vocab_size)
        else:
            self.src_embedder = Embedding(src_vocab_size, self.model_depth)
            self.tgt_embedder = Embedding(tgt_vocab_size, self.model_depth)
            self.generator = Generator(self.model_depth, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encoder(self.src_embedder(src), src_mask)
        dec_out = self.decoder(self.tgt_embedder(tgt), enc_out, src_mask, tgt_mask)
        return dec_out
