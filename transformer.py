# -*- Coding: utf-8 -*-
# @Time     : 2/24/2023 4:50 PM
# @Author   : Linqi Xiao
# @Software : PyCharm
# @Version  : python 3.10
# @Description :

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        """

        :param vocab_size:
        :param d_model: The dimensionality of input and output
        """
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):  # position
            for i in range(0, d_model, 2):  # dimension
                pe[pos, i] = math.sin(pos / 10000 ** (2 * i / d_model))
                pe[pos, i + 1] = math.cos(pos / 10000 ** (2 * (i + 1) / d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x):
        # make embeddings relatively larger. This means the original meaning in the embedding vector won’t be lost when we add them together
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        """
        Batch_size * seq_len * d_model -> batch_size * num_heads * seq_len * (d_model // num_heads)
        :param num_heads:
        :param d_model: The dimensionality of input and output
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        # self.d_model = d_model
        self.d_k = d_model // num_heads   # head dim
        self.num_heads = num_heads
        self.q_linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.v_linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.k_linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        In the case of the Encoder, V, K and G will simply be identical copies of the embedding vector (plus positional encoding).
        They will have the dimensions (Batch_size * seq_len * d_model).
        :param q: query
        :param k: key
        :param v: value
        :param mask:
        :return:
        """
        batch_size = q.shape(0)
        k = self.k_linear(k).view(batch_size, self.num_heads, -1, self.d_k)
        q = self.q_linear(q).view(batch_size, self.num_heads, -1, self.d_k)
        v = self.v_linear(v).view(batch_size, self.num_heads, -1, self.d_k)
        # calculating attention
        dots = torch.matmul(q, k.transpose(dim0=-1, dim1=-2))
        scale = dots / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scale = scale.masked_fill(mask == 0, -1e9)
        softmax = F.softmax(scale, dim=-1)
        dropout = self.dropout(softmax)
        attn = torch.matmul(dropout, v)
        return attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.feedforward(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm = Norm(d_model)
        self.attn = MultiHeadAttention(num_heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_norm = self.norm(x)
        x = x + self.dropout(self.attn(x_norm, x_norm, x_norm, mask))
        x_norm = self.norm(x)
        x = x + self.dropout(self.ff(x_norm))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm = Norm(d_model)
        self.attn = MultiHeadAttention(num_heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x_norm = self.norm(x)
        x = x + self.dropout(self.attn(x_norm, x_norm, x_norm, trg_mask))
        x_norm = self.norm(x)
        x = x + self.dropout(self.attn(x_norm, e_outputs, e_outputs, src_mask))
        x_norm = self.norm(x)
        x = x + self.dropout(self.ff(x_norm))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_blocks, num_heads):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        self.embed = Embedder(vocab_size=vocab_size, d_model=d_model)
        self.pe = PositionalEncoder(d_model=d_model)
        self.layers = get_clones(EncoderLayer(d_model=d_model, num_heads=num_heads), N=num_blocks)
        self.norm = Norm(d_model=d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.num_blocks):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_blocks, num_heads):
        super(Decoder, self).__init__()
        self.num_blocks = num_blocks
        self.embed = Embedder(vocab_size=vocab_size, d_model=d_model)
        self.pe = PositionalEncoder(d_model=d_model)
        self.layers = get_clones(EncoderLayer(d_model=d_model, num_heads=num_heads), N=num_blocks)
        self.norm = Norm(d_model=d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)












