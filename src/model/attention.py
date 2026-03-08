import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch, heads, seq_len, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def _split_heads(self, x):
        # x: (batch, seq_len, d_model) -> (batch, heads, seq_len, d_k)
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        q = self._split_heads(self.w_q(q))
        k = self._split_heads(self.w_k(k))
        v = self._split_heads(self.w_v(v))

        out, attn = self.attention(q, k, v, mask)

        # (batch, heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch, _, seq_len, _ = out.size()
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.w_o(out)
