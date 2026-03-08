import torch
import torch.nn as nn
from src.model.embedding import TransformerEmbedding
from src.model.encoder import Encoder
from src.model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,          # 中文词表大小
        tgt_vocab_size,          # 英文词表大小
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        # 独立的 Embedding：Encoder 用中文词表，Decoder 用英文词表
        self.src_emb = TransformerEmbedding(src_vocab_size, d_model, max_seq_len, dropout)
        self.tgt_emb = TransformerEmbedding(tgt_vocab_size, d_model, max_seq_len, dropout)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_decoder_layers, dropout)

        # 输出层只预测英文 token
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # (batch, 1, 1, src_len)
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        batch, tgt_len = tgt.size()
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        return pad_mask & causal_mask

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        return self.encoder(self.src_emb(src), src_mask), src_mask

    def decode(self, tgt, enc_out, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        return self.decoder(self.tgt_emb(tgt), enc_out, tgt_mask, src_mask)

    def forward(self, src, tgt):
        enc_out, src_mask = self.encode(src)
        dec_out = self.decode(tgt, enc_out, src_mask)
        return self.fc_out(dec_out)  # (batch, tgt_len, tgt_vocab_size)
