"""
P2-9: 单元测试 — 验证所有模块输出 shape 正确
用法: conda run -n ds_env python scripts/test_shapes.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.attention import ScaledDotProductAttention, MultiHeadAttention
from src.model.ffn import PositionwiseFeedForward
from src.model.embedding import TransformerEmbedding
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.transformer import Transformer

B, S, T, D, H = 2, 10, 8, 128, 4          # batch, src_len, tgt_len, d_model, heads
SV, TV = 8000, 8000                        # src_vocab(ZH), tgt_vocab(EN)

def check(name, tensor, expected):
    assert tensor.shape == expected, f"{name}: expected {expected}, got {tensor.shape}"
    print(f"  OK  {name}: {tuple(tensor.shape)}")

print("=== Shape Tests ===")

# Scaled dot-product attention
q = k = v = torch.randn(B, H, S, D // H)
attn = ScaledDotProductAttention()
out, w = attn(q, k, v)
check("ScaledDotProductAttention output", out, (B, H, S, D // H))
check("ScaledDotProductAttention weights", w, (B, H, S, S))

# Multi-head attention
x = torch.randn(B, S, D)
mha = MultiHeadAttention(D, H)
out = mha(x, x, x)
check("MultiHeadAttention", out, (B, S, D))

# FFN
ffn = PositionwiseFeedForward(D, D * 4)
out = ffn(x)
check("PositionwiseFeedForward", out, (B, S, D))

# Encoder embedding (ZH vocab)
src_emb = TransformerEmbedding(SV, D, max_seq_len=64)
src_ids = torch.randint(1, SV, (B, S))
src_x = src_emb(src_ids)
check("Encoder TransformerEmbedding (ZH)", src_x, (B, S, D))

# Encoder
enc = Encoder(D, H, D * 4, num_layers=2)
enc_out = enc(src_x)
check("Encoder", enc_out, (B, S, D))

# Decoder embedding (EN vocab)
tgt_emb = TransformerEmbedding(TV, D, max_seq_len=64)
tgt_ids = torch.randint(1, TV, (B, T))
tgt_x = tgt_emb(tgt_ids)
dec = Decoder(D, H, D * 4, num_layers=2)
dec_out = dec(tgt_x, enc_out)
check("Decoder", dec_out, (B, T, D))

# Full Transformer — 分离词表
model = Transformer(src_vocab_size=SV, tgt_vocab_size=TV,
                    d_model=D, num_heads=H, num_encoder_layers=2,
                    num_decoder_layers=2, d_ff=D*4, max_seq_len=64)
src = torch.randint(1, SV, (B, S))
tgt = torch.randint(1, TV, (B, T))
logits = model(src, tgt)
check("Transformer logits (tgt_vocab)", logits, (B, T, TV))

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel params (debug size, 分离词表): {n_params:,}")
print("\nAll shape tests passed!")
