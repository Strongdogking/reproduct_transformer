# Transformer Architecture — Chinese to English Translation

> 复现论文：Attention Is All You Need (Vaswani et al., 2017)
> 任务：中文 → 英文翻译

---

## 1. 整体架构

```
Input (ZH)                          Output (EN)
    │                                    │
[Tokenizer / BPE]              [Tokenizer / BPE]
    │                                    │
[Embedding + Positional Encoding]   [Embedding + Positional Encoding]
    │                                    │
┌───▼──────────────┐          ┌─────────▼──────────────┐
│   ENCODER × N    │          │    DECODER × N          │
│  ┌─────────────┐ │          │  ┌─────────────────┐   │
│  │ Multi-Head  │ │          │  │ Masked Multi-Head│   │
│  │ Self-Attn   │ │          │  │ Self-Attn        │   │
│  ├─────────────┤ │          │  ├─────────────────┤   │
│  │ Add & Norm  │ │──────────►  │ Cross-Attention  │   │
│  ├─────────────┤ │          │  ├─────────────────┤   │
│  │ Feed-Forward│ │          │  │ Add & Norm       │   │
│  ├─────────────┤ │          │  ├─────────────────┤   │
│  │ Add & Norm  │ │          │  │ Feed-Forward     │   │
│  └─────────────┘ │          │  ├─────────────────┤   │
└──────────────────┘          │  │ Add & Norm       │   │
                              │  └─────────────────┘   │
                              └────────────────────────┘
                                          │
                                  [Linear + Softmax]
                                          │
                                    Output Token
```

---

## 2. 超参数（论文默认值）

| 参数 | 值 |
|---|---|
| `d_model` | 512 |
| `num_heads` | 8 |
| `d_k = d_v` | 64 |
| `d_ff` (FFN 内层) | 2048 |
| `num_encoder_layers` | 6 |
| `num_decoder_layers` | 6 |
| `dropout` | 0.1 |
| `max_seq_len` | 200（云端实际配置；理论上限 512）|
| `vocab_size` ZH | 32000（独立 BPE） |
| `vocab_size` EN | 32000（独立 BPE） |

---

## 3. 核心模块

### 3.1 Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

### 3.2 Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)
```

### 3.3 Position-wise Feed-Forward Network
```
FFN(x) = max(0, x*W_1 + b_1) * W_2 + b_2
```

### 3.4 Positional Encoding
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 3.5 Label Smoothing
- epsilon = 0.1（论文中使用）

### 3.6 Learning Rate Schedule（Warm-up）
```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
warmup_steps = 4000
```

---

## 4. 数据集

| 项目 | 内容 |
|---|---|
| 数据集 | Helsinki-NLP/opus-100 en-zh |
| 分词 | BPE，ZH/EN **独立**词表（非共享） |
| 词表大小 | ZH: 32000 / EN: 32000 |
| 训练集规模 | 885,196 句对（清洗后）|
| 验证集 | opus-100 validation split，1928 句对 |
| 评估指标 | BLEU score（sacrebleu） |

---

## 5. 项目目录结构

```
reproduct_transformer/
├── architect.md          # 架构文档（本文件）
├── plan.md               # 开发计划
├── status.md             # 当前进度
├── action_guide.md       # 工作流指南
│
├── data/
│   ├── raw/              # 原始语料
│   ├── processed/        # 预处理后数据
│   └── vocab/            # BPE 词表文件
│
├── src/
│   ├── data/
│   │   ├── dataset.py    # Dataset & DataLoader
│   │   └── tokenizer.py  # BPE 分词器封装
│   ├── model/
│   │   ├── attention.py  # Multi-Head Attention
│   │   ├── encoder.py    # Encoder Layer & Stack
│   │   ├── decoder.py    # Decoder Layer & Stack
│   │   ├── embedding.py  # Token + Positional Embedding
│   │   ├── ffn.py        # Feed-Forward Network
│   │   └── transformer.py# 完整 Transformer 模型
│   ├── train/
│   │   ├── trainer.py    # 训练循环
│   │   ├── scheduler.py  # Warmup LR Scheduler
│   │   └── loss.py       # Label Smoothing Loss
│   └── utils/
│       ├── bleu.py       # BLEU 评估
│       └── checkpoint.py # 模型保存与加载
│
├── configs/
│   ├── debug.yaml        # 本地调试（d_model=128, 2层, 10k数据）
│   ├── base.yaml         # 全量基础配置（d_model=512, 6层）
│   └── cloud.yaml        # 云端训练配置（batch=64, AMP, grad_accum=2）
│
├── scripts/
│   ├── prepare_data.py   # 数据准备（--debug 10k / --full 全量）
│   ├── train_local.py    # 本地调试训练脚本
│   ├── train_cloud.py    # 云端 GPU 训练脚本（支持 --resume）
│   └── evaluate.py       # 推理评估（greedy / beam search）
```

---

## 6. 本地 vs 云端部署差异

| 环境 | 设备 | 用途 |
|---|---|---|
| 本地 Mac（Apple Silicon） | CPU / MPS | 代码开发、小规模调试（debug.yaml）|
| 云端（RTX 4090 24G） | CUDA GPU | 完整数据集训练（cloud.yaml）|

- 本地调试使用小数据子集（如 10k 句对）
- `configs/base.yaml` 中通过 `device: auto` 自动选择设备
- 云端使用 `torch.cuda`，支持混合精度训练（AMP）
