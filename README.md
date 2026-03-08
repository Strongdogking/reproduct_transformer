# Transformer 中英翻译复现

从零复现论文 **"Attention Is All You Need"**（Vaswani et al., 2017），实现中文 → 英文神经机器翻译。

---

## 项目简介

本项目使用 PyTorch 完整实现论文中描述的 Transformer 架构，不依赖任何高级封装库（如 Hugging Face Transformers），所有模块均从头手写，目标是深入理解 Transformer 的每个细节。

**核心特性：**
- 完整实现论文架构：Multi-Head Attention、Positional Encoding、Label Smoothing、Warmup LR Schedule
- 中英文独立 BPE 词表（非共享），中文字符级预分词
- 支持 Greedy Decode 与 Beam Search 推理
- 支持混合精度训练（AMP）+ 梯度累积
- TensorBoard 训练曲线可视化
- 本地调试（CPU/MPS）与云端训练（CUDA）双模式

---

## 架构

```
Input (ZH)                            Output (EN)
    │                                      │
[ZH BPE Tokenizer]              [EN BPE Tokenizer]
    │                                      │
[Embedding + Positional Encoding]  [Embedding + Positional Encoding]
    │                                      │
┌───▼──────────────────┐       ┌───────────▼──────────────────┐
│    Encoder × 6       │       │       Decoder × 6             │
│  ┌────────────────┐  │       │  ┌───────────────────────┐   │
│  │ Multi-Head     │  │       │  │ Masked Multi-Head     │   │
│  │ Self-Attention │  │       │  │ Self-Attention        │   │
│  ├────────────────┤  │       │  ├───────────────────────┤   │
│  │ Add & Norm     │  │──────►│  │ Cross-Attention       │   │
│  ├────────────────┤  │       │  ├───────────────────────┤   │
│  │ Feed-Forward   │  │       │  │ Feed-Forward          │   │
│  ├────────────────┤  │       │  ├───────────────────────┤   │
│  │ Add & Norm     │  │       │  │ Add & Norm (×3)       │   │
│  └────────────────┘  │       │  └───────────────────────┘   │
└──────────────────────┘       └──────────────────────────────┘
                                               │
                                       [Linear + Softmax]
                                               │
                                         Output Token
```

**超参数（Full Model）：**

| 参数 | 值 |
|---|---|
| d_model | 512 |
| num_heads | 8 |
| num_layers (enc/dec) | 6 / 6 |
| d_ff | 2048 |
| dropout | 0.1 |
| vocab_size (ZH / EN) | 32000 / 32000 |
| warmup_steps | 4000 |
| label_smoothing | 0.1 |
| 模型参数量 | ~93M |

---

## 项目结构

```
reproduct_transformer/
├── src/
│   ├── model/
│   │   ├── attention.py      # ScaledDotProductAttention + MultiHeadAttention
│   │   ├── encoder.py        # EncoderLayer + Encoder
│   │   ├── decoder.py        # DecoderLayer + Decoder
│   │   ├── embedding.py      # TransformerEmbedding（token + sinusoidal PE）
│   │   ├── ffn.py            # PositionwiseFeedForward
│   │   └── transformer.py    # 完整 Transformer
│   ├── data/
│   │   ├── tokenizer.py      # ZhTokenizer / EnTokenizer（独立 BPE）
│   │   └── dataset.py        # make_dataloader（padding + masking）
│   ├── train/
│   │   ├── trainer.py        # Trainer（AMP + 梯度累积 + TensorBoard）
│   │   ├── scheduler.py      # WarmupScheduler（论文公式）
│   │   └── loss.py           # LabelSmoothingLoss
│   └── utils/
│       ├── inference.py      # greedy_decode + beam_search
│       ├── bleu.py           # sacrebleu 集成
│       └── checkpoint.py     # save / load checkpoint
├── scripts/
│   ├── prepare_data.py       # 数据下载与预处理
│   ├── train_local.py        # 本地调试训练
│   ├── train_cloud.py        # 云端 GPU 训练（支持断点续训）
│   ├── evaluate.py           # BLEU 评估
│   └── test_shapes.py        # 模块 shape 单元测试
├── configs/
│   ├── debug.yaml            # 本地调试（d_model=128, 2层, 10k数据）
│   ├── base.yaml             # 全量基础配置
│   └── cloud.yaml            # 云端训练（batch=64, AMP, grad_accum=2）
└── conductor/                # 项目管理文档
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio
pip install tokenizers datasets sacrebleu pyyaml tensorboard
```

### 2. 数据准备

**本地调试（10k 句对，vocab=8000）：**
```bash
python scripts/prepare_data.py --debug
```

**云端全量训练（885k 句对，vocab=32000）：**
```bash
# 如需镜像加速（中国大陆）
export HF_ENDPOINT=https://hf-mirror.com

python scripts/prepare_data.py --full
```

数据来源：[Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)（en-zh）

### 3. 本地调试训练

```bash
python scripts/train_local.py
```

使用 `configs/debug.yaml`（d_model=128，2层，10k 数据），快速验证代码正确性。

### 4. 云端 GPU 训练

```bash
python scripts/train_cloud.py
# 断点续训
python scripts/train_cloud.py --resume checkpoints/cloud/best.pt
```

使用 `configs/cloud.yaml`（d_model=512，6层，AMP，batch 等效 128）。

### 5. 查看训练曲线

```bash
tensorboard --logdir checkpoints/cloud/tb_logs --port 6006
```

### 6. 评估

```bash
# Beam Search（默认）
python scripts/evaluate.py --ckpt checkpoints/cloud/best.pt

# Greedy Decode
python scripts/evaluate.py --ckpt checkpoints/cloud/best.pt --method greedy

# 快速评估（200 样本）
python scripts/evaluate.py --ckpt checkpoints/cloud/best.pt --samples 200
```

---

## 关键设计决策

### 中英文独立词表

本项目 Encoder（中文）和 Decoder（英文）使用**各自独立训练的 BPE 词表**，而非共享词表。

中文预分词：在每个 CJK 字符两侧插入空格，使 BPE 能在字符级别处理中文：
```python
"你好世界" → "你 好 世 界"
```

### AMP + 梯度累积

RTX 4090（24GB）上使用 AMP（fp16）+ 梯度累积（accum=2）实现等效 batch size=128，单步峰值显存约 11GB：

```yaml
batch_size: 64
grad_accum_steps: 2   # 等效 batch = 128
use_amp: true
```

### Attention Mask 精度修复

AMP 下 `float16` 最大约 ±65504，原始 `-1e9` 会溢出。已修复为：
```python
scores.masked_fill(mask == 0, float('-inf'))
```

### Warmup LR Schedule

严格按论文公式实现：
```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

---

## 复现结果

| 模型 | 数据 | 方法 | BLEU |
|---|---|---|---|
| Debug（d_model=128, 2层）| 9k 对，5 epoch | Greedy | 0.62 |
| Debug（d_model=128, 2层）| 9k 对，5 epoch | Beam Search (k=4) | 1.22 |
| Full（d_model=512, 6层）| 885k 对，训练中 | — | — |

> 论文原始结果：EN→DE newstest2014 BLEU=27.3（Base Model）

---

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)（Vaswani et al., 2017）
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)

---

## License

MIT License
