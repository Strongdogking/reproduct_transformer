# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A from-scratch PyTorch reproduction of "Attention Is All You Need" (Vaswani et al., 2017), implementing a Chinese→English translation Transformer. The conda environment is `base` .

## Common Commands

All scripts must be run from the **project root** directory.

```bash
# Run shape/unit tests (validates all module output shapes)
python scripts/test_shapes.py

# Prepare debug dataset (downloads opus-100 en-zh, trains BPE tokenizers, saves 10k pairs)
python scripts/prepare_data.py --debug

# Train locally with debug config (d_model=128, 2 layers, 10k pairs)
python scripts/train_local.py

# Evaluate with beam search (default); loads checkpoints/debug/best.pt
python scripts/evaluate.py
python scripts/evaluate.py --method greedy
python scripts/evaluate.py --samples 200          # faster partial eval
python scripts/evaluate.py --ckpt checkpoints/debug/best.pt
```

## Architecture

### Model (`src/model/`)

The full Transformer is assembled in `transformer.py`. Key design: **ZH and EN use separate independent embeddings and vocabularies** — not a shared vocab.

- `attention.py`: `ScaledDotProductAttention` and `MultiHeadAttention` (with Q/K/V projections)
- `embedding.py`: `TransformerEmbedding` = token embedding + sinusoidal positional encoding
- `encoder.py`: `EncoderLayer` (self-attn → Add&Norm → FFN → Add&Norm) + `Encoder` (N-layer stack)
- `decoder.py`: `DecoderLayer` (masked self-attn → cross-attn → FFN, each with Add&Norm) + `Decoder` stack
- `ffn.py`: `PositionwiseFeedForward` (two linear layers + ReLU)
- `transformer.py`: `Transformer` — separate `src_emb` (ZH) and `tgt_emb` (EN), encoder, decoder, final linear projection to `tgt_vocab_size`

### Data (`src/data/`)

- `tokenizer.py`: `ZhTokenizer` and `EnTokenizer` — both BPE-based but trained independently. Chinese text is pre-processed by inserting spaces around every CJK character (`tokenize_zh()`) before BPE, enabling character-level BPE for Chinese.
- `dataset.py`: `make_dataloader` — pads sequences, applies src/tgt masks

### Training (`src/train/`)

- `trainer.py`: `Trainer` — uses teacher forcing (feeds `tgt[:-1]`, predicts `tgt[1:]`), gradient clipping, saves `best.pt` on val loss improvement
- `scheduler.py`: `WarmupScheduler` — implements the paper's warmup LR formula: `lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))`
- `loss.py`: `LabelSmoothingLoss` (epsilon=0.1)

### Inference (`src/utils/`)

- `inference.py`: `greedy_decode` and `beam_search` (length-normalized log-prob scoring)
- `bleu.py`: sacrebleu integration
- `checkpoint.py`: `save_checkpoint` / `load_checkpoint`

## Configs

| Config | Use case | Key params |
|---|---|---|
| `configs/debug.yaml` | Local dev/debug | d_model=128, 2 layers, vocab=8000, 10k pairs |
| `configs/base.yaml` | Cloud/full training | d_model=512, 6 layers, vocab=32000 |

**Important**: `train_local.py` reads actual vocab size from the loaded tokenizer (not the config value) and passes that to `Transformer(src_vocab_size=..., tgt_vocab_size=...)`. Always do the same when building the model.

## Project Tracking

Progress is tracked in `conductor/`:
- `plan.md`: task checklist by phase (P0–P6)
- `status.md`: current phase status and next actions
- `architect.md`: architecture reference with hyperparameters
- `action_guide.md`: workflow protocol for each session

**Current status**: Phases 0–4 complete (local smoke test done, debug BLEU: greedy=0.62, beam=1.22). Phase 5 (cloud training) not yet started.

## Key Design Decisions

- ZH/EN tokenizers are **separate** (not shared vocab) — Encoder uses ZH-only BPE, Decoder uses EN-only BPE
- Chinese pre-tokenization spaces CJK characters to enable character-level BPE
- Device selection: `cuda` → `mps` → `cpu` (auto-detected in training scripts)
- Data format: encoded token IDs stored as `src.json` / `tgt.json` (lists of lists) under `data/processed/`
