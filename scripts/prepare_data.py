"""
P1-1 ~ P1-6: 下载 opus-100 en-zh，分别训练 ZH / EN 独立 BPE tokenizer，
编码数据并保存。

用法:
    python scripts/prepare_data.py --debug          # 10k 子集，vocab=8000
    python scripts/prepare_data.py --full           # 全量数据，vocab=32000
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

from datasets import load_dataset
from src.data.tokenizer import ZhTokenizer, EnTokenizer


def clean_pair(zh, en, max_len=150, min_len=3):
    zh, en = zh.strip(), en.strip()
    if not zh or not en:
        return None
    if len(zh) > max_len or len(en.split()) > max_len:
        return None
    if len(zh) < min_len or len(en.split()) < min_len:
        return None
    return zh, en


def prepare_debug(n=10000, src_vocab_size=8000, tgt_vocab_size=8000,
                  out_dir="data/processed/debug_10k", vocab_dir="data/vocab",
                  cache_dir="data/raw/hf_cache"):

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Loading Helsinki-NLP/opus-100 en-zh -> cache: {cache_dir}")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train",
                      cache_dir=cache_dir)

    zh_texts, en_texts = [], []
    for item in ds:
        pair = clean_pair(item["translation"]["zh"], item["translation"]["en"])
        if pair:
            zh_texts.append(pair[0])
            en_texts.append(pair[1])
        if len(zh_texts) >= n:
            break

    print(f"Collected {len(zh_texts)} pairs after cleaning.")

    split = int(len(zh_texts) * 0.9)
    splits = {
        "train": (zh_texts[:split], en_texts[:split]),
        "val":   (zh_texts[split:], en_texts[split:]),
    }

    # 分别训练两个 tokenizer，只用各自语言的文本
    print(f"\nTraining ZH tokenizer (vocab_size={src_vocab_size}) on ZH text only ...")
    zh_tok = ZhTokenizer()
    zh_tok.train(
        splits["train"][0],
        vocab_size=src_vocab_size,
        save_path=os.path.join(vocab_dir, "bpe_zh_debug.json"),
    )

    print(f"\nTraining EN tokenizer (vocab_size={tgt_vocab_size}) on EN text only ...")
    en_tok = EnTokenizer()
    en_tok.train(
        splits["train"][1],
        vocab_size=tgt_vocab_size,
        save_path=os.path.join(vocab_dir, "bpe_en_debug.json"),
    )

    # 分别编码并保存
    for split_name, (zh, en) in splits.items():
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        src_ids = [zh_tok.encode(s) for s in zh]
        tgt_ids = [en_tok.encode(s) for s in en]

        with open(os.path.join(split_dir, "src.json"), "w") as f:
            json.dump(src_ids, f)
        with open(os.path.join(split_dir, "tgt.json"), "w") as f:
            json.dump(tgt_ids, f)

        print(f"Saved {len(src_ids)} {split_name} pairs -> {split_dir}")

    print("\nData preparation complete!")
    print(f"  ZH tokenizer : {vocab_dir}/bpe_zh_debug.json  (vocab={zh_tok.vocab_size})")
    print(f"  EN tokenizer : {vocab_dir}/bpe_en_debug.json  (vocab={en_tok.vocab_size})")
    print(f"  Train/Val    : {out_dir}/train | {out_dir}/val")


def prepare_full(src_vocab_size=32000, tgt_vocab_size=32000,
                 out_dir="data/processed/full", vocab_dir="data/vocab",
                 cache_dir="data/raw/hf_cache"):
    """下载 opus-100 en-zh 全量训练集，训练 vocab=32000 的 BPE tokenizer。
    原始 HF 数据集缓存到 data/raw/hf_cache/（项目目录，可持久化）。
    """
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading Helsinki-NLP/opus-100 en-zh (train split) -> cache: {cache_dir}")
    ds_train = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train",
                            cache_dir=cache_dir)
    print(f"  Raw train size: {len(ds_train):,}")

    print(f"Loading Helsinki-NLP/opus-100 en-zh (validation split) -> cache: {cache_dir}")
    ds_val = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="validation",
                          cache_dir=cache_dir)
    print(f"  Raw val size: {len(ds_val):,}")

    def collect(ds, desc=""):
        zh_list, en_list = [], []
        for item in ds:
            pair = clean_pair(item["translation"]["zh"], item["translation"]["en"])
            if pair:
                zh_list.append(pair[0])
                en_list.append(pair[1])
        print(f"  {desc}: {len(zh_list):,} pairs after cleaning.")
        return zh_list, en_list

    train_zh, train_en = collect(ds_train, "Train")
    val_zh,   val_en   = collect(ds_val,   "Val")

    os.makedirs(vocab_dir, exist_ok=True)

    print(f"\nTraining ZH tokenizer (vocab_size={src_vocab_size}) ...")
    zh_tok = ZhTokenizer()
    zh_tok.train(
        train_zh,
        vocab_size=src_vocab_size,
        save_path=os.path.join(vocab_dir, "bpe_zh_full.json"),
    )

    print(f"\nTraining EN tokenizer (vocab_size={tgt_vocab_size}) ...")
    en_tok = EnTokenizer()
    en_tok.train(
        train_en,
        vocab_size=tgt_vocab_size,
        save_path=os.path.join(vocab_dir, "bpe_en_full.json"),
    )

    splits = {
        "train": (train_zh, train_en),
        "val":   (val_zh, val_en),
    }

    for split_name, (zh, en) in splits.items():
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\nEncoding {split_name} set ({len(zh):,} pairs) ...")
        src_ids = [zh_tok.encode(s) for s in zh]
        tgt_ids = [en_tok.encode(s) for s in en]

        with open(os.path.join(split_dir, "src.json"), "w") as f:
            json.dump(src_ids, f)
        with open(os.path.join(split_dir, "tgt.json"), "w") as f:
            json.dump(tgt_ids, f)

        print(f"  Saved {len(src_ids):,} {split_name} pairs -> {split_dir}")

    print("\nFull data preparation complete!")
    print(f"  ZH tokenizer : {vocab_dir}/bpe_zh_full.json  (vocab={zh_tok.vocab_size})")
    print(f"  EN tokenizer : {vocab_dir}/bpe_en_full.json  (vocab={en_tok.vocab_size})")
    print(f"  Train        : {out_dir}/train")
    print(f"  Val          : {out_dir}/val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Prepare 10k debug subset")
    parser.add_argument("--full", action="store_true", help="Prepare full dataset (vocab=32000)")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--src_vocab_size", type=int, default=8000)
    parser.add_argument("--tgt_vocab_size", type=int, default=8000)
    args = parser.parse_args()

    if args.debug:
        prepare_debug(n=args.n,
                      src_vocab_size=args.src_vocab_size,
                      tgt_vocab_size=args.tgt_vocab_size)
    elif args.full:
        prepare_full()
    else:
        print("Use --debug for 10k subset, or --full for full training data.")
