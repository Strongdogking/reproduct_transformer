"""独立的 ZH / EN BPE tokenizer。中英分离词表，互不干扰。"""
import os
import unicodedata
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def tokenize_zh(text):
    """在每个 CJK 字符两侧插入空格，使 BPE 能在字符级别处理中文。

    "我爱NLP！" -> "我 爱 N L P ！"
    """
    result = []
    for ch in text:
        cp = ord(ch)
        if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2E80 <= cp <= 0x2EFF)
            or (0x3000 <= cp <= 0x303F)
            or (0xFF00 <= cp <= 0xFFEF)
        ):
            result.append(f" {ch} ")
        else:
            result.append(ch)
    return "".join(result).strip()


def _build_tokenizer(texts, vocab_size, save_path=None):
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, BOS_ID), (EOS_TOKEN, EOS_ID)],
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tokenizer.save(save_path)
    return tokenizer


class ZhTokenizer:
    """专门处理中文的 BPE tokenizer。训练前自动字符化。"""

    def __init__(self, path=None):
        self.tok = Tokenizer.from_file(path) if path and os.path.exists(path) else None

    def train(self, zh_texts, vocab_size, save_path=None):
        charified = [tokenize_zh(t) for t in zh_texts]
        self.tok = _build_tokenizer(charified, vocab_size, save_path)
        if save_path:
            print(f"ZH tokenizer saved: {save_path}  (vocab={self.tok.get_vocab_size()})")

    def encode(self, text):
        return self.tok.encode(tokenize_zh(text)).ids

    def decode(self, ids):
        return self.tok.decode(ids)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()


class EnTokenizer:
    """专门处理英文的 BPE tokenizer。按空格切词后走 BPE。"""

    def __init__(self, path=None):
        self.tok = Tokenizer.from_file(path) if path and os.path.exists(path) else None

    def train(self, en_texts, vocab_size, save_path=None):
        self.tok = _build_tokenizer(en_texts, vocab_size, save_path)
        if save_path:
            print(f"EN tokenizer saved: {save_path}  (vocab={self.tok.get_vocab_size()})")

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(ids)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()
