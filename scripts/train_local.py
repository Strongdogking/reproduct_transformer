"""
本地调试训练脚本（debug 配置 + 10k 数据，ZH/EN 分离词表）

用法:
    python scripts/train_local.py
"""
import os
import sys
import json
import yaml
import torch

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

from src.model.transformer import Transformer
from src.data.tokenizer import ZhTokenizer, EnTokenizer
from src.data.dataset import make_dataloader
from src.train.loss import LabelSmoothingLoss
from src.train.scheduler import WarmupScheduler
from src.train.trainer import Trainer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(split_dir):
    with open(os.path.join(split_dir, "src.json")) as f:
        src = json.load(f)
    with open(os.path.join(split_dir, "tgt.json")) as f:
        tgt = json.load(f)
    return src, tgt


def main():
    with open("configs/debug.yaml") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")

    train_src, train_tgt = load_data(cfg["data"]["train_path"])
    val_src,   val_tgt   = load_data(cfg["data"]["val_path"])
    print(f"Train pairs: {len(train_src)} | Val pairs: {len(val_src)}")

    mcfg = cfg["model"]
    tcfg = cfg["train"]
    dcfg = cfg["data"]

    # 从 tokenizer 读取实际 vocab size（可能小于 config 里的上限值）
    zh_tok = ZhTokenizer(dcfg["src_vocab_path"])
    en_tok = EnTokenizer(dcfg["tgt_vocab_path"])
    actual_src_vocab = zh_tok.vocab_size
    actual_tgt_vocab = en_tok.vocab_size
    print(f"ZH vocab: {actual_src_vocab}  |  EN vocab: {actual_tgt_vocab}")

    train_loader = make_dataloader(
        train_src, train_tgt,
        batch_size=tcfg["batch_size"],
        max_seq_len=mcfg["max_seq_len"],
    )
    val_loader = make_dataloader(
        val_src, val_tgt,
        batch_size=tcfg["batch_size"],
        max_seq_len=mcfg["max_seq_len"],
        shuffle=False,
    )

    model = Transformer(
        src_vocab_size=actual_src_vocab,
        tgt_vocab_size=actual_tgt_vocab,
        d_model=mcfg["d_model"],
        num_heads=mcfg["num_heads"],
        num_encoder_layers=mcfg["num_encoder_layers"],
        num_decoder_layers=mcfg["num_decoder_layers"],
        d_ff=mcfg["d_ff"],
        max_seq_len=mcfg["max_seq_len"],
        dropout=mcfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = LabelSmoothingLoss(
        vocab_size=actual_tgt_vocab,   # loss 只在英文词表上计算
        smoothing=tcfg["label_smoothing"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, mcfg["d_model"], tcfg["warmup_steps"])

    os.makedirs(tcfg["checkpoint_dir"], exist_ok=True)

    trainer = Trainer(model, optimizer, scheduler, criterion, device, cfg)
    trainer.fit(train_loader, val_loader, num_epochs=tcfg["max_epochs"])
    print("\nDone!")


if __name__ == "__main__":
    main()
