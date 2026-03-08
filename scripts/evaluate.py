"""
P4-3 & P4-4: 在验证集上评估 BLEU，并演示单句翻译。

用法:
    python scripts/evaluate.py
    python scripts/evaluate.py --method greedy
    python scripts/evaluate.py --samples 200
"""
import os
import sys
import json
import yaml
import argparse
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
from src.utils.checkpoint import load_checkpoint
from src.utils.inference import greedy_decode, beam_search
from src.utils.bleu import evaluate_bleu


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def translate(text, model, zh_tok, en_tok, device, method="beam", beam_size=4, max_len=128):
    ids = zh_tok.encode(text)
    src = torch.tensor([ids], dtype=torch.long)
    bos_id, eos_id = en_tok.tok.token_to_id("[BOS]"), en_tok.tok.token_to_id("[EOS]")

    if method == "beam":
        out_ids = beam_search(model, src, max_len, bos_id, eos_id, device, beam_size)
    else:
        out_ids = greedy_decode(model, src, max_len, bos_id, eos_id, device)

    out_ids = [t for t in out_ids if t not in (0, bos_id, eos_id)]
    return en_tok.decode(out_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",  default="beam", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--samples",   type=int, default=None,
                        help="评估多少条验证集（None=全部，慢；建议先用200）")
    parser.add_argument("--ckpt", default="checkpoints/debug/best.pt")
    args = parser.parse_args()

    with open("configs/debug.yaml") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    mcfg = cfg["model"]
    dcfg = cfg["data"]

    # 加载两个独立 tokenizer
    zh_tok = ZhTokenizer(dcfg["src_vocab_path"])
    en_tok = EnTokenizer(dcfg["tgt_vocab_path"])
    print(f"ZH vocab: {zh_tok.vocab_size}  |  EN vocab: {en_tok.vocab_size}")

    # 加载模型
    model = Transformer(
        src_vocab_size=zh_tok.vocab_size,
        tgt_vocab_size=en_tok.vocab_size,
        d_model=mcfg["d_model"],
        num_heads=mcfg["num_heads"],
        num_encoder_layers=mcfg["num_encoder_layers"],
        num_decoder_layers=mcfg["num_decoder_layers"],
        d_ff=mcfg["d_ff"],
        max_seq_len=mcfg["max_seq_len"],
        dropout=0.0,   # 推理时关闭 dropout
    ).to(device)

    load_checkpoint(args.ckpt, model, device=device)

    # ── 单句演示 ──────────────────────────────────────────
    demos = [
        "这是一个机器翻译的例子。",
        "我爱机器学习。",
        "今天天气很好。",
        "他们在公园里散步。",
    ]
    print(f"\n{'='*60}")
    print(f"单句翻译演示（method={args.method}）")
    print(f"{'='*60}")
    for zh in demos:
        en = translate(zh, model, zh_tok, en_tok, device,
                       method=args.method, beam_size=args.beam_size)
        print(f"ZH: {zh}")
        print(f"EN: {en}\n")

    # ── 验证集 BLEU ────────────────────────────────────────
    with open(os.path.join(dcfg["val_path"], "src.json")) as f:
        val_src = json.load(f)
    with open(os.path.join(dcfg["val_path"], "tgt.json")) as f:
        val_tgt = json.load(f)

    val_loader = make_dataloader(
        val_src, val_tgt,
        batch_size=1,
        max_seq_len=mcfg["max_seq_len"],
        shuffle=False,
    )

    n = args.samples or len(val_src)
    print(f"{'='*60}")
    print(f"计算 BLEU（method={args.method}, samples={n}）...")
    print(f"{'='*60}")

    bos_id = en_tok.tok.token_to_id("[BOS]")
    eos_id = en_tok.tok.token_to_id("[EOS]")

    bleu, hyps, refs = evaluate_bleu(
        model, val_loader, en_tok, device,
        max_len=mcfg["max_seq_len"],
        bos_id=bos_id, eos_id=eos_id,
        method=args.method, beam_size=args.beam_size,
        max_samples=args.samples,
    )

    print(f"\nBLEU score: {bleu:.2f}")
    print(f"\n前5条对比：")
    for i in range(min(5, len(hyps))):
        print(f"  REF: {refs[i]}")
        print(f"  HYP: {hyps[i]}")
        print()


if __name__ == "__main__":
    main()
