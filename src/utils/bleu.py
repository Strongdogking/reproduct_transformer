"""P4-3: 计算验证集 BLEU 分数（使用 sacrebleu）。"""
import re
import torch
import sacrebleu
from src.utils.inference import greedy_decode, beam_search


def detokenize_en(text):
    """BPE decode 后的简单后处理，恢复英文标点和缩写。"""
    # 去掉标点前的空格：word . → word.
    text = re.sub(r'\s+([.,!?;:\)])', r'\1', text)
    # 去掉左括号后的空格
    text = re.sub(r'(\()\s+', r'\1', text)
    # 修复英文缩写：don ' t → don't, I ' m → I'm
    text = re.sub(r"\s'\s*(s|t|re|ve|ll|d|m|S|T|Re|Ve|Ll|D|M)\b", r"'\1", text)
    # 去掉行首尾多余空格
    return text.strip()


@torch.no_grad()
def evaluate_bleu(model, dataloader, en_tokenizer, device,
                  max_len=128, bos_id=2, eos_id=3,
                  method="beam", beam_size=4, max_samples=None):
    """
    在 dataloader 上跑推理，计算 BLEU 分数。

    Args:
        model:         Transformer 模型
        dataloader:    val DataLoader（batch 内 src/tgt 已编码）
        en_tokenizer:  EN tokenizer，用于 decode 预测结果和参考答案
        method:        "greedy" 或 "beam"
        max_samples:   最多评估多少条（None = 全部）

    Returns:
        float: BLEU score
    """
    model.eval()
    hypotheses = []   # 模型生成的英文
    references  = []  # 标准答案英文

    n = 0
    for src_batch, tgt_batch in dataloader:
        for i in range(src_batch.size(0)):
            src = src_batch[i].unsqueeze(0)   # (1, src_len)
            ref_ids = tgt_batch[i].tolist()

            # 推理
            if method == "beam":
                pred_ids = beam_search(model, src, max_len, bos_id, eos_id,
                                       device, beam_size=beam_size)
            else:
                pred_ids = greedy_decode(model, src, max_len, bos_id, eos_id, device)

            # 解码 token ID → 文本，过滤掉 EOS/PAD
            pred_ids = [t for t in pred_ids if t not in (0, 2, 3)]
            ref_ids  = [t for t in ref_ids  if t not in (0, 2, 3)]

            hypotheses.append(detokenize_en(en_tokenizer.decode(pred_ids)))
            references.append(detokenize_en(en_tokenizer.decode(ref_ids)))

            n += 1
            if max_samples and n >= max_samples:
                break
        if max_samples and n >= max_samples:
            break

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score, hypotheses, references
