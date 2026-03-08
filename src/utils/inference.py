"""P4-1 & P4-2: Greedy Decode + Beam Search 推理。"""
import torch


@torch.no_grad()
def greedy_decode(model, src, max_len, bos_id, eos_id, device):
    """
    Greedy decoding：每步选概率最高的 token。

    Args:
        model:  Transformer 模型
        src:    (1, src_len) 中文 token ID tensor
        max_len: 最大生成长度
        bos_id / eos_id: 特殊 token ID
        device: torch.device

    Returns:
        list[int]: 生成的英文 token ID 序列（不含 BOS，含 EOS）
    """
    model.eval()
    src = src.to(device)

    enc_out, src_mask = model.encode(src)

    # 初始化 decoder 输入：只有 BOS
    tgt = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_len):
        dec_out = model.decode(tgt, enc_out, src_mask)
        logits = model.fc_out(dec_out)          # (1, cur_len, tgt_vocab)
        next_id = logits[:, -1, :].argmax(-1)   # 取最后一个位置的最高分
        tgt = torch.cat([tgt, next_id.unsqueeze(0)], dim=1)

        if next_id.item() == eos_id:
            break

    return tgt[0, 1:].tolist()   # 去掉 BOS


@torch.no_grad()
def beam_search(model, src, max_len, bos_id, eos_id, device, beam_size=4):
    """
    Beam Search：维护 beam_size 条候选序列，按累积 log-prob 排序。

    Returns:
        list[int]: 最优序列的 token ID（不含 BOS，含 EOS）
    """
    model.eval()
    src = src.to(device)

    enc_out, src_mask = model.encode(src)

    # 每条 beam: (log_prob, token_ids)
    beams = [(0.0, [bos_id])]
    completed = []

    for _ in range(max_len):
        candidates = []

        for log_prob, seq in beams:
            if seq[-1] == eos_id:
                completed.append((log_prob, seq))
                continue

            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            dec_out = model.decode(tgt, enc_out, src_mask)
            logits = model.fc_out(dec_out[:, -1, :])           # (1, tgt_vocab)
            log_probs = torch.log_softmax(logits, dim=-1)[0]   # (tgt_vocab,)

            # 取 top beam_size 个候选
            topk_lp, topk_ids = log_probs.topk(beam_size)
            for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                candidates.append((log_prob + lp, seq + [tid]))

        if not candidates:
            break

        # 保留分数最高的 beam_size 条
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

        # 如果所有 beam 都结束了，提前退出
        if all(seq[-1] == eos_id for _, seq in beams):
            completed.extend(beams)
            break

    if not completed:
        completed = beams

    # 按长度归一化 log-prob 选最优
    best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
    return best[1][1:]   # 去掉 BOS
