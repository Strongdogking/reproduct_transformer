import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss as described in the paper (epsilon=0.1)."""

    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        # logits: (batch * tgt_len, vocab_size)
        # target: (batch * tgt_len,)
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)

        with torch.no_grad():
            smooth_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0
            # Zero out rows where target is pad
            pad_mask = target == self.pad_idx
            smooth_dist[pad_mask] = 0

        log_prob = F.log_softmax(logits, dim=-1)
        loss = -(smooth_dist * log_prob).sum(dim=-1)

        # Average only over non-pad tokens
        non_pad = (~pad_mask).sum()
        return loss.sum() / non_pad.clamp(min=1)
