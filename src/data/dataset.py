import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs, max_seq_len=128):
        self.src = src_seqs
        self.tgt = tgt_seqs
        self.max_len = max_seq_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx][: self.max_len]
        tgt = self.tgt[idx][: self.max_len]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    src_batch, tgt_batch = zip(*batch)

    src_lens = [s.size(0) for s in src_batch]
    tgt_lens = [t.size(0) for t in tgt_batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = torch.full((len(src_batch), max_src), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(tgt_batch), max_tgt), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, : s.size(0)] = s
        tgt_padded[i, : t.size(0)] = t

    return src_padded, tgt_padded


def make_dataloader(src_seqs, tgt_seqs, batch_size, max_seq_len, pad_id=0, shuffle=True):
    dataset = TranslationDataset(src_seqs, tgt_seqs, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,
    )
