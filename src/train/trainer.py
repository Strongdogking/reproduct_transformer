import os
import time
import torch
from src.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, cfg):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        self.global_step = 0

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Teacher forcing: feed tgt[:-1] as input, predict tgt[1:]
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = self.model(src, tgt_in)  # (batch, tgt_len-1, vocab)
            loss = self.criterion(logits, tgt_out)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["train"]["grad_clip"]
            )
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            total_loss += loss.item()
            n_batches += 1

            if self.global_step % self.cfg["train"]["log_interval"] == 0:
                avg = total_loss / n_batches
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch} | Step {self.global_step} | "
                    f"Loss {avg:.4f} | LR {self.scheduler.last_lr:.2e} | "
                    f"Elapsed {elapsed:.1f}s"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = self.model(src, tgt_in)
            loss = self.criterion(logits, tgt_out)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(self, train_loader, val_loader, num_epochs):
        best_val_loss = float("inf")
        ckpt_dir = self.cfg["train"]["checkpoint_dir"]

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.evaluate(val_loader)

            print(
                f"\n{'='*60}\n"
                f"Epoch {epoch} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
                f"{'='*60}\n"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.global_step,
                    val_loss,
                    path=os.path.join(ckpt_dir, "best.pt"),
                )
