import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
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
        self.use_amp = cfg["train"].get("use_amp", False) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.grad_accum_steps = cfg["train"].get("grad_accum_steps", 1)
        log_dir = os.path.join(cfg["train"]["checkpoint_dir"], "tb_logs")
        self.writer = SummaryWriter(log_dir=log_dir)
        if self.use_amp:
            print("AMP (mixed precision) enabled.")
        if self.grad_accum_steps > 1:
            print(f"Gradient accumulation: {self.grad_accum_steps} steps "
                  f"(effective batch = {cfg['train']['batch_size'] * self.grad_accum_steps})")
        print(f"TensorBoard logs -> {log_dir}")

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        accum_count = 0

        self.optimizer.zero_grad()

        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(src, tgt_in)
                    loss = self.criterion(logits, tgt_out) / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(src, tgt_in)
                loss = self.criterion(logits, tgt_out) / self.grad_accum_steps
                loss.backward()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1
            accum_count += 1

            if accum_count == self.grad_accum_steps:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg["train"]["grad_clip"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg["train"]["grad_clip"]
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                accum_count = 0

                if self.global_step % self.cfg["train"]["log_interval"] == 0:
                    avg = total_loss / n_batches
                    elapsed = time.time() - t0
                    lr = self.scheduler.last_lr
                    print(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"Loss {avg:.4f} | LR {lr:.2e} | "
                        f"Elapsed {elapsed:.1f}s"
                    )
                    self.writer.add_scalar("train/loss", avg, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

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

    def fit(self, train_loader, val_loader, num_epochs, start_epoch=1):
        best_val_loss = float("inf")
        ckpt_dir = self.cfg["train"]["checkpoint_dir"]

        for epoch in range(start_epoch, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.evaluate(val_loader)

            print(
                f"\n{'='*60}\n"
                f"Epoch {epoch} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
                f"{'='*60}\n"
            )

            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
            self.writer.flush()

            # 每 epoch 保存 last.pt（断点续训用）
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, self.global_step, val_loss,
                path=os.path.join(ckpt_dir, "last.pt"),
            )

            # 每 5 epoch 保存带编号快照
            if epoch % 5 == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step, val_loss,
                    path=os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
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
