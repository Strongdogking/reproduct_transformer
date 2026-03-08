import os
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_step": scheduler._step,
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler:
        scheduler._step = ckpt["scheduler_step"]
    print(f"Checkpoint loaded: {path} (epoch {ckpt['epoch']}, step {ckpt['step']})")
    return ckpt["epoch"], ckpt["step"], ckpt["loss"]
