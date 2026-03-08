import torch.optim as optim


class WarmupScheduler:
    """
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        s = self._step
        return self.d_model ** (-0.5) * min(s ** (-0.5), s * self.warmup_steps ** (-1.5))

    @property
    def last_lr(self):
        return self._get_lr()
