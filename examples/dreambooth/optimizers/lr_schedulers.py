import torch

from torch._six import inf


class LRReduceOnPlateauWrapper(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Wrapper for ReduceLROnPlateau to be compatible with the other LR schedulers
    """

    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(optimizer, *args, **kwargs)

    def step(self, metrics, epoch=None):
        super().step(metrics)

    def get_last_lr(self):
        # return self.optimizer.param_groups[0]["lr"]
        return self._last_lr


class LRReduceOnPlateauComposer:
    """
    Compose LRReduceOnPlateau over another scheduler
    """

    def __init__(
        self,
        scheduler,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
    ):
        self.scheduler = scheduler  # other scheduler being composed around

        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.min_lr = min_lr
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0

        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch = self.last_epoch + 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(self.last_epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        # self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        # also step the scheduler being composed around
        self.scheduler.step()

    def _reduce_lr(self, epoch):
        # for i, param_group in enumerate(self.optimizer.param_groups):
        #     old_lr = float(param_group["lr"])
        #     new_lr = max(old_lr * self.factor, self.min_lrs[i])
        #     if old_lr - new_lr > self.eps:
        #         param_group["lr"] = new_lr
        #         if self.verbose:
        #             epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
        #             print("Epoch {}: reducing learning rate" " of group {} to {:.4e}.".format(epoch_str, i, new_lr))

        # reduce the lr of the sheduler being composed
        for i, lr in enumerate(self.scheduler.base_lrs):
            old_lr = float(lr)
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                self.scheduler.base_lrs[i] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                    print("Epoch {}: reducing learning rate" " of group {} to {:.4e}.".format(epoch_str, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
