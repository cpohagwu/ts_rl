import time
import pytorch_lightning as pl


class EpochTimerCallback(pl.callbacks.Callback):
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        t = getattr(pl_module, "_epoch_start_time", None)
        if t is not None:
            duration = time.time() - t
            pl_module.log("time/epoch_sec", duration, prog_bar=False, on_epoch=True, logger=True)

