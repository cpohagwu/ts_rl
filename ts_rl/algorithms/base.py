from typing import Any, Dict
import pytorch_lightning as pl


class BaseRLModule(pl.LightningModule):
    """Base class for RL-style Lightning modules with manual optimization.

    Notes:
    - We keep manual gradient computations via torch.autograd.grad and avoid loss.backward().
    - Subclasses should implement configure_optimizers only if they truly need an optimizer
      object for schedulers or Lightning integrations; parameter updates can be done manually.
    """

    def __init__(self) -> None:
        super().__init__()
        # Force manual optimization style (no automatic optimizer steps)
        self.automatic_optimization = False

    def configure_optimizers(self):  # type: ignore[override]
        # We do manual parameter updates; return empty list to keep Lightning happy.
        return []

    def on_train_start(self) -> None:
        # Hook for subclasses (e.g., init baselines)
        pass

    def log_dict_step(self, metrics: Dict[str, Any], prog_bar: bool = True) -> None:
        for k, v in metrics.items():
            self.log(k, v, prog_bar=prog_bar, on_step=True, on_epoch=False, batch_size=1)

