from typing import Optional
import pytorch_lightning as pl


class BaseDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self._train_dataset = None
        self._val_dataset = None

    @property
    def has_train(self) -> bool:
        return self._train_dataset is not None

    @property
    def has_val(self) -> bool:
        return self._val_dataset is not None

