from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import pytorch_lightning as pl


@dataclass
class TimeseriesConfig:
    parquet_path: str
    lookback: int = 32
    normalize: bool = True
    features: Optional[List[str]] = None  # If None, use all non-index columns minus control columns
    target_columns: Optional[List[str]] = None  # If None, predict next-step of features
    train_val_split: float = 0.9  # split by seq_ix
    batch_size: int = 256
    num_workers: int = 0


class TimeseriesParquetDataset(Dataset):
    """Create (X, y) pairs from Parquet sequences.

    - Rows have columns: seq_ix, step_in_seq, need_prediction, feature_* (N features)
    - For each sequence, we iterate chronological order.
    - We generate samples only where need_prediction == True and step_in_seq >= lookback.
    - X: flattened window of last `lookback` rows' features.
    - y: next-step vector for `target_columns` (or features if None).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 32,
        normalize: bool = True,
        features: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        seq_ix_filter: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()

        required_cols = {"seq_ix", "step_in_seq", "need_prediction"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataframe missing required columns: {missing}")

        # Determine features
        control_cols = ["seq_ix", "step_in_seq", "need_prediction"]
        if features is None:
            features = [c for c in df.columns if c not in control_cols]
        if not features:
            raise ValueError("No feature columns found.")

        if target_columns is None:
            target_columns = list(features)

        self.lookback = int(lookback)
        self.normalize = bool(normalize)
        self.features = features
        self.target_columns = target_columns

        # Filter by seq_ix if provided
        if seq_ix_filter is not None:
            df = df[df["seq_ix"].isin(list(seq_ix_filter))].copy()

        # Ensure correct ordering inside sequences
        df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)

        # Build index of samples: (end_abs_idx, target_abs_idx)
        self._df = df
        self._seq_groups: Dict[int, np.ndarray] = {
            int(k): v.index.values for k, v in df.groupby("seq_ix")
        }
        self._samples: List[Tuple[int, int]] = []

        for seq, idxs in self._seq_groups.items():
            seq_df = df.loc[idxs]
            need_pred = seq_df["need_prediction"].astype(bool).values
            steps = seq_df["step_in_seq"].values
            # Iterate local indices; end_idx is inclusive end of input window
            for local_end in range(self.lookback - 1, len(idxs) - 1):  # ensure a next-step exists
                if not need_pred[local_end]:
                    continue
                # Safety: within sequence length and respect lookback
                # Input window will include rows [local_end - lookback + 1, ..., local_end]
                end_abs_idx = int(idxs[local_end])
                target_abs_idx = int(idxs[local_end + 1])
                self._samples.append((end_abs_idx, target_abs_idx))

        # Precompute numpy arrays for speed
        self._feature_array = df[self.features].values.astype(np.float32)
        self._target_array = df[self.target_columns].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end_abs_idx, target_abs_idx = self._samples[idx]
        # Locate sequence for end_abs_idx
        # Find its sequence by searching membership in groups (rarely executed; acceptable for training)
        # For speed, we can binary search but this is fine given pre-processing load in pandas dominates.
        seq_ix = int(self._df.loc[end_abs_idx, "seq_ix"])  # type: ignore[index]
        seq_indices = self._seq_groups[seq_ix]
        local_end = int(np.where(seq_indices == end_abs_idx)[0][0])

        # Window includes current end_abs_idx
        start_local = local_end - (self.lookback - 1)
        if start_local < 0:
            start_local = 0
        window_idxs = seq_indices[start_local : local_end + 1]

        X = self._feature_array[window_idxs]
        if self.normalize:
            mu = X.mean(axis=0, keepdims=True)
            sigma = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mu) / sigma
        x_flat = X.reshape(-1).astype(np.float32)

        # Target is the actual next-step after the input window end
        y = self._target_array[target_abs_idx]
        return torch.from_numpy(x_flat), torch.from_numpy(y)


class SequenceShardedSampler(Sampler[int]):
    """Shard dataset samples by sequence across distributed workers.

    We assume dataset._samples is a list of (seq_ix, pos).
    Each rank gets a subset of seq_ix; we then yield indices belonging only to those seq_ix.
    """

    def __init__(self, dataset: TimeseriesParquetDataset, shuffle: bool = True) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle

        # Precompute mapping from seq_ix -> list of dataset indices
        self._seq_to_indices: Dict[int, List[int]] = {}
        for ds_index, (end_abs_idx, _tgt) in enumerate(dataset._samples):
            seq = int(dataset._df.loc[end_abs_idx, "seq_ix"])  # type: ignore[index]
            self._seq_to_indices.setdefault(seq, []).append(ds_index)

    def __iter__(self):
        import torch.distributed as dist

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        seqs = sorted(self._seq_to_indices.keys())
        # Shard sequences across ranks
        assigned = [s for i, s in enumerate(seqs) if (i % max(world_size, 1)) == rank]

        if self.shuffle:
            rng = np.random.default_rng(seed=rank)
            rng.shuffle(assigned)

        for s in assigned:
            idxs = self._seq_to_indices[s]
            if self.shuffle:
                rng = np.random.default_rng(seed=s + rank)
                idxs = list(idxs)
                rng.shuffle(idxs)
            for i in idxs:
                yield i

    def __len__(self) -> int:
        # Only the portion for this rank; conservative upper bound is dataset length
        # Lightning uses DistributedSamplerWrapper-like behavior; returning dataset length is acceptable
        return len(self.dataset)


class TimeseriesDataModule(pl.LightningDataModule):
    def __init__(self, cfg: TimeseriesConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._train_ds: Optional[TimeseriesParquetDataset] = None
        self._val_ds: Optional[TimeseriesParquetDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        # Load parquet into DataFrame
        df = pd.read_parquet(self.cfg.parquet_path)

        # Split by unique seq_ix
        seqs = df["seq_ix"].unique().tolist()
        rng = np.random.default_rng(seed=42)
        rng.shuffle(seqs)
        split = int(len(seqs) * float(self.cfg.train_val_split))
        # Ensure both splits are non-empty when possible
        if split == 0 and len(seqs) > 1:
            split = 1
        if split == len(seqs) and len(seqs) > 1:
            split = len(seqs) - 1
        train_seqs = set(seqs[:split])
        val_seqs = set(seqs[split:])

        # Console logs: number of sequences discovered
        print(f"[Data] Total sequences discovered (unique seq_ix): {len(seqs)}")
        # Sanity: ensure equal lengths per sequence
        lengths = df.groupby("seq_ix")["step_in_seq"].nunique().tolist()
        if len(set(lengths)) != 1:
            print("[Data][warn] Sequences have varying lengths; proceeding but RNN/windowing assumes uniform length.")
        else:
            L = int(lengths[0])
            print(f"[Data] Sequence length (unique step_in_seq): {L}")
            windows_per_seq = max(0, L - int(self.cfg.lookback))
            print(
                f"[Data] Theoretical windows per seq (L - lookback): {windows_per_seq} | "
                f"Train seqs: {len(train_seqs)} -> {len(train_seqs) * windows_per_seq} | "
                f"Val seqs: {len(val_seqs)} -> {len(val_seqs) * windows_per_seq}"
            )

        self._train_ds = TimeseriesParquetDataset(
            df,
            lookback=self.cfg.lookback,
            normalize=self.cfg.normalize,
            features=self.cfg.features,
            target_columns=self.cfg.target_columns,
            seq_ix_filter=train_seqs,
        )

        self._val_ds = TimeseriesParquetDataset(
            df,
            lookback=self.cfg.lookback,
            normalize=self.cfg.normalize,
            features=self.cfg.features,
            target_columns=self.cfg.target_columns,
            seq_ix_filter=val_seqs,
        )

        # Log available samples (proxy for step_in_seq - lookback under need_prediction mask)
        print(
            f"[Data] Train samples (need_prediction & windowable): {len(self._train_ds)} | "
            f"Val samples: {len(self._val_ds)} | Lookback: {self.cfg.lookback}"
        )

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._train_ds is not None
        sampler = SequenceShardedSampler(self._train_ds, shuffle=True)
        return DataLoader(
            self._train_ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._val_ds is not None
        sampler = SequenceShardedSampler(self._val_ds, shuffle=False)
        return DataLoader(
            self._val_ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            sampler=sampler,
        )
