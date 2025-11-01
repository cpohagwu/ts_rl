import argparse
import os
from typing import List, Optional, Tuple

import yaml
import numpy as np
import pandas as pd
import torch

from ts_rl.algorithms.reward_weighted_pg import RWRRegressionLightning


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_indices(df: pd.DataFrame, lookback: int) -> Tuple[List[Tuple[int, int]], dict]:
    # Ensure ordering
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    seq_groups = {int(k): v.index.values for k, v in df.groupby("seq_ix")}
    samples: List[Tuple[int, int]] = []
    for _, idxs in seq_groups.items():
        seq_df = df.loc[idxs]
        need_pred = seq_df["need_prediction"].astype(bool).values
        for local_end in range(lookback - 1, len(idxs) - 1):
            if not need_pred[local_end]:
                continue
            end_abs = int(idxs[local_end])
            tgt_abs = int(idxs[local_end + 1])
            samples.append((end_abs, tgt_abs))
    return samples, seq_groups


def infer_columns(df: pd.DataFrame, features: Optional[List[str]], targets: Optional[List[str]]):
    control = {"seq_ix", "step_in_seq", "need_prediction"}
    if features is None:
        features = [c for c in df.columns if c not in control]
    if targets is None:
        targets = list(features)
    return features, targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on need_prediction rows (R2 per feature)")
    parser.add_argument("--config", type=str, default=os.path.join("ts_rl", "configs", "test_config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    parquet_path = cfg["data"]["parquet_path"]
    lookback = int(cfg["data"]["lookback"])
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)

    features, targets = infer_columns(df, cfg["data"].get("features"), cfg["data"].get("target_columns"))
    X_array = df[features].values.astype(np.float32)
    Y_array = df[targets].values.astype(np.float32)

    samples, seq_groups = build_indices(df, lookback)
    print(f"[Test] Total sequences: {len(seq_groups)} | Samples (need_prediction & windowable): {len(samples)}")

    # Load model
    ckpt_path = cfg["checkpoint"]["path"]
    module: RWRRegressionLightning = RWRRegressionLightning.load_from_checkpoint(ckpt_path)
    model = module.model.eval()

    D = len(targets)
    sum_y = torch.zeros(D)
    sum_y2 = torch.zeros(D)
    ss_res = torch.zeros(D)
    total = 0.0

    # Simple CPU evaluation in chunks
    bs = int(cfg.get("eval", {}).get("batch_size", 512))
    def make_window(end_abs_idx: int) -> np.ndarray:
        # Find seq and local index for end_abs
        seq_ix = int(df.loc[end_abs_idx, "seq_ix"])  # type: ignore[index]
        idxs = seq_groups[seq_ix]
        local_end = int(np.where(idxs == end_abs_idx)[0][0])
        start_local = max(0, local_end - (lookback - 1))
        win = idxs[start_local: local_end + 1]
        X = X_array[win]
        # Per-window normalization
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-8
        Xn = (X - mu) / sigma
        return Xn.reshape(-1).astype(np.float32)

    with torch.no_grad():
        for i in range(0, len(samples), bs):
            batch = samples[i: i + bs]
            x_np = np.stack([make_window(end) for end, _ in batch], axis=0)
            y_np = np.stack([Y_array[tgt] for _, tgt in batch], axis=0)
            x = torch.from_numpy(x_np)
            y = torch.from_numpy(y_np)
            y_pred = model(x, training=False)

            sum_y += torch.sum(y, dim=0)
            sum_y2 += torch.sum(y ** 2, dim=0)
            ss_res += torch.sum((y - y_pred) ** 2, dim=0)
            total += y.size(0)

    eps = 1e-8
    mean_y = sum_y / max(total, 1.0)
    ss_tot = (sum_y2 - total * (mean_y ** 2)).clamp_min(eps)
    r2_per_feature = 1.0 - (ss_res / ss_tot)
    r2_avg = float(torch.mean(r2_per_feature).item())

    print(f"[Test] R2_avg={r2_avg:.5f}")
    for i, v in enumerate(r2_per_feature.tolist()):
        print(f"[Test] R2_feature_{i}={v:.5f}")


if __name__ == "__main__":
    main()

