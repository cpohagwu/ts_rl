import argparse
import os
import socket
from typing import Any, Dict, List, Optional, Tuple
import warnings
# Ignore UserWarnings, RequestsDependencyWarning from PyTorch Lightning about some deprecated features
warnings.filterwarnings("ignore", category=UserWarning)

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from ts_rl.data.timeseries import TimeseriesConfig, TimeseriesDataModule
from ts_rl.algorithms.reward_weighted_pg import RWRRegressionLightning
from ts_rl.utils.callbacks import EpochTimerCallback


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_input_output_sizes(cfg: Dict[str, Any]) -> Tuple[int, int]:
    lookback = int(cfg["data"]["lookback"])
    features = cfg["data"].get("features", None)
    target_columns = cfg["data"].get("target_columns", None)
    if features is None or target_columns is None:
        # Attempt to infer by reading parquet columns
        import pandas as pd
        df = pd.read_parquet(cfg["data"]["parquet_path"], columns=None)
        control_cols = {"seq_ix", "step_in_seq", "need_prediction"}
        if features is None:
            features = [c for c in df.columns if c not in control_cols]
            cfg["data"]["features"] = features
        if target_columns is None:
            target_columns = list(features)
            cfg["data"]["target_columns"] = target_columns
    input_size = lookback * len(features)
    output_size = len(target_columns)
    return input_size, output_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Reward-Weighted Regression (manual grads) with Lightning")
    parser.add_argument("--config", type=str, default=os.path.join("ts_rl", "configs", "default_config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--parquet", type=str, default=None)
    parser.add_argument("--features", type=str, nargs="*", default=None)
    parser.add_argument("--targets", type=str, nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)

    # Data config
    if args.parquet is not None:
        cfg["data"]["parquet_path"] = args.parquet
    if args.features is not None:
        cfg["data"]["features"] = args.features
    if args.targets is not None:
        cfg["data"]["target_columns"] = args.targets

    data_cfg = TimeseriesConfig(
        parquet_path=cfg["data"]["parquet_path"],
        lookback=int(cfg["data"]["lookback"]),
        normalize=bool(cfg["data"]["normalize"]),
        features=cfg["data"].get("features"),
        target_columns=cfg["data"].get("target_columns"),
        train_val_split=float(cfg["data"]["train_val_split"]),
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
    )

    datamodule = TimeseriesDataModule(data_cfg)
    # We avoid reading data here to keep entry point focused; setup() happens in Trainer.fit

    # Sizes (infer if not set)
    input_size, output_size = infer_input_output_sizes(cfg)
    # Update data_cfg with inferred lists if necessary
    data_cfg.features = cfg["data"].get("features")
    data_cfg.target_columns = cfg["data"].get("target_columns")

    # Model/Algo
    algo = RWRRegressionLightning(
        input_size=input_size,
        hidden_sizes=list(cfg["model"]["hidden_sizes"]),
        output_size=output_size,
        dropout_rate=float(cfg["model"]["dropout_rate"]),
        learning_rate=float(cfg["algo"]["learning_rate"]),
        l2_reg=float(cfg["algo"]["l2_reg"]),
        baseline_decay=float(cfg["algo"]["baseline_decay"]),
        reward_scale_method=str(cfg["algo"]["reward_scale_method"]),
        normalize_advantage=bool(cfg["algo"]["normalize_advantage"]),
        clip_grad_norm=cfg["algo"]["clip_grad_norm"],
        accumulate_batches=int(cfg["algo"]["accumulate_batches"]),
    )

    # Logging and callbacks
    logger = TensorBoardLogger(
        save_dir=cfg["logging"]["save_dir"],
        name=cfg["logging"]["name"],
        default_hp_metric=False
    ) 
    # Create checkpoint directory if it doesn't exist
    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    

    # Configure checkpoint monitoring
    monitor_metric = str(cfg["checkpoint"]["monitor"])
    monitor_mode = str(cfg["checkpoint"]["mode"])
    save_top_k = int(cfg["checkpoint"]["save_top_k"])
    print(f"[Checkpoint] Config: monitor={monitor_metric} mode={monitor_mode} save_top_k={save_top_k}")
    
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="model-{epoch:02d}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=save_top_k,
            save_last=True,
            verbose=True,
            every_n_epochs=int(cfg["checkpoint"].get("every_n_epochs", 1)),
            save_on_train_epoch_end=False
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EpochTimerCallback(),
    ]
    print(f"[Checkpoint] Saving to: {ckpt_dir} monitored={cfg['checkpoint']['monitor']}")

    # Trainer
    # Console log: training strategy summary
    print(
        f"[Trainer] accelerator={cfg['trainer'].get('accelerator','auto')} "
        f"devices={cfg['trainer'].get('devices','auto')} "
        f"strategy={cfg['trainer'].get('strategy','auto')}"
    )

    # Ensure local init for CPU DDP on Windows
    accelerator = str(cfg["trainer"].get("accelerator", "auto"))
    strategy = str(cfg["trainer"].get("strategy", "auto"))
    ddp_custom = None
    if accelerator == "cpu" and strategy.startswith("ddp"):
        # Construct an explicit DDP strategy with a local TCP init method to avoid hostname resolution issues.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        init_method = f"tcp://127.0.0.1:{port}"
        ddp_custom = DDPStrategy(process_group_backend="gloo", init_method=init_method)
        print(f"[Trainer] Using DDP init_method={init_method}")

    trainer = pl.Trainer(
        max_epochs=int(args.max_epochs or cfg["trainer"]["max_epochs"]),
        accelerator=cfg["trainer"].get("accelerator", "auto"),
        devices=cfg["trainer"].get("devices", "auto"),
        strategy=(ddp_custom or cfg["trainer"].get("strategy", None)),
        precision=cfg["trainer"].get("precision", 32),
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=25,
        enable_model_summary=False,
    )

    trainer.fit(algo, datamodule=datamodule)


if __name__ == "__main__":
    main()
