from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from ..models.regression import RegressionPolicyGradient
from .base import BaseRLModule


def _grad_norm(grad_tensors: List[torch.Tensor]) -> float:
    total = 0.0
    for g in grad_tensors:
        total += float(torch.sum(g.detach() ** 2))
    return float(np.sqrt(total))


class RWRRegressionLightning(BaseRLModule):
    """Reward-Weighted Regression using manual gradients (REINFORCE-style).

    This mirrors your notebook's reward-weighted gradient scheme:
    - Per-sample loss: MSE over output dims
    - Reward: negative MSE
    - Advantage: reward - EMA baseline (with optional normalization)
    - Positive scaling via exp/tanh/linear
    - Manual gradients via torch.autograd.grad and manual param updates

    We update parameters every `accumulate_batches` steps using the averaged accumulated grads,
    with optional grad-norm clipping and L2 penalty.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        l2_reg: float = 0.0,
        baseline_decay: float = 0.95,
        reward_scale_method: str = "exp",  # 'exp', 'tanh', 'linear'
        normalize_advantage: bool = True,
        clip_grad_norm: Optional[float] = 1.0,
        accumulate_batches: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = RegressionPolicyGradient(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout_rate=dropout_rate,
        )
        self._output_size = int(output_size)

        # Running EMA baseline for rewards
        self.register_buffer("baseline", torch.tensor(0.0))
        self.example_input_array = torch.randn(2, input_size)

        # Accumulators for manual optimization
        self._accum_grads: Dict[str, torch.Tensor] = {}
        self._accum_count: int = 0

    @property
    def learning_rate(self) -> float:
        return float(self.hparams.learning_rate)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:  # type: ignore[override]
        return self.model(x, training=training)

    def _ensure_accumulators(self) -> None:
        if self._accum_grads:
            return
        for name, p in self.model.named_parameters():
            self._accum_grads[name] = torch.zeros_like(p)

    def _reward_scale(self, advantages: np.ndarray) -> np.ndarray:
        method = str(self.hparams.reward_scale_method).lower()
        if method == "exp":
            return np.exp(advantages * 0.5)
        if method == "tanh":
            return 0.5 * (1.0 + np.tanh(advantages))
        if method == "linear":
            return 1.0 + advantages
        raise ValueError(f"Unknown reward_scale_method={self.hparams.reward_scale_method}")

    def _maybe_step(self) -> None:
        if self._accum_count < int(self.hparams.accumulate_batches):
            return

        # Average grads
        for k in self._accum_grads:
            self._accum_grads[k] = self._accum_grads[k] / max(1, self._accum_count)

        # L2 penalty
        l2_reg: float = float(self.hparams.l2_reg)
        if l2_reg > 0:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if "weight" in name:
                        self._accum_grads[name] += 2.0 * l2_reg * p.data

        # Grad norm clipping
        clip = self.hparams.clip_grad_norm
        if clip is not None:
            with torch.no_grad():
                grad_list = list(self._accum_grads.values())
                total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grad_list) + 1e-12)
                clip_coef = float(clip) / float(total_norm + 1e-12)
                if clip_coef < 1.0:
                    for k in self._accum_grads:
                        self._accum_grads[k] *= clip_coef

        # Manual param update
        lr = self.learning_rate
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.data -= lr * self._accum_grads[name]

        # Log and reset
        grad_norm_val = _grad_norm(list(self._accum_grads.values()))
        self.log("train/grad_norm", grad_norm_val, prog_bar=False, on_step=True, on_epoch=False)
        for k in self._accum_grads:
            self._accum_grads[k].zero_()
        self._accum_count = 0

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):  # type: ignore[override]
        x, y_true = batch

        self._ensure_accumulators()
        y_pred = self.forward(x, training=True)

        # Per-sample MSE (mean over output dims)
        per_sample_mse = torch.mean((y_true - y_pred) ** 2, dim=1)  # [B]

        # Rewards and EMA baseline
        rewards = -per_sample_mse.detach().cpu().numpy()  # numpy for normalization ops
        # Update baseline (EMA of mean reward)
        mean_r = float(np.mean(rewards)) if rewards.size > 0 else 0.0
        with torch.no_grad():
            self.baseline.mul_(float(self.hparams.baseline_decay)).add_((1.0 - float(self.hparams.baseline_decay)) * mean_r)

        advantages = rewards - float(self.baseline.item())
        if bool(self.hparams.normalize_advantage) and advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        reward_scales = self._reward_scale(advantages).astype(np.float32)
        reward_scales_t = torch.from_numpy(reward_scales).to(per_sample_mse.device)

        weighted_losses = per_sample_mse * reward_scales_t
        weighted_loss = torch.mean(weighted_losses)

        # Manual grads via autograd.grad
        grads = torch.autograd.grad(
            outputs=weighted_loss,
            inputs=list(self.model.parameters()),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

        # Accumulate
        for (name, _p), g in zip(self.model.named_parameters(), grads):
            self._accum_grads[name] += g.detach()
        self._accum_count += 1

        # Manual step conditionally
        self._maybe_step()

        # Logging
        mse = torch.mean(per_sample_mse.detach())
        self.log("train/mse", mse, prog_bar=True, on_step=True, on_epoch=False, batch_size=x.size(0))
        self.log("train/weighted_loss", weighted_loss.detach(), prog_bar=False, on_step=True, on_epoch=False, batch_size=x.size(0))

        return {
            "loss": weighted_loss.detach(),
        }

    def on_train_epoch_end(self) -> None:
        # Flush any leftover accumulation at epoch end
        if self._accum_count > 0:
            self._maybe_step()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):  # type: ignore[override]
        x, y_true = batch
        with torch.no_grad():
            y_pred = self.forward(x, training=False)

        # Log MSE for reference
        mse = torch.mean((y_true - y_pred) ** 2)
        self.log("val/mse", mse, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0), sync_dist=True)

        # Accumulate stats for R2 across the epoch
        # Maintain running sums to compute per-feature R2 at epoch end
        self._val_sum_y += torch.sum(y_true, dim=0)
        self._val_sum_y2 += torch.sum(y_true ** 2, dim=0)
        self._val_ss_res += torch.sum((y_true - y_pred) ** 2, dim=0)
        self._val_count += y_true.new_tensor(float(y_true.size(0)))

    def on_validation_epoch_start(self) -> None:
        # Initialize accumulators at start of validation (work across DDP ranks)
        D = self._output_size
        device = self.device if isinstance(self.device, torch.device) else torch.device("cpu")
        self._val_sum_y = torch.zeros(D, device=device)
        self._val_sum_y2 = torch.zeros(D, device=device)
        self._val_ss_res = torch.zeros(D, device=device)
        self._val_count = torch.tensor(0.0, device=device)

    def on_validation_epoch_end(self) -> None:
        # All-gather to aggregate stats across ranks
        sum_y = self.all_gather(self._val_sum_y)
        sum_y2 = self.all_gather(self._val_sum_y2)
        ss_res = self.all_gather(self._val_ss_res)
        N = self.all_gather(self._val_count)
        # Reduce along first dim if gathered
        if sum_y.dim() >= 2:
            sum_y = torch.sum(sum_y, dim=0)
            sum_y2 = torch.sum(sum_y2, dim=0)
            ss_res = torch.sum(ss_res, dim=0)
            N = torch.sum(N, dim=0)
        eps = 1e-8
        N = N.clamp_min(1.0)
        mean_y = sum_y / N
        ss_tot = (sum_y2 - N * (mean_y ** 2)).clamp_min(eps)
        r2_per_feature = 1.0 - (ss_res / ss_tot)
        r2_avg = torch.mean(r2_per_feature).item()

        # Log detailed per-feature R2 to TensorBoard; average to both TB and console
        with torch.no_grad():
            self.log("val/r2_avg", float(r2_avg), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            if self.trainer.is_global_zero:
                for i, v in enumerate(r2_per_feature.detach().cpu().tolist()):
                    self.log(f"val/r2_feature_{i}", float(v), prog_bar=False, on_step=False, on_epoch=True)
                print(f"[Val][epoch={int(self.current_epoch)}] R2_avg={r2_avg:.5f}")
