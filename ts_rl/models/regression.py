from __future__ import annotations

from typing import List
import numpy as np
import torch
import torch.nn as nn


class RegressionPolicyGradient(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self.output_size = output_size

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = input_size
        for h in self.hidden_sizes:
            layer = nn.Linear(in_dim, h)
            nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(2.0 / (h + in_dim)))
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout_rate))
            in_dim = h

        self.output_layer = nn.Linear(in_dim, output_size)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=np.sqrt(2.0 / (output_size + in_dim)))
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        y = x
        for i, layer in enumerate(self.layers):
            y = torch.tanh(layer(y))
            if training:
                y = self.dropouts[i](y)
        y = self.output_layer(y)
        return y

