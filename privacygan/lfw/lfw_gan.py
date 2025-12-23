# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_optimizer(model: nn.Module, lr: float = 0.0002, beta1: float = 0.5) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))


class LFW_Generator(nn.Module):
    """Generator for LFW (fully-connected)."""

    def __init__(self, randomDim: int = 100):
        super().__init__()
        self.randomDim = randomDim
        self.net = nn.Sequential(
            nn.Linear(randomDim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2914),
            nn.Tanh(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LFW_Discriminator(nn.Module):
    """Discriminator for LFW (binary classification)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2914, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class LFW_DiscriminatorPrivate(nn.Module):
    """Classifier that predicts generator index."""

    def __init__(self, OutSize: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2914, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, OutSize),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


__all__ = [
    "LFW_Generator",
    "LFW_Discriminator",
    "LFW_DiscriminatorPrivate",
    "make_optimizer",
]