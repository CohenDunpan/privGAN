# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn


def _init_weights(module: nn.Module) -> None:
    # Match Keras RandomNormal(stddev=0.02)
    if isinstance(module, (nn.Linear, nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_optimizer(model: nn.Module, lr: float = 0.0002, beta1: float = 0.5) -> torch.optim.Optimizer:
    """Adam optimizer used throughout the GANs."""

    return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))


class MNIST_Generator(nn.Module):
    """Generator for MNIST (fully-connected, 784 output with tanh)."""

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
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MNIST_Discriminator(nn.Module):
    """Discriminator for MNIST (binary classification with sigmoid)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 2048),
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


class MNIST_DiscriminatorPrivate(nn.Module):
    """Classifier to guess which generator produced the sample."""

    def __init__(self, OutSize: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 2048),
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
    "MNIST_Generator",
    "MNIST_Discriminator",
    "MNIST_DiscriminatorPrivate",
    "make_optimizer",
]