# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_optimizer(model: nn.Module, lr: float = 0.0002, beta1: float = 0.5) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))


class CIFAR_Generator(nn.Module):
    """Generator for CIFAR-10 (DCGAN-style)."""

    def __init__(self, randomDim: int = 100):
        super().__init__()
        self.randomDim = randomDim
        self.fc = nn.Sequential(
            nn.Linear(randomDim, 2 * 2 * 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.size(0), 512, 2, 2)
        return self.net(x)


class CIFAR_Discriminator(nn.Module):
    """Discriminator for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1),
            nn.Sigmoid(),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CIFAR_DiscriminatorPrivate(nn.Module):
    """Classifier that predicts which generator produced the sample."""

    def __init__(self, OutSize: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, OutSize),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = [
    "CIFAR_Generator",
    "CIFAR_Discriminator",
    "CIFAR_DiscriminatorPrivate",
    "make_optimizer",
]

