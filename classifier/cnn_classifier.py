# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _ensure_channel_first(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4 and x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
        return x.permute(0, 3, 1, 2)
    return x


class _CNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class CNNClassifier:
    """PyTorch CNN classifier mirroring the original Keras model."""

    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        dropout: float = 0.5,
        learning_rate: float = 1.0,
        rho: float = 0.95,
        epsilon: float = 1e-06,
        device: Optional[torch.device] = None,
    ):
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.__build_model().to(self.device)

    def train(self, x_train, y_train, x_validation, y_validation, batch_size: int = 256, epochs: int = 25):
        """Train and evaluate the classifier; returns (loss, accuracy)."""

        x_train_t = _ensure_channel_first(torch.tensor(x_train, dtype=torch.float32, device=self.device))
        x_val_t = _ensure_channel_first(torch.tensor(x_validation, dtype=torch.float32, device=self.device))

        y_train_np = np.asarray(y_train)
        y_val_np = np.asarray(y_validation)
        if y_train_np.ndim > 1 and y_train_np.shape[-1] > 1:
            y_train_np = np.argmax(y_train_np, axis=1)
        if y_val_np.ndim > 1 and y_val_np.shape[-1] > 1:
            y_val_np = np.argmax(y_val_np, axis=1)

        y_train_t = torch.tensor(y_train_np, dtype=torch.long, device=self.device)
        y_val_t = torch.tensor(y_val_np, dtype=torch.long, device=self.device)

        train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate, rho=self.rho, eps=self.epsilon)

        for _ in range(epochs):
            self.model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # evaluation
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = self.model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)

        val_loss /= max(total, 1)
        val_acc = correct / max(total, 1)
        return val_loss, val_acc

    def __build_model(self) -> nn.Module:
        return _CNN(num_classes=self.num_classes, dropout=self.dropout)


__all__ = ["CNNClassifier"]
