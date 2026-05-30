from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class ExpertPrediction:
    label: str
    confidence: float
    scores: dict[str, float]


class ExpertClassifier(nn.Module):
    def __init__(self, input_dim: int, classes: list[str], hidden_dim: int = 64):
        super().__init__()
        self.classes = classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, features: np.ndarray) -> ExpertPrediction:
        with torch.no_grad():
            x = torch.as_tensor(features[None, :], dtype=torch.float32)
            probs = torch.softmax(self(x), dim=-1).cpu().numpy()[0]
        label_idx = int(np.argmax(probs))
        return ExpertPrediction(
            label=self.classes[label_idx],
            confidence=float(probs[label_idx]),
            scores={name: float(score) for name, score in zip(self.classes, probs)},
        )


def train_supervised_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    epochs: int = 25,
    lr: float = 1e-3,
) -> ExpertClassifier:
    model = ExpertClassifier(features.shape[1], classes)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    y = np.asarray([class_to_idx[str(label)] for label in labels], dtype=np.int64)

    x_t = torch.as_tensor(features, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        logits = model(x_t)
        loss = loss_fn(logits, y_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return model

