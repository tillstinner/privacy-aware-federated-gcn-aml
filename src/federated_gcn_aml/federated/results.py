from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class BestRoundState:
    round: int
    val_aupr: float
    model_state: dict[str, torch.Tensor]
    metrics: dict[str, Any]
    predictions: dict[str, Any]


@dataclass(frozen=True)
class FederatedRunResult:
    model_state: dict[str, torch.Tensor]
    metrics: dict[str, Any]
    round_metrics: list[dict[str, Any]]
    predictions: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
