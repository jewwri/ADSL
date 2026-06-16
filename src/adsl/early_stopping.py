from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyStoppingDecision:
    should_stop: bool
    reason: str
    best_smoothed_return: float
    stale_evaluations: int


class EarlyStoppingMonitor:
    def __init__(
        self,
        *,
        enabled: bool,
        min_steps: int,
        patience_evals: int,
        min_delta: float,
        smoothing_window: int,
    ) -> None:
        self.enabled = enabled
        self.min_steps = max(0, int(min_steps))
        self.patience_evals = max(1, int(patience_evals))
        self.min_delta = max(0.0, float(min_delta))
        self._recent_returns: deque[float] = deque(maxlen=max(1, int(smoothing_window)))
        self._best_smoothed_return = float("-inf")
        self._stale_evaluations = 0

    @property
    def best_smoothed_return(self) -> float:
        if self._best_smoothed_return == float("-inf"):
            return float("nan")
        return self._best_smoothed_return

    @property
    def stale_evaluations(self) -> int:
        return self._stale_evaluations

    def update(self, *, step: int, eval_return: float) -> EarlyStoppingDecision:
        self._recent_returns.append(float(eval_return))
        smoothed_return = sum(self._recent_returns) / len(self._recent_returns)
        required_improvement = self._required_improvement()

        if smoothed_return > self._best_smoothed_return + required_improvement:
            self._best_smoothed_return = smoothed_return
            self._stale_evaluations = 0
        else:
            self._stale_evaluations += 1

        should_stop = (
            self.enabled
            and step >= self.min_steps
            and self._stale_evaluations >= self.patience_evals
        )
        reason = "plateau" if should_stop else ""
        return EarlyStoppingDecision(
            should_stop=should_stop,
            reason=reason,
            best_smoothed_return=self._best_smoothed_return,
            stale_evaluations=self._stale_evaluations,
        )

    def _required_improvement(self) -> float:
        if self._best_smoothed_return == float("-inf"):
            return 0.0
        return max(1e-8, abs(self._best_smoothed_return) * self.min_delta)


def build_early_stopping_monitor(training_config) -> EarlyStoppingMonitor:
    return EarlyStoppingMonitor(
        enabled=bool(training_config.early_stopping_enabled),
        min_steps=int(training_config.early_stopping_min_steps),
        patience_evals=int(training_config.early_stopping_patience_evals),
        min_delta=float(training_config.early_stopping_min_delta),
        smoothing_window=int(training_config.early_stopping_smoothing_window),
    )
