from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class RunRecorder:
    run_dir: Path
    metrics: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    detector_windows: list[dict] = field(default_factory=list)
    captured_windows: list[dict] = field(default_factory=list)
    mcts_traces: list[dict] = field(default_factory=list)

    def log_metric(self, payload: dict) -> None:
        self.metrics.append(payload)

    def log_decision(self, payload: dict) -> None:
        self.decisions.append(payload)

    def log_detector_window(self, payload: dict) -> None:
        self.detector_windows.append(payload)

    def log_captured_window(self, payload: dict) -> None:
        self.captured_windows.append(payload)

    def log_mcts_trace(self, payload: dict) -> None:
        self.mcts_traces.append(payload)

    def flush(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.metrics:
            pd.DataFrame(self.metrics).to_csv(self.run_dir / "metrics.csv", index=False)
        if self.decisions:
            pd.DataFrame(self.decisions).to_csv(self.run_dir / "control_decisions.csv", index=False)
        if self.detector_windows:
            pd.DataFrame(self.detector_windows).to_csv(self.run_dir / "detector_windows.csv", index=False)
        if self.captured_windows:
            pd.DataFrame(self.captured_windows).to_csv(
                self.run_dir / "captured_suspicious_windows.csv",
                index=False,
            )
        if self.mcts_traces:
            pd.DataFrame(self.mcts_traces).to_csv(self.run_dir / "mcts_traces.csv", index=False)
