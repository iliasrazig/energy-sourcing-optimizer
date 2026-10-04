"""Experiment tracking seam.

Only a no-op tracker exists for now. Experiment tracking (e.g. MLflow) will be
added later as another :class:`ExperimentTracker` implementation, without
touching the trainer or the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol


class RunLogger(Protocol):
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...
    def set_tags(self, tags: Mapping[str, str]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...


class ExperimentTracker(ABC):
    @abstractmethod
    @contextmanager
    def start_run(self, experiment: str, run_name: str) -> Iterator[RunLogger]:
        """Open a run inside ``experiment``; closes it even when the body raises."""


class _NullRun:
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...
    def set_tags(self, tags: Mapping[str, str]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...


class NullTracker(ExperimentTracker):
    """No-op tracker (Null Object pattern)."""

    @contextmanager
    def start_run(self, experiment: str, run_name: str) -> Iterator[RunLogger]:
        yield _NullRun()
