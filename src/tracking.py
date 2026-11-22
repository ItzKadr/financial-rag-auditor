"""Fail-safe MLflow tracking (local SQLite store inside the repository).

Any MLflow error is logged as a warning and disables further tracking for the run, so the
pipeline (and the files written to ``artifacts/``) never depends on MLflow being healthy.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)
_KEY_RE = re.compile(r"[^0-9A-Za-z_\-. /]")


def sanitize_key(key: str) -> str:
    """MLflow keys allow ``[A-Za-z0-9_-. /]``; e.g. 'recall@3' -> 'recall_at_3'."""
    return _KEY_RE.sub("_", key.replace("@", "_at_"))


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


class MLflowTracker:
    def __init__(self, repo_dir: Path, experiment: str, run_name: str | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.run_name = run_name
        self._mlflow = None
        if not enabled:
            return
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow

            logging.getLogger("mlflow").setLevel(logging.WARNING)
            logging.getLogger("alembic").setLevel(logging.WARNING)
            mlflow.set_tracking_uri(f"sqlite:///{(Path(repo_dir) / 'mlflow.db').as_posix()}")
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(experiment, artifact_location=(Path(repo_dir) / "mlruns").resolve().as_uri())
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
        except Exception as exc:
            LOG.warning("MLflow disabled (%s)", exc)
            self.enabled = False

    def _safe(self, fn: str, *args: Any, **kwargs: Any) -> None:
        if not self.enabled or self._mlflow is None:
            return
        try:
            getattr(self._mlflow, fn)(*args, **kwargs)
        except Exception as exc:
            LOG.warning("MLflow %s failed (%s); tracking disabled for this run", fn, exc)
            self.enabled = False

    def __enter__(self) -> "MLflowTracker":
        self._safe("start_run", run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._safe("end_run", status="FAILED" if exc_type else "FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {sanitize_key(k): str(v)[:5000] for k, v in flatten(params).items()}
        self._safe("log_params", flat)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        clean = {}
        for k, v in flatten(metrics).items():
            if isinstance(v, (int, float)) and v == v and abs(float(v)) != float("inf"):
                clean[sanitize_key(k)] = float(v)
        self._safe("log_metrics", clean, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._safe("log_metric", sanitize_key(key), float(value), step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        self._safe("set_tags", tags)

    def log_artifacts(self, path: Path) -> None:
        self._safe("log_artifacts", str(path))
