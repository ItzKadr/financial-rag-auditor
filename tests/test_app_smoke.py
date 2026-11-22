"""Streamlit smoke tests: the dashboard renders without exceptions (trained and cold-start modes)."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

# AppTest resolves relative paths against this file, so point at the repo-level app.py.
APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def _fresh_caches():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def test_app_runs_without_exception() -> None:
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 6


def test_app_cold_start_without_corpus_or_artifacts(tmp_path, monkeypatch) -> None:
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FRA_DATA_DIR", str(tmp_path / "no_data"))
    monkeypatch.setenv("FRA_ARTIFACTS_DIR", str(tmp_path / "no_artifacts"))
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Pages", "Findings", "Covenants verified"} <= labels
