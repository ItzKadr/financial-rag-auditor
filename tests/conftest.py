"""Shared fixtures: a tiny synthetic corpus, parsed + chunked + indexed once per session."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from src.pipeline import PipelineConfig, ProcessedContract, build_indices, fit_embedder, process_pdfs
from src.synthetic import Corpus, SyntheticConfig, generate_corpus


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    # holdout phrasing disabled so that rule-based recall can be asserted exactly
    return generate_corpus(SyntheticConfig(n_contracts=4, seed=11, holdout_phrasing_frac=0.0))


@pytest.fixture(scope="session")
def pipeline_cfg() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture(scope="session")
def processed(corpus: Corpus, pipeline_cfg: PipelineConfig) -> dict[str, ProcessedContract]:
    procs = process_pdfs(corpus.pdfs, pipeline_cfg)
    embedder = fit_embedder(procs.values(), pipeline_cfg)
    build_indices(procs, embedder, pipeline_cfg)
    return procs


@pytest.fixture(scope="session")
def first_id(corpus: Corpus) -> str:
    return corpus.contracts[0].contract_id
