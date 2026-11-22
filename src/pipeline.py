"""Glue code shared by ``train.py`` and ``app.py``: parse -> chunk -> index -> extract -> audit."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .auditor import AuditConfig, audit_portfolio
from .chunker import Chunk, ChunkerConfig, chunk_document
from .extractor import BaseExtractor, ExtractorConfig
from .guardrails import GuardrailConfig
from .pdf_parser import ParsedDocument, ParserConfig, parse_pdf
from .retriever import Embedder, HybridIndex, RetrieverConfig, make_embedder
from .schemas import AuditFinding, ExtractionResult

LOG = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    parser: ParserConfig = field(default_factory=ParserConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    embedder: str = "lsa"
    lsa_components: int = 128
    seed: int = 42


@dataclass
class ProcessedContract:
    contract_id: str
    parsed: ParsedDocument
    chunks: list[Chunk]
    index: HybridIndex | None = None

    @property
    def page_texts(self) -> dict[int, str]:
        return self.parsed.page_texts()


def process_pdf(contract_id: str, pdf: bytes, cfg: PipelineConfig) -> ProcessedContract:
    parsed = parse_pdf(pdf, doc_id=contract_id, cfg=cfg.parser)
    return ProcessedContract(contract_id, parsed, chunk_document(parsed, cfg.chunker))


def process_pdfs(pdfs: dict[str, bytes], cfg: PipelineConfig) -> dict[str, ProcessedContract]:
    return {cid: process_pdf(cid, pdf, cfg) for cid, pdf in pdfs.items()}


def fit_embedder(processed: Iterable[ProcessedContract], cfg: PipelineConfig) -> Embedder:
    """Fit the dense embedder on the chunks of the given (dev) contracts."""
    texts = [c.context_text for p in processed for c in p.chunks]
    emb = make_embedder(cfg.embedder, n_components=cfg.lsa_components, seed=cfg.seed)
    return emb.fit(texts)


def build_indices(processed: dict[str, ProcessedContract], embedder: Embedder, cfg: PipelineConfig) -> None:
    for p in processed.values():
        p.index = HybridIndex(p.chunks, embedder, cfg.retriever)


def extract_all(processed: dict[str, ProcessedContract], extractor: BaseExtractor) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    for cid, p in processed.items():
        if p.index is None:
            raise RuntimeError(f"contract {cid} has no index; call build_indices first")
        results.append(extractor.extract(cid, p.index, p.page_texts))
    return results


def audit(results: list[ExtractionResult], ledger: pd.DataFrame, cfg: PipelineConfig) -> list[AuditFinding]:
    return audit_portfolio(results, ledger, cfg.audit)
