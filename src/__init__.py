"""Financial AI Agent & RAG Auditing Pipeline.

Modules
-------
synthetic    generator of multi-page credit agreement PDFs + Master Ledger
pdf_parser   layout-aware PDF parsing (reading order, tables -> Markdown, header/footer strip)
chunker      structure-aware semantic chunking
retriever    hybrid BM25 + dense (FAISS) retrieval with RRF / weighted fusion
schemas      Pydantic v2 models used for structured outputs and audit findings
guardrails   anti-hallucination verification of extracted values
providers    OpenAI-compatible LLM provider configuration (OpenAI / DeepSeek / Groq)
extractor    offline rule-based extractor and LLM structured-output extractor
auditor      reconciliation of covenants against the Master Ledger
evaluation   retrieval / extraction / hallucination / breach-detection metrics
pipeline     glue code shared by train.py and app.py
tracking     fail-safe MLflow logging
"""

from __future__ import annotations

import os

# faiss and torch both ship an OpenMP runtime on Windows; allow them to co-exist.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "1.0.0"
