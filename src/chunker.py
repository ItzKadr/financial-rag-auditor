"""Structure-aware semantic chunking.

Rules (in priority order):

1. **Sections are hard boundaries.** A heading element (ARTICLE / SCHEDULE / "10.2 ...")
   opens a new section; chunks never span two sections. The breadcrumb of headings
   (e.g. ``ARTICLE 10 - FINANCIAL COVENANTS > 10.2 Financial condition``) is stored as
   ``chunk.section`` and prepended to the indexed text ("contextual chunk header").
2. **Tables are atomic.** A Markdown table is one unit and is never split (it may exceed
   the budget on its own - the only allowed overflow).
3. **Sentences are merged greedily** under ``max_tokens``; when a chunk is closed, its
   trailing sentences (up to ``overlap_tokens``) are carried into the next chunk of the
   same section so that a clause cut at a boundary is still retrievable with context.
4. **Optional semantic breakpoints.** With ``semantic_breaks=True`` a TF-IDF cosine
   similarity between consecutive sentences is computed; a chunk is also closed before a
   sentence whose similarity to its predecessor is below the ``semantic_percentile``-th
   percentile of the document (a topic shift), provided the chunk has ``min_tokens``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .pdf_parser import ParsedDocument
from .text_utils import count_tokens, split_sentences

LOG = logging.getLogger(__name__)

_LEVEL1_RE = re.compile(r"^(ARTICLE|SCHEDULE|TABLE OF CONTENTS|CREDIT AGREEMENT)", re.IGNORECASE)


@dataclass
class ChunkerConfig:
    max_tokens: int = 160
    overlap_tokens: int = 32
    min_tokens: int = 24
    semantic_breaks: bool = False
    semantic_percentile: float = 20.0


@dataclass
class Unit:
    text: str
    page: int
    kind: str  # "sentence" | "table"
    section: str
    tokens: int


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    page_start: int
    page_end: int
    n_tokens: int
    has_table: bool
    order: int
    unit_texts: list[str] = field(default_factory=list, repr=False)

    @property
    def context_text(self) -> str:
        """Text that is actually indexed: section breadcrumb + body."""
        return f"[{self.section}]\n{self.text}" if self.section else self.text

    @property
    def pages(self) -> list[int]:
        return list(range(self.page_start, self.page_end + 1))

    @property
    def page_label(self) -> str:
        return f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"


def _units_from_document(doc: ParsedDocument) -> list[Unit]:
    units: list[Unit] = []
    level1, level2 = "", ""
    for el in doc.elements:
        if el.kind == "heading":
            if _LEVEL1_RE.match(el.text):
                level1, level2 = el.text, ""
            else:
                level2 = el.text
            continue
        section = " > ".join(s for s in (level1, level2) if s)
        if el.kind == "table":
            units.append(Unit(el.text, el.page, "table", section, count_tokens(el.text)))
            continue
        for sent in split_sentences(el.text):
            units.append(Unit(sent, el.page, "sentence", section, count_tokens(sent)))
    return units


def _split_long_sentence(unit: Unit, max_tokens: int) -> list[Unit]:
    words = unit.text.split()
    out: list[Unit] = []
    current: list[str] = []
    for w in words:
        if current and count_tokens(" ".join(current + [w])) > max_tokens:
            text = " ".join(current)
            out.append(Unit(text, unit.page, "sentence", unit.section, count_tokens(text)))
            current = []
        current.append(w)
    if current:
        text = " ".join(current)
        out.append(Unit(text, unit.page, "sentence", unit.section, count_tokens(text)))
    return out


def semantic_breakpoints(units: list[Unit], percentile: float) -> set[int]:
    """Indices ``i`` such that a topic shift occurs between unit ``i-1`` and ``i``.

    Similarity is the cosine of TF-IDF vectors (rows are L2-normalised, so the dot
    product equals the cosine). Breakpoints are similarities below the given percentile.
    """
    if len(units) < 3:
        return set()
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english")
    try:
        X = vec.fit_transform([u.text for u in units])
    except ValueError:  # empty vocabulary
        return set()
    sims = np.asarray(X[1:].multiply(X[:-1]).sum(axis=1)).ravel()
    threshold = float(np.percentile(sims, percentile))
    return {i + 1 for i, s in enumerate(sims) if s < threshold}


def chunk_document(doc: ParsedDocument, cfg: ChunkerConfig | None = None) -> list[Chunk]:
    """Chunk a parsed document according to the rules in the module docstring."""
    cfg = cfg or ChunkerConfig()
    raw = _units_from_document(doc)
    units: list[Unit] = []
    for u in raw:
        if u.kind == "sentence" and u.tokens > cfg.max_tokens:
            units.extend(_split_long_sentence(u, cfg.max_tokens))
        else:
            units.append(u)
    breaks = semantic_breakpoints(units, cfg.semantic_percentile) if cfg.semantic_breaks else set()

    chunks: list[Chunk] = []

    def emit(group: list[Unit]) -> None:
        parts: list[str] = []
        for u in group:
            if u.kind == "table":
                parts.append(("\n" if parts else "") + u.text + "\n")
            else:
                parts.append(u.text)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", " ".join(parts)).strip()
        order = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{order:04d}",
                doc_id=doc.doc_id,
                text=text,
                section=group[0].section,
                page_start=min(u.page for u in group),
                page_end=max(u.page for u in group),
                n_tokens=sum(u.tokens for u in group),
                has_table=any(u.kind == "table" for u in group),
                order=order,
                unit_texts=[u.text for u in group],
            )
        )

    current: list[Unit] = []
    cur_tokens = 0
    n_new = 0  # units in `current` that are not overlap carried from the previous chunk
    for i, u in enumerate(units):
        if current and u.section != current[0].section:
            if n_new:
                emit(current)
            current, cur_tokens, n_new = [], 0, 0
        topic_shift = i in breaks and cur_tokens >= cfg.min_tokens
        if n_new and (cur_tokens + u.tokens > cfg.max_tokens or topic_shift):
            emit(current)
            tail: list[Unit] = []
            acc = 0
            if not topic_shift:
                for v in reversed(current):
                    if v.kind != "sentence" or acc + v.tokens > cfg.overlap_tokens:
                        break
                    tail.insert(0, v)
                    acc += v.tokens
                if acc + u.tokens > cfg.max_tokens:
                    tail, acc = [], 0
            current, cur_tokens, n_new = tail, acc, 0
        current.append(u)
        cur_tokens += u.tokens
        n_new += 1
    if n_new:
        emit(current)
    LOG.debug("chunked %s into %d chunks", doc.doc_id, len(chunks))
    return chunks
