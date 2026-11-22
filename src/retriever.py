"""Hybrid retrieval: BM25 (sparse) + FAISS inner-product search (dense) + fusion.

Dense side
    Embeddings are L2-normalised so that inner product == cosine similarity and a FAISS
    ``IndexFlatIP`` performs exact cosine search. The default embedder is fully offline:
    TF-IDF (uni+bi-grams) followed by TruncatedSVD = Latent Semantic Analysis (LSA).
    ``sentence-transformers`` (``all-MiniLM-L6-v2``) can be used instead (``--embedder st``).

Sparse side
    Okapi BM25 (rank-bm25):

    .. math:: \\mathrm{BM25}(q,d)=\\sum_{t\\in q}\\mathrm{IDF}(t)\\,
              \\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\\,(1-b+b\\,|d|/\\mathrm{avgdl})}

Fusion
    * Reciprocal Rank Fusion: :math:`\\mathrm{RRF}(d)=\\sum_i w_i/(k+r_i(d))`, ``k=60``.
    * Weighted score fusion: :math:`\\alpha\\,\\widehat{s}_{dense}+(1-\\alpha)\\,\\widehat{s}_{bm25}`
      with min-max normalised scores.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunker import Chunk
from .schemas import CovenantType
from .text_utils import normalize_text

LOG = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "by", "with", "as", "at",
    "be", "is", "are", "was", "were", "this", "that", "which", "it", "its", "any", "each",
    "from", "such", "shall", "will", "may", "all", "other", "than", "has", "have", "been",
    "under", "into", "if", "so", "no", "per",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

# Query templates with domain expansion (synonyms used by credit lawyers).
COVENANT_QUERIES: dict[CovenantType, str] = {
    CovenantType.MAX_LEVERAGE: "Leverage Ratio Total Debt to EBITDA shall not exceed maximum leverage financial covenant",
    CovenantType.MIN_DSCR: "Debt Service Cover Ratio DSCR EBITDA to Debt Service shall not be less than minimum at least",
    CovenantType.MIN_INTEREST_COVERAGE: "Interest Cover Coverage Ratio EBITDA to Net Finance Charges shall not be less than minimum at least",
    CovenantType.MIN_CURRENT_RATIO: "Current Ratio Current Assets to Current Liabilities shall not be less than minimum liquidity",
    CovenantType.MAX_CAPEX: "Capital Expenditure in any Financial Year shall not exceed maximum aggregate capex limit",
    CovenantType.MIN_NET_WORTH: "Tangible Net Worth shall not be less than minimum at all times net worth",
}
# Obligation-context expansion: vocabulary of the LMA-style "Financial Condition" clause /
# financial covenant schedule. It pushes obligation clauses above *definitions* that merely
# name the same ratio (the dominant rank-1 error without expansion).
QUERY_EXPANSION = "Borrower shall ensure financial condition financial covenant levels requirement"


def covenant_query(ctype: CovenantType, expand: bool = True) -> str:
    """Query template for a covenant type, optionally with obligation-context expansion."""
    base = COVENANT_QUERIES[ctype]
    return f"{base} {QUERY_EXPANSION}" if expand else base


LOAN_TERM_QUERIES: dict[str, str] = {
    "borrower": "agreement dated made between Borrower Lender parties incorporated",
    "principal": "Lender makes available to the Borrower term loan facility aggregate amount equal to",
    "rate": "rate of interest Margin per cent per annum fixed rate Loan shall bear interest",
    "maturity": "Final Maturity Date means the date falling months after being",
    "frequency": "financial covenants tested quarterly semi-annually Quarter Date Half-Year Date Compliance Certificate",
}


def bm25_tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens, stop-words removed, naive plural stripping."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(normalize_text(text)):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss") and tok.isalpha():
            tok = tok[:-1]
        out.append(tok)
    return out


# --------------------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------------------
class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> "Embedder": ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(norms, eps)).astype(np.float32)


class LSAEmbedder:
    """TF-IDF -> TruncatedSVD (Latent Semantic Analysis), L2-normalised. Fully offline."""

    def __init__(self, n_components: int = 128, seed: int = 0) -> None:
        self.name = f"lsa-{n_components}"
        self.n_components = n_components
        self.seed = seed
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: Sequence[str]) -> "LSAEmbedder":
        self.vectorizer = TfidfVectorizer(
            tokenizer=bm25_tokenize, token_pattern=None, lowercase=False, ngram_range=(1, 2),
            sublinear_tf=True, min_df=1,
        )
        X = self.vectorizer.fit_transform(texts)
        k = int(max(2, min(self.n_components, X.shape[0] - 1, X.shape[1] - 1)))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self.svd.fit(X)
        LOG.info("LSA fitted on %d texts: vocab=%d, components=%d, explained var=%.3f",
                 X.shape[0], X.shape[1], k, float(self.svd.explained_variance_ratio_.sum()))
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("LSAEmbedder.encode called before fit")
        return l2_normalize(self.svd.transform(self.vectorizer.transform(texts)))


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers (needs the model weights, i.e. network once)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = model_name.split("/")[-1]
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(emb, dtype=np.float32)


def make_embedder(kind: str = "lsa", n_components: int = 128, seed: int = 0) -> Embedder:
    """'lsa' (default, offline) or 'st' (sentence-transformers; falls back to LSA on failure)."""
    if kind == "st":
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # network / weights unavailable
            LOG.warning("sentence-transformers unavailable (%s); falling back to LSA", exc)
    return LSAEmbedder(n_components=n_components, seed=seed)


# --------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> dict[str, float]:
    """RRF score per id: ``sum_i w_i / (k + rank_i)`` with 1-based ranks."""
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return scores


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def weighted_fusion(bm25_scores: np.ndarray, dense_scores: np.ndarray, alpha: float) -> np.ndarray:
    """``alpha * minmax(dense) + (1 - alpha) * minmax(bm25)``."""
    return alpha * minmax(dense_scores) + (1.0 - alpha) * minmax(bm25_scores)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """1-based ranks (stable; ties broken by original order)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


# --------------------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------------------
@dataclass
class RetrieverConfig:
    method: str = "rrf"  # "rrf" | "weighted" | "bm25" | "dense"
    alpha: float = 0.5  # dense weight (weighted fusion) or dense share of RRF weights
    rrf_k: int = 60
    top_k: int = 5
    expand_queries: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class RetrievalHit:
    chunk: Chunk
    rank: int
    score: float
    bm25_score: float
    dense_score: float
    bm25_rank: int
    dense_rank: int


class HybridIndex:
    """Per-document hybrid index over chunks (BM25 + FAISS IndexFlatIP)."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, cfg: RetrieverConfig | None = None) -> None:
        if not chunks:
            raise ValueError("cannot index an empty chunk list")
        self.cfg = cfg or RetrieverConfig()
        self.chunks = chunks
        self.embedder = embedder
        texts = [c.context_text for c in chunks]
        self.bm25 = BM25Okapi([bm25_tokenize(t) or ["_empty_"] for t in texts], k1=self.cfg.bm25_k1, b=self.cfg.bm25_b)
        emb = embedder.encode(texts)
        self.dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(emb, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.chunks)

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(bm25_tokenize(query)), dtype=float)

    def dense_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query])
        sims, idx = self.index.search(np.ascontiguousarray(q, dtype=np.float32), len(self.chunks))
        out = np.zeros(len(self.chunks), dtype=float)
        out[idx[0]] = sims[0]
        return out

    def search(self, query: str, top_k: int | None = None, method: str | None = None, alpha: float | None = None) -> list[RetrievalHit]:
        top_k = top_k or self.cfg.top_k
        method = method or self.cfg.method
        alpha = self.cfg.alpha if alpha is None else alpha
        bm = self.bm25_scores(query)
        de = self.dense_scores(query)
        bm_rank, de_rank = ranks_from_scores(bm), ranks_from_scores(de)
        if method == "bm25":
            fused = bm
        elif method == "dense":
            fused = de
        elif method == "weighted":
            fused = weighted_fusion(bm, de, alpha)
        elif method == "rrf":
            ids = [str(i) for i in range(len(self.chunks))]
            rrf = reciprocal_rank_fusion(
                [sorted(ids, key=lambda i: bm_rank[int(i)]), sorted(ids, key=lambda i: de_rank[int(i)])],
                k=self.cfg.rrf_k,
                weights=[2 * (1 - alpha), 2 * alpha],
            )
            fused = np.array([rrf[i] for i in ids])
        else:
            raise ValueError(f"unknown retrieval method {method!r}")
        order = np.lexsort((bm_rank, -fused))[:top_k]  # primary: fused desc, tie-break: bm25 rank
        return [
            RetrievalHit(
                chunk=self.chunks[i], rank=r, score=float(fused[i]), bm25_score=float(bm[i]),
                dense_score=float(de[i]), bm25_rank=int(bm_rank[i]), dense_rank=int(de_rank[i]),
            )
            for r, i in enumerate(order, start=1)
        ]

    def ranking(self, query: str, method: str | None = None, alpha: float | None = None) -> list[str]:
        """Full ranking of chunk ids (used for evaluation)."""
        return [h.chunk.chunk_id for h in self.search(query, top_k=len(self.chunks), method=method, alpha=alpha)]
