import faiss
import numpy as np
import pytest

from src.retriever import (
    LSAEmbedder,
    bm25_tokenize,
    covenant_query,
    l2_normalize,
    minmax,
    ranks_from_scores,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from src.schemas import CovenantType


def test_rrf_hand_computed() -> None:
    # list1: a, b, c ; list2: b, c, a   (k = 60)
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert sorted(scores, key=scores.get, reverse=True) == ["b", "a", "c"]


def test_rrf_weights_and_swapped_ties() -> None:
    # symmetric swap -> exact tie with equal weights; weights break it
    eq = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)
    assert eq["x"] == pytest.approx(eq["y"])
    w = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60, weights=[1.4, 0.6])
    assert w["x"] > w["y"]


def test_weighted_fusion_and_minmax() -> None:
    bm = np.array([10.0, 5.0, 0.0])
    de = np.array([0.1, 0.9, 0.5])
    np.testing.assert_allclose(minmax(bm), [1.0, 0.5, 0.0])
    np.testing.assert_allclose(weighted_fusion(bm, de, 0.0), minmax(bm))
    np.testing.assert_allclose(weighted_fusion(bm, de, 1.0), minmax(de))
    np.testing.assert_allclose(weighted_fusion(bm, de, 0.5), [0.5, 0.75, 0.25])
    np.testing.assert_allclose(minmax(np.array([3.0, 3.0])), [0.0, 0.0])


def test_ranks_from_scores() -> None:
    np.testing.assert_array_equal(ranks_from_scores(np.array([0.2, 0.9, 0.5])), [3, 1, 2])


def test_l2_normalised_embeddings_make_ip_equal_cosine() -> None:
    rng = np.random.default_rng(0)
    X = l2_normalize(rng.normal(size=(20, 16)))
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, rtol=1e-5)
    index = faiss.IndexFlatIP(16)
    index.add(X)
    sims, idx = index.search(X[:1], 1)
    assert idx[0, 0] == 0 and sims[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_lsa_embedder_unit_norm() -> None:
    texts = ["leverage ratio shall not exceed", "interest cover minimum", "net worth at least", "capital expenditure limit"]
    emb = LSAEmbedder(n_components=8).fit(texts)
    E = emb.encode(texts)
    assert E.shape[0] == 4
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, rtol=1e-5)


def test_bm25_tokenizer() -> None:
    # stop-words dropped, "not" kept (negation matters), plural stripped, decimals kept whole
    assert bm25_tokenize("The Covenants shall NOT exceed 3.50x") == ["covenant", "not", "exceed", "3.50", "x"]


def test_hybrid_search_finds_covenant_clause(corpus, processed) -> None:
    from src.evaluation import relevant_chunk_ids

    found = 0
    total = 0
    for cid, p in processed.items():
        for cov in corpus.ground_truth[cid]["covenants"]:
            rel = relevant_chunk_ids(p, cov["evidence"])
            hits = p.index.search(covenant_query(CovenantType(cov["covenant_type"])), top_k=5, method="rrf", alpha=0.5)
            assert len(hits) == 5 and [h.rank for h in hits] == [1, 2, 3, 4, 5]
            found += any(h.chunk.chunk_id in rel for h in hits)
            total += 1
    assert found == total  # Recall@5 = 1 on the fixture corpus


def test_search_methods_are_consistent(processed, first_id) -> None:
    index = processed[first_id].index
    q = covenant_query(CovenantType.MAX_LEVERAGE)
    bm = index.search(q, top_k=3, method="bm25")
    assert [h.bm25_rank for h in bm] == [1, 2, 3]
    de = index.search(q, top_k=3, method="dense")
    assert [h.dense_rank for h in de] == [1, 2, 3]
    w0 = index.search(q, top_k=3, method="weighted", alpha=0.0)
    assert [h.chunk.chunk_id for h in w0] == [h.chunk.chunk_id for h in bm]
    with pytest.raises(ValueError):
        index.search(q, method="nope")


@pytest.mark.network
def test_sentence_transformer_embedder_unit_norm() -> None:
    """Needs the all-MiniLM-L6-v2 weights (downloaded once); run with `pytest -m network`."""
    from src.retriever import SentenceTransformerEmbedder

    E = SentenceTransformerEmbedder().encode(["Leverage Ratio shall not exceed 3.50x", "Tangible Net Worth"])
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, rtol=1e-4)
