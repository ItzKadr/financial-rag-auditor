from src.chunker import ChunkerConfig, chunk_document
from src.text_utils import compact, split_sentences


def test_chunk_budget_and_ids(processed, pipeline_cfg) -> None:
    budget = pipeline_cfg.chunker.max_tokens
    for p in processed.values():
        ids = [c.chunk_id for c in p.chunks]
        assert len(ids) == len(set(ids))
        assert [c.order for c in p.chunks] == list(range(len(p.chunks)))
        for c in p.chunks:
            # only an atomic table may exceed the budget on its own
            assert c.n_tokens <= budget or (c.has_table and len(c.unit_texts) == 1)
            assert c.page_start <= c.page_end


def test_tables_are_never_split(processed) -> None:
    for p in processed.values():
        for table in (e for e in p.parsed.elements if e.kind == "table"):
            holders = [c for c in p.chunks if table.text in c.text]
            assert len(holders) == 1, "a table must appear intact in exactly one chunk"


def test_chunks_do_not_cross_sections(processed) -> None:
    for p in processed.values():
        for c in p.chunks:
            assert c.section == c.section.strip()
        # consecutive chunks of different sections never share units
        for a, b in zip(p.chunks, p.chunks[1:]):
            if a.section != b.section:
                assert not set(a.unit_texts) & set(b.unit_texts)


def test_all_text_is_covered(processed) -> None:
    for p in processed.values():
        corpus_text = compact(" ".join(c.text for c in p.chunks))
        for el in p.parsed.elements:
            if el.kind == "text":
                for s in split_sentences(el.text):
                    assert compact(s) in corpus_text


def test_overlap_carries_trailing_sentences(processed, first_id) -> None:
    doc = processed[first_id].parsed
    chunks = chunk_document(doc, ChunkerConfig(max_tokens=60, overlap_tokens=25))
    shared = 0
    for a, b in zip(chunks, chunks[1:]):
        if a.section == b.section and not a.has_table:
            shared += bool(set(a.unit_texts) & set(b.unit_texts))
    assert shared > 0
    no_overlap = chunk_document(doc, ChunkerConfig(max_tokens=60, overlap_tokens=0))
    for a, b in zip(no_overlap, no_overlap[1:]):
        assert not set(a.unit_texts) & set(b.unit_texts)


def test_semantic_breaks_only_add_boundaries(processed, first_id) -> None:
    doc = processed[first_id].parsed
    base = chunk_document(doc, ChunkerConfig(overlap_tokens=0))
    sem = chunk_document(doc, ChunkerConfig(overlap_tokens=0, semantic_breaks=True, semantic_percentile=30))
    assert len(sem) >= len(base)
    assert compact("".join(c.text for c in sem)) == compact("".join(c.text for c in base))
