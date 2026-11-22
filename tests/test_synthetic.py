import math

import pymupdf

from src.auditor import audit_portfolio
from src.evaluation import gt_to_extraction
from src.schemas import CovenantType, FindingStatus
from src.synthetic import LEDGER_COLUMNS, SyntheticConfig, fmt_amount, fmt_ratio, generate_corpus
from src.text_utils import compact, parse_numbers


def test_ground_truth_evidence_is_on_the_stated_page(corpus) -> None:
    for c in corpus.contracts:
        doc = pymupdf.open(stream=c.pdf_bytes, filetype="pdf")
        assert len(doc) == c.ground_truth["n_pages"] >= 4
        for cov in c.ground_truth["covenants"]:
            page_text = doc[cov["page"] - 1].get_text()
            # table evidence is "a | b | c" -> compare alphanumeric fingerprints
            assert compact(cov["evidence"]) in compact(page_text), cov
        doc.close()


def test_surface_forms_parse_back_to_threshold(corpus) -> None:
    for gt in corpus.ground_truth.values():
        for cov in gt["covenants"]:
            values = [m.value for m in parse_numbers(cov["surface"])]
            assert any(math.isclose(v, cov["threshold"]) for v in values), cov["surface"]


def test_ledger_shape_and_columns(corpus) -> None:
    led = corpus.ledger
    cfg = corpus.config
    assert len(led) == cfg.n_contracts * cfg.n_quarters
    for col in LEDGER_COLUMNS + ["borrower_id", "borrower_name", "period", "fiscal_year", "quarter"]:
        assert col in led.columns
    assert (led.groupby("borrower_id").size() == cfg.n_quarters).all()


def test_every_injected_breach_is_detected_with_true_covenants() -> None:
    corpus = generate_corpus(SyntheticConfig(n_contracts=10, seed=3, breach_borrower_frac=1.0))
    findings = audit_portfolio([gt_to_extraction(g) for g in corpus.ground_truth.values()], corpus.ledger)
    breaches = {f.key() for f in findings if f.status == FindingStatus.BREACH}
    injected = [(g["contract_id"], e["covenant_type"], e["period"]) for g in corpus.ground_truth.values()
                for e in g["injected_breaches"]]
    assert injected, "breach_borrower_frac=1 must inject breaches"
    assert all(k in breaches for k in injected)


def test_generation_is_deterministic() -> None:
    a = generate_corpus(SyntheticConfig(n_contracts=2, seed=99))
    b = generate_corpus(SyntheticConfig(n_contracts=2, seed=99))
    assert a.ground_truth == b.ground_truth
    assert a.ledger.equals(b.ledger)


def test_formatters() -> None:
    assert fmt_ratio(3.5, "x") == "3.50x"
    assert fmt_ratio(3.5, "to") == "3.5 to 1.0"
    assert fmt_ratio(1.25, "colon") == "1.25:1.00"
    assert fmt_amount(12_500_000, "USD", "million") == "USD 12.5 million"
    assert fmt_amount(12_500_000, "USD", "code_words").startswith("USD 12,500,000 (twelve million five hundred thousand")
    assert CovenantType.MAX_CAPEX.direction == "max"
