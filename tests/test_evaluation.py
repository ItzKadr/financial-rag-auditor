import pytest

from src.evaluation import (
    breach_metrics,
    build_stress_cases,
    evaluate_retrieval,
    extraction_metrics,
    guardrail_stress_test,
    prf,
    ranking_metrics,
)
from src.schemas import (
    AuditFinding,
    Covenant,
    ExtractionResult,
    FindingStatus,
    GuardrailCheck,
    GuardrailStatus,
    VerifiedCovenant,
)


def test_prf() -> None:
    assert prf(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    p, r, f = prf(3, 1, 3)
    assert (p, r) == pytest.approx((0.75, 0.5)) and f == pytest.approx(2 * 0.75 * 0.5 / 1.25)


def test_ranking_metrics_hand_computed() -> None:
    rankings = [["a", "b", "c"], ["x", "y", "z"], ["p", "q", "r"]]
    relevant = [{"a"}, {"z"}, {"nothing"}]
    m = ranking_metrics(rankings, relevant, ks=(1, 3))
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@3"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)


def _vc(ctype: str, op: str, thr: float, unit: str, ok: bool = True) -> VerifiedCovenant:
    cov = Covenant(covenant_type=ctype, operator=op, threshold=thr, unit=unit, test_frequency="quarterly",
                   source_page=1, evidence_quote="placeholder evidence text")
    return VerifiedCovenant(covenant=cov, check=GuardrailCheck(status=GuardrailStatus.VERIFIED if ok else GuardrailStatus.REJECTED))


def test_extraction_metrics_hand_computed() -> None:
    gt = {"C1": {"covenants": [
        {"covenant_type": "max_leverage", "operator": "<=", "threshold": 3.5, "unit": "ratio", "test_frequency": "quarterly"},
        {"covenant_type": "min_dscr", "operator": ">=", "threshold": 1.25, "unit": "ratio", "test_frequency": "quarterly"},
    ]}}
    res = ExtractionResult(contract_id="C1", extractor="t", covenants=[
        _vc("max_leverage", "<=", 3.5, "ratio"),                 # fully correct (5/5 fields)
        _vc("min_dscr", ">", 1.25, "ratio"),                     # wrong operator (4/5)
        _vc("min_current_ratio", ">=", 1.1, "ratio"),            # hallucinated covenant (0/5)
        _vc("min_net_worth", ">=", 5e6, "currency", ok=False),   # rejected by guardrail -> not counted
    ])
    m = extraction_metrics([res], gt)
    assert m["covenant_precision"] == pytest.approx(2 / 3) and m["covenant_recall"] == 1.0
    assert m["field_precision"] == pytest.approx(9 / 15)
    assert m["field_recall"] == pytest.approx(9 / 10)
    assert m["numeric_exact_match"] == 1.0
    assert m["hallucination_rate"] == pytest.approx(1 / 3)
    assert m["raw_hallucination_rate"] == pytest.approx(2 / 4)
    assert m["guardrail_rejections"] == 1


def _finding(cid: str, period: str, status: FindingStatus) -> AuditFinding:
    return AuditFinding(contract_id=cid, borrower="B", period=period, covenant_type="max_leverage", operator="<=",
                        threshold=3.0, unit="ratio", actual_value=1.0, headroom_pct=0.0, status=status, source_page=1,
                        evidence_quote="q")


def test_breach_metrics() -> None:
    B, C = FindingStatus.BREACH, FindingStatus.COMPLIANT
    gt = [_finding("1", "2023Q1", B), _finding("1", "2023Q2", B), _finding("2", "2023Q1", C)]
    pred = [_finding("1", "2023Q1", B), _finding("2", "2023Q1", B)]
    m = breach_metrics(pred, gt)
    assert (m["breach_tp"], m["breach_fp"], m["breach_fn"]) == (1, 1, 1)
    assert m["breach_precision"] == 0.5 and m["breach_recall"] == 0.5
    assert m["status_agreement"] == pytest.approx(1 / 3)


def test_retrieval_eval_and_stress_test_on_fixture(corpus, processed) -> None:
    ids = list(processed)
    m, rows = evaluate_retrieval(processed, corpus.ground_truth, ids, "rrf", 0.5)
    assert m["evidence_coverage"] == 1.0 and m["recall@5"] == 1.0
    assert len(rows) == sum(len(g["covenants"]) for g in corpus.ground_truth.values())

    import numpy as np

    cases = build_stress_cases(corpus.ground_truth[ids[0]], np.random.default_rng(0))
    assert {c.kind for c in cases} >= {"genuine", "numeric_perturbation", "near_copy_quote", "fabricated_quote"}
    sm, _ = guardrail_stress_test(processed, corpus.ground_truth, ids, seed=0)
    assert sm["stress_false_rejection_rate"] == 0.0
    assert sm["stress_hallucination_detection"] == 1.0
