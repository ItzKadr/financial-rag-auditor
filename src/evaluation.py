"""Evaluation: retrieval, extraction, hallucination, guardrail stress test, breach detection.

Retrieval (one query per ground-truth covenant; a chunk is *relevant* if it contains the
covenant's evidence sentence / table row, compared on alphanumeric fingerprints):

.. math::

    \\mathrm{Recall@k} = \\frac{1}{|Q|}\\sum_{q\\in Q}\\mathbb{1}[\\exists d\\in\\mathrm{top}_k(q): d\\ \\text{relevant}],
    \\qquad \\mathrm{MRR} = \\frac{1}{|Q|}\\sum_{q}\\frac{1}{\\mathrm{rank}_q}

Extraction (five fields per covenant: type, operator, threshold, unit, frequency):
precision = correct fields / predicted fields, recall = correct fields / ground-truth
fields, F1 = harmonic mean. Hallucination rate = share of predicted covenants whose
(type, threshold) pair does not exist in the contract.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .auditor import normalize_name
from .guardrails import GuardrailConfig, SourceText, sources_from_chunks, verify_covenant
from .pipeline import ProcessedContract
from .retriever import covenant_query
from .schemas import (
    AuditFinding,
    Covenant,
    CovenantType,
    ExtractionResult,
    FindingStatus,
    Frequency,
    GuardrailCheck,
    GuardrailStatus,
    LoanTerms,
    Operator,
    Unit,
    VerifiedCovenant,
)
from .text_utils import compact

LOG = logging.getLogger(__name__)

FIELDS = ("covenant_type", "operator", "threshold", "unit", "test_frequency")


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 with the 0/0 := 0 convention."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------
def relevant_chunk_ids(processed: ProcessedContract, evidence: str) -> set[str]:
    key = compact(evidence)
    return {c.chunk_id for c in processed.chunks if key and key in compact(c.text)}


def ranking_metrics(
    rankings: list[list[str]], relevant: list[set[str]], ks: Iterable[int] = (1, 3, 5), mrr_cutoff: int = 10
) -> dict[str, float]:
    """Recall@k (hit rate of the single information need) and MRR@cutoff."""
    ks = tuple(ks)
    hits = {k: 0.0 for k in ks}
    rr = 0.0
    n = len(rankings)
    for ranking, rel in zip(rankings, relevant):
        first = next((i for i, d in enumerate(ranking, start=1) if d in rel), None)
        for k in ks:
            hits[k] += 1.0 if first is not None and first <= k else 0.0
        if first is not None and first <= mrr_cutoff:
            rr += 1.0 / first
    out = {f"recall@{k}": hits[k] / n if n else 0.0 for k in ks}
    out["mrr"] = rr / n if n else 0.0
    return out


def evaluate_retrieval(
    processed: dict[str, ProcessedContract],
    ground_truth: dict[str, dict],
    contract_ids: list[str],
    method: str,
    alpha: float = 0.5,
    ks: Iterable[int] = (1, 3, 5),
    expand: bool = True,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rankings: list[list[str]] = []
    relevant: list[set[str]] = []
    rows: list[dict[str, Any]] = []
    for cid in contract_ids:
        p = processed[cid]
        assert p.index is not None
        for cov in ground_truth[cid]["covenants"]:
            ctype = CovenantType(cov["covenant_type"])
            rel = relevant_chunk_ids(p, cov["evidence"])
            ranking = p.index.ranking(covenant_query(ctype, expand), method=method, alpha=alpha)
            rankings.append(ranking)
            relevant.append(rel)
            first = next((i for i, d in enumerate(ranking, start=1) if d in rel), None)
            rows.append({"contract_id": cid, "covenant_type": ctype.value, "presentation": cov["presentation"],
                         "n_relevant": len(rel), "first_relevant_rank": first})
    metrics = ranking_metrics(rankings, relevant, ks)
    metrics["evidence_coverage"] = float(np.mean([len(r) > 0 for r in relevant])) if relevant else 0.0
    return metrics, rows


def tune_fusion(
    processed: dict[str, ProcessedContract],
    ground_truth: dict[str, dict],
    dev_ids: list[str],
    alphas: Iterable[float],
    methods: Iterable[str] = ("weighted", "rrf"),
    expansions: Iterable[bool] = (False, True),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grid search over (query expansion, fusion method, alpha) maximising dev MRR.

    Plateau-aware selection: among configurations with the best dev MRR (and Recall@3)
    prefer the one whose *neighbourhood* (alpha +/- ``smooth_radius``) also scores well,
    i.e. the centre of a plateau rather than its edge next to a cliff; remaining ties go
    to the more balanced fusion (alpha closest to 0.5), then to no expansion (simpler).
    """
    alphas = list(alphas)
    smooth_radius = 0.2
    curve: list[dict[str, Any]] = []
    for expand in expansions:
        for method in methods:
            for alpha in alphas:
                m, _ = evaluate_retrieval(processed, ground_truth, dev_ids, method, float(alpha), expand=expand)
                curve.append({"expand": bool(expand), "method": method, "alpha": round(float(alpha), 3), **m})
    for row in curve:
        neigh = [
            r["mrr"] for r in curve
            if r["expand"] == row["expand"] and r["method"] == row["method"]
            and abs(r["alpha"] - row["alpha"]) <= smooth_radius + 1e-9
        ]
        row["mrr_smoothed"] = float(np.mean(neigh))
    best = max(
        curve,
        key=lambda r: (round(r["mrr"], 9), round(r["recall@3"], 9), round(r["mrr_smoothed"], 9),
                       -abs(r["alpha"] - 0.5), not r["expand"]),
    )
    return dict(best), curve


# --------------------------------------------------------------------------------------
# Ground truth helpers
# --------------------------------------------------------------------------------------
def gt_covenant(cov: dict) -> Covenant:
    return Covenant(
        covenant_type=CovenantType(cov["covenant_type"]),
        operator=Operator(cov["operator"]),
        threshold=float(cov["threshold"]),
        unit=Unit(cov["unit"]),
        test_frequency=Frequency(cov["test_frequency"]),
        source_page=int(cov["page"]),
        evidence_quote=cov["evidence"],
    )


def gt_to_extraction(gt: dict) -> ExtractionResult:
    """Reference 'perfect extraction' used to derive ground-truth breach labels."""
    verified = [
        VerifiedCovenant(
            covenant=gt_covenant(c),
            check=GuardrailCheck(status=GuardrailStatus.VERIFIED, quote_match="exact", match_score=1.0,
                                 number_grounded=True, keyword_ok=True, located_page=int(c["page"])),
        )
        for c in gt["covenants"]
    ]
    lt = gt["loan_terms"]
    return ExtractionResult(
        contract_id=gt["contract_id"],
        extractor="ground-truth",
        loan_terms=LoanTerms(
            borrower=lt["borrower"], borrower_quote=f"(1) {lt['borrower']}", principal=None, currency=None,
            principal_quote=None, rate_type=None, base_rate=None, rate_pct=None, rate_quote=None,
            maturity_date=None, maturity_quote=None,
        ),
        covenants=verified,
    )


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------
def _field_equal(pred: Covenant, gt: dict, field: str) -> bool:
    if field == "threshold":
        return math.isclose(pred.threshold, float(gt["threshold"]), rel_tol=1e-6)
    value = getattr(pred, field)
    return (value.value if hasattr(value, "value") else value) == gt[field]


def extraction_metrics(results: list[ExtractionResult], ground_truth: dict[str, dict]) -> dict[str, float]:
    tp = fp = fn = 0
    correct_fields = pred_fields = gt_fields = 0
    exact_numeric = n_gt = 0
    hallucinated = n_pred = 0
    raw_hallucinated = n_raw = 0
    rejected = 0
    for res in results:
        gt = ground_truth[res.contract_id]
        gt_by_type = {c["covenant_type"]: c for c in gt["covenants"]}
        gt_pairs = {(c["covenant_type"], round(float(c["threshold"]), 6)) for c in gt["covenants"]}
        n_gt += len(gt_by_type)
        gt_fields += len(FIELDS) * len(gt_by_type)
        for vc in res.covenants:
            n_raw += 1
            pair = (vc.covenant.covenant_type.value, round(vc.covenant.threshold, 6))
            raw_hallucinated += pair not in gt_pairs
            rejected += not vc.accepted
        used: set[str] = set()
        for vc in res.accepted_covenants():
            cov = vc.covenant
            t = cov.covenant_type.value
            n_pred += 1
            pred_fields += len(FIELDS)
            hallucinated += (t, round(cov.threshold, 6)) not in gt_pairs
            g = gt_by_type.get(t)
            if g is None or t in used:
                fp += 1
                continue
            used.add(t)
            tp += 1
            correct_fields += sum(_field_equal(cov, g, f) for f in FIELDS)
            exact_numeric += _field_equal(cov, g, "threshold")
        fn += len(gt_by_type) - len(used)
    cp, cr, cf = prf(tp, fp, fn)
    fp_ = correct_fields / pred_fields if pred_fields else 0.0
    fr_ = correct_fields / gt_fields if gt_fields else 0.0
    ff_ = 2 * fp_ * fr_ / (fp_ + fr_) if fp_ + fr_ else 0.0
    return {
        "covenant_precision": cp,
        "covenant_recall": cr,
        "covenant_f1": cf,
        "field_precision": fp_,
        "field_recall": fr_,
        "field_f1": ff_,
        "numeric_exact_match": exact_numeric / n_gt if n_gt else 0.0,
        "hallucination_rate": hallucinated / n_pred if n_pred else 0.0,
        "raw_hallucination_rate": raw_hallucinated / n_raw if n_raw else 0.0,
        "guardrail_rejections": float(rejected),
        "n_gt_covenants": float(n_gt),
        "n_accepted_covenants": float(n_pred),
    }


def recall_by_phrasing(results: list[ExtractionResult], ground_truth: dict[str, dict]) -> dict[str, float]:
    """Exact-match recall (type + threshold + operator) split by how the clause was written."""
    hits: dict[str, list[bool]] = {}
    for res in results:
        accepted = {
            (vc.covenant.covenant_type.value, round(vc.covenant.threshold, 6), vc.covenant.operator.value)
            for vc in res.accepted_covenants()
        }
        for c in ground_truth[res.contract_id]["covenants"]:
            key = (c["covenant_type"], round(float(c["threshold"]), 6), c["operator"])
            hits.setdefault(c.get("phrasing", "standard"), []).append(key in accepted)
    return {f"recall_{k}": float(np.mean(v)) for k, v in sorted(hits.items())} | {
        f"n_{k}": float(len(v)) for k, v in sorted(hits.items())
    }


def loan_terms_metrics(results: list[ExtractionResult], ground_truth: dict[str, dict]) -> dict[str, float]:
    fields = ("borrower", "principal", "currency", "rate_type", "rate_pct", "maturity_date")
    correct = {f: 0 for f in fields}
    for res in results:
        gt = ground_truth[res.contract_id]["loan_terms"]
        lt = res.loan_terms
        if lt is None:
            continue
        correct["borrower"] += bool(lt.borrower) and normalize_name(lt.borrower) == normalize_name(gt["borrower"])
        correct["principal"] += lt.principal is not None and math.isclose(lt.principal, gt["principal"], rel_tol=1e-9)
        correct["currency"] += lt.currency == gt["currency"]
        correct["rate_type"] += lt.rate_type is not None and lt.rate_type.value == gt["rate_type"]
        correct["rate_pct"] += lt.rate_pct is not None and math.isclose(lt.rate_pct, gt["rate_pct"], rel_tol=1e-9)
        correct["maturity_date"] += lt.maturity_date == gt["maturity_date"]
    n = max(1, len(results))
    out = {f"loan_{f}_acc": correct[f] / n for f in fields}
    out["loan_terms_acc"] = float(np.mean(list(out.values())))
    return out


# --------------------------------------------------------------------------------------
# Breach detection
# --------------------------------------------------------------------------------------
def breach_metrics(pred: list[AuditFinding], gt: list[AuditFinding]) -> dict[str, float]:
    pred_b = {f.key() for f in pred if f.status == FindingStatus.BREACH}
    gt_b = {f.key() for f in gt if f.status == FindingStatus.BREACH}
    tp, fp, fn = len(pred_b & gt_b), len(pred_b - gt_b), len(gt_b - pred_b)
    p, r, f = prf(tp, fp, fn)
    pred_status = {f.key(): f.status for f in pred}
    agree = sum(pred_status.get(f.key()) == f.status for f in gt)
    return {
        "breach_precision": p,
        "breach_recall": r,
        "breach_f1": f,
        "breach_tp": float(tp),
        "breach_fp": float(fp),
        "breach_fn": float(fn),
        "status_agreement": agree / len(gt) if gt else 0.0,
        "n_gt_breaches": float(len(gt_b)),
    }


# --------------------------------------------------------------------------------------
# Guardrail stress test
# --------------------------------------------------------------------------------------
@dataclass
class StressCase:
    kind: str
    covenant: Covenant
    expect_reject: bool


def _perturb(value: float, unit: str, rng: np.random.Generator) -> float:
    if unit == Unit.RATIO.value:
        for _ in range(20):
            new = round(value + float(rng.choice([-0.5, -0.25, 0.25, 0.5, 0.75])), 2)
            if new > 0 and not math.isclose(new, value):
                return new
        return round(value * 1.3, 2)
    return float(round(value * float(rng.choice([0.8, 0.9, 1.1, 1.25])) / 50_000) * 50_000)


def _render_value(value: float, unit: str) -> str:
    return f"{value:.2f}x" if unit == Unit.RATIO.value else f"USD {value:,.0f}"


def build_stress_cases(gt: dict, rng: np.random.Generator) -> list[StressCase]:
    """Genuine covenants + three families of synthetic hallucinations for one contract."""
    cases: list[StressCase] = []
    min_ratio = [CovenantType.MIN_DSCR, CovenantType.MIN_INTEREST_COVERAGE, CovenantType.MIN_CURRENT_RATIO]
    for c in gt["covenants"]:
        base = gt_covenant(c)
        cases.append(StressCase("genuine", base, False))
        wrong = _perturb(base.threshold, c["unit"], rng)
        cases.append(StressCase("numeric_perturbation", base.model_copy(update={"threshold": wrong}), True))
        near_copy = c["evidence"].replace(c["surface"], _render_value(wrong, c["unit"]))
        cases.append(StressCase("near_copy_quote", base.model_copy(update={"threshold": wrong, "evidence_quote": near_copy}), True))
        fabricated = (
            f"The Borrower undertakes that the {c['covenant_type'].replace('_', ' ')} test level is "
            f"{_render_value(wrong, c['unit'])} for every reporting date under the Facility."
        )
        cases.append(StressCase("fabricated_quote", base.model_copy(update={"threshold": wrong, "evidence_quote": fabricated}), True))
        if base.covenant_type in min_ratio:
            others = [t for t in min_ratio if t != base.covenant_type]
            swapped = others[int(rng.integers(len(others)))]
            cases.append(StressCase("type_swap", base.model_copy(update={"covenant_type": swapped}), True))
    return cases


def guardrail_stress_test(
    processed: dict[str, ProcessedContract],
    ground_truth: dict[str, dict],
    contract_ids: list[str],
    seed: int = 0,
    cfg: GuardrailConfig | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Feed genuine and hallucinated covenants through the guardrail (sources = whole contract)."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for cid in contract_ids:
        p = processed[cid]
        sources: list[SourceText] = sources_from_chunks(p.chunks)
        for case in build_stress_cases(ground_truth[cid], rng):
            check = verify_covenant(case.covenant, sources, p.page_texts, cfg)
            rows.append({
                "contract_id": cid,
                "kind": case.kind,
                "covenant_type": case.covenant.covenant_type.value,
                "expect_reject": case.expect_reject,
                "rejected": not check.accepted,
                "quote_match": check.quote_match,
                "reasons": "; ".join(check.reasons),
            })
    metrics: dict[str, float] = {}
    kinds = sorted({r["kind"] for r in rows})
    for k in kinds:
        sub = [r for r in rows if r["kind"] == k]
        rate = float(np.mean([r["rejected"] for r in sub]))
        metrics[f"stress_{k}_n"] = float(len(sub))
        if k == "genuine":
            metrics["stress_false_rejection_rate"] = rate
        else:
            metrics[f"stress_{k}_detection"] = rate
    halluc = [r for r in rows if r["expect_reject"]]
    metrics["stress_hallucination_detection"] = float(np.mean([r["rejected"] for r in halluc])) if halluc else 0.0
    return metrics, rows
