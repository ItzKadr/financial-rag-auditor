"""Anti-hallucination guardrails: every extracted value must be grounded in the source.

For each extracted covenant / loan term the guardrail runs three checks:

1. **Quote grounding** - the evidence quote (normalised: case, whitespace, quotes,
   table pipes) must occur in the *retrieved* source chunks, either exactly or as a
   fuzzy match with ``difflib`` ratio >= ``fuzzy_threshold`` (tolerates OCR / spacing /
   hyphenation noise in LLM copies).
2. **Number grounding** - the extracted number must occur in the *matched source span*
   (not in the quote the model wrote!). Checking the source span is what defeats the
   "near-copy" hallucination: a quote that is 95 % identical to the real sentence but
   with a different number fuzzy-matches the real sentence, whose number then disagrees.
3. **Semantic anchor** - the source span must mention the covenant concept (e.g.
   "leverage" / "Total Debt ... EBITDA" for ``max_leverage``), which catches a correct
   quote attached to the wrong covenant type.

A value failing any check is REJECTED (kept in the report with reasons, never audited).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .chunker import Chunk
from .schemas import (
    Covenant,
    CovenantType,
    GuardrailCheck,
    GuardrailStatus,
    LoanTerms,
    Unit,
    VerifiedCovenant,
)
from .text_utils import compact, normalize_text, parse_dates, parse_numbers

TYPE_KEYWORDS: dict[CovenantType, re.Pattern[str]] = {
    CovenantType.MAX_LEVERAGE: re.compile(r"leverage|total (?:net )?debt\b.{0,60}\bebitda|debt\s*/\s*ebitda"),
    CovenantType.MIN_DSCR: re.compile(r"debt service cover|dscr"),
    CovenantType.MIN_INTEREST_COVERAGE: re.compile(r"interest cover|\bicr\b|ebitda to net finance charges"),
    CovenantType.MIN_CURRENT_RATIO: re.compile(r"current ratio|current assets\b.{0,80}\bcurrent liabilities"),
    CovenantType.MAX_CAPEX: re.compile(r"capital expenditure|\bcapex\b"),
    CovenantType.MIN_NET_WORTH: re.compile(r"net worth"),
}


@dataclass
class GuardrailConfig:
    fuzzy_threshold: float = 0.90
    require_keyword: bool = True
    rel_tol: float = 1e-6
    max_fuzzy_sources: int = 3


@dataclass
class SourceText:
    """A retrieved passage the extractor was allowed to read."""

    text: str
    chunk_id: str | None = None
    page: int | None = None

    def __post_init__(self) -> None:
        self.norm = normalize_text(self.text)


@dataclass
class QuoteMatch:
    kind: str  # "exact" | "fuzzy" | "none"
    score: float
    source: SourceText | None = None
    span: str = ""


def value_grounded(span: str, value: float, kinds: set[str], rel_tol: float = 1e-6) -> bool:
    """True if a *typed* number mention of an allowed kind in ``span`` equals ``value``.

    Typed mentions matter: in "1.25:1.00" the ratio mention is 1.25 and the denominator
    "1.00" is not a separate number, so a hallucinated threshold of 1.0 is not grounded.
    """
    return any(
        m.kind in kinds and abs(m.value - value) <= rel_tol * max(1.0, abs(value)) for m in parse_numbers(span)
    )


def kinds_for(ctype: CovenantType) -> set[str]:
    return {"ratio", "number"} if ctype.expected_unit == Unit.RATIO else {"currency", "number"}


def sources_from_chunks(chunks: list[Chunk]) -> list[SourceText]:
    return [SourceText(c.text, c.chunk_id, c.page_start) for c in chunks]


def _expand_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and text[start - 1] != " ":
        start -= 1
    while end < len(text) and text[end] != " ":
        end += 1
    return start, end


def find_quote(quote: str, sources: list[SourceText], cfg: GuardrailConfig | None = None) -> QuoteMatch:
    """Locate ``quote`` in the sources: exact normalised substring, else best fuzzy window."""
    cfg = cfg or GuardrailConfig()
    q = normalize_text(quote)
    if len(q) < 8:
        return QuoteMatch("none", 0.0)
    for s in sources:
        pos = s.norm.find(q)
        if pos >= 0:
            return QuoteMatch("exact", 1.0, s, s.norm[pos : pos + len(q)])

    # fuzzy: anchor on the longest common block, then score windows around it
    anchored: list[tuple[int, SourceText, int]] = []
    for s in sources:
        sm = difflib.SequenceMatcher(None, q, s.norm, autojunk=False)
        m = sm.find_longest_match(0, len(q), 0, len(s.norm))
        if m.size >= max(6, int(0.15 * len(q))):
            anchored.append((m.size, s, m.b - m.a))
    anchored.sort(key=lambda x: -x[0])
    best = QuoteMatch("none", 0.0)
    for _, s, start0 in anchored[: cfg.max_fuzzy_sources]:
        for shift in (-6, -3, 0, 3, 6):
            for dl in (-6, -3, 0, 3, 6):
                st = max(0, start0 + shift)
                en = min(len(s.norm), st + len(q) + dl)
                if en <= st:
                    continue
                window = s.norm[st:en]
                sm = difflib.SequenceMatcher(None, q, window, autojunk=False)
                if sm.real_quick_ratio() < best.score or sm.quick_ratio() < best.score:
                    continue
                r = sm.ratio()
                if r > best.score:
                    a, b = _expand_to_words(s.norm, st, en)
                    best = QuoteMatch("fuzzy", r, s, s.norm[a:b])
    if best.score >= cfg.fuzzy_threshold:
        return best
    return QuoteMatch("none", best.score, best.source, best.span)


def locate_page(span: str, page_texts: dict[int, str] | None) -> int | None:
    """1-based page whose text contains ``span`` (compared on alphanumeric fingerprints)."""
    if not page_texts or not span:
        return None
    key = compact(span)
    for page, text in sorted(page_texts.items()):
        if key and key in compact(text):
            return page
    return None


def verify_covenant(
    cov: Covenant,
    sources: list[SourceText],
    page_texts: dict[int, str] | None = None,
    cfg: GuardrailConfig | None = None,
) -> GuardrailCheck:
    cfg = cfg or GuardrailConfig()
    match = find_quote(cov.evidence_quote, sources, cfg)
    reasons: list[str] = []
    if match.kind == "none":
        reasons.append(f"evidence quote not found in retrieved context (best fuzzy score {match.score:.2f})")
        return GuardrailCheck(status=GuardrailStatus.REJECTED, quote_match="none", match_score=match.score, reasons=reasons)
    number_ok = value_grounded(match.span, cov.threshold, kinds_for(cov.covenant_type), cfg.rel_tol)
    if not number_ok:
        reasons.append(f"threshold {cov.threshold:g} does not occur in the matched source text")
    keyword_ok = bool(TYPE_KEYWORDS[cov.covenant_type].search(match.span))
    if cfg.require_keyword and not keyword_ok:
        reasons.append(f"matched source text does not mention the {cov.covenant_type.value} concept")
    page = locate_page(match.span, page_texts) or (match.source.page if match.source else None)
    if page is not None and page != cov.source_page:
        reasons.append(f"cited page {cov.source_page} corrected to {page}")
    ok = number_ok and (keyword_ok or not cfg.require_keyword)
    return GuardrailCheck(
        status=GuardrailStatus.VERIFIED if ok else GuardrailStatus.REJECTED,
        quote_match=match.kind,
        match_score=round(match.score, 4),
        number_grounded=number_ok,
        keyword_ok=keyword_ok,
        located_page=page,
        chunk_id=match.source.chunk_id if match.source else None,
        matched_text=match.span,
        reasons=reasons,
    )


def _field_check(
    quote: str | None,
    predicate,
    failure: str,
    sources: list[SourceText],
    page_texts: dict[int, str] | None,
    cfg: GuardrailConfig,
) -> GuardrailCheck:
    if not quote:
        return GuardrailCheck(status=GuardrailStatus.REJECTED, reasons=["no evidence quote supplied"])
    match = find_quote(quote, sources, cfg)
    if match.kind == "none":
        return GuardrailCheck(
            status=GuardrailStatus.REJECTED, match_score=match.score,
            reasons=[f"evidence quote not found in retrieved context (best fuzzy score {match.score:.2f})"],
        )
    ok = bool(predicate(match.span))
    return GuardrailCheck(
        status=GuardrailStatus.VERIFIED if ok else GuardrailStatus.REJECTED,
        quote_match=match.kind,
        match_score=round(match.score, 4),
        number_grounded=ok,
        keyword_ok=True,
        located_page=locate_page(match.span, page_texts) or (match.source.page if match.source else None),
        chunk_id=match.source.chunk_id if match.source else None,
        matched_text=match.span,
        reasons=[] if ok else [failure],
    )


def verify_loan_terms(
    lt: LoanTerms | None,
    sources: list[SourceText],
    page_texts: dict[int, str] | None = None,
    cfg: GuardrailConfig | None = None,
) -> tuple[LoanTerms | None, dict[str, GuardrailCheck]]:
    """Verify each loan-term field; rejected fields are set to ``None`` in the returned copy."""
    cfg = cfg or GuardrailConfig()
    if lt is None:
        return None, {}
    checks: dict[str, GuardrailCheck] = {}
    data = lt.model_dump()

    if lt.borrower is not None:
        name_key = compact(lt.borrower)
        checks["borrower"] = _field_check(
            lt.borrower_quote, lambda span: name_key and name_key in compact(span),
            "borrower name not present in matched source text", sources, page_texts, cfg,
        )
    if lt.principal is not None:
        def _principal_ok(span: str) -> bool:
            for m in parse_numbers(span):
                if m.kind == "currency" and abs(m.value - lt.principal) <= cfg.rel_tol * lt.principal:
                    return lt.currency is None or m.currency == lt.currency
            return False

        checks["principal"] = _field_check(
            lt.principal_quote, _principal_ok, "principal amount / currency not present in matched source text",
            sources, page_texts, cfg,
        )
    if lt.rate_pct is not None:
        checks["rate_pct"] = _field_check(
            lt.rate_quote, lambda span: value_grounded(span, lt.rate_pct, {"percent", "number"}, cfg.rel_tol),
            "interest rate not present in matched source text", sources, page_texts, cfg,
        )
    if lt.maturity_date is not None:
        checks["maturity_date"] = _field_check(
            lt.maturity_quote, lambda span: any(d.isoformat() == lt.maturity_date for d in parse_dates(span)),
            "maturity date not present in matched source text", sources, page_texts, cfg,
        )

    if "borrower" in checks and not checks["borrower"].accepted:
        data["borrower"] = None
        data["borrower_quote"] = None
    if "principal" in checks and not checks["principal"].accepted:
        data["principal"] = data["currency"] = data["principal_quote"] = None
    if "rate_pct" in checks and not checks["rate_pct"].accepted:
        data["rate_pct"] = data["rate_quote"] = data["rate_type"] = data["base_rate"] = None
    if "maturity_date" in checks and not checks["maturity_date"].accepted:
        data["maturity_date"] = data["maturity_quote"] = None
    return LoanTerms.model_validate(data), checks


def verify_covenants(
    covenants: list[Covenant],
    sources: list[SourceText],
    page_texts: dict[int, str] | None = None,
    cfg: GuardrailConfig | None = None,
) -> list[VerifiedCovenant]:
    return [VerifiedCovenant(covenant=c, check=verify_covenant(c, sources, page_texts, cfg)) for c in covenants]
