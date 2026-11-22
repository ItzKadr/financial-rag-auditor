"""Covenant & loan-term extraction: offline rules or an LLM with Structured Outputs.

Both extractors share the same RAG front-end and back-end:

    per-covenant hybrid retrieval -> context chunks -> propose ExtractionPayload
                                                      -> guardrails -> ExtractionResult

* :class:`OfflineExtractor` - deterministic rules over the retrieved chunks (table rows
  and sentences): covenant keyword + obligation modal ("shall") + operator cue + a number
  of the right kind (ratio vs currency). Runs without any API key.
* :class:`LLMExtractor` - any OpenAI-compatible endpoint. The Pydantic schema of
  :class:`~src.schemas.ExtractionPayload` is the output contract (``.parse()`` strict
  Structured Outputs, ``json_schema`` or ``json_object`` mode), temperature 0. If the
  output fails Pydantic validation, the validation errors are sent back to the model and
  the call is retried (up to ``max_attempts``).
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from .chunker import Chunk
from .guardrails import (
    TYPE_KEYWORDS,
    GuardrailConfig,
    SourceText,
    locate_page,
    sources_from_chunks,
    verify_covenants,
    verify_loan_terms,
)
from .providers import ProviderConfig, make_client, resolve_provider
from .retriever import LOAN_TERM_QUERIES, HybridIndex, RetrievalHit, covenant_query
from .schemas import (
    Covenant,
    CovenantType,
    ExtractionPayload,
    ExtractionResult,
    Frequency,
    LoanTerms,
    Operator,
    RateType,
    Unit,
)
from .text_utils import normalize_text, parse_dates, parse_numbers, split_sentences

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Rule helpers (offline extractor)
# --------------------------------------------------------------------------------------
_OPERATOR_PATTERNS: list[tuple[re.Pattern[str], Operator]] = [
    (re.compile(r"\bnot\b[^.;]{0,60}?\b(?:be )?(?:less than|lower than|below)\b"), Operator.GE),
    (re.compile(r"\bnot\b[^.;]{0,60}?\bfall below\b"), Operator.GE),
    (re.compile(r"\bnot\b[^.;]{0,60}?\b(?:exceed(?:ing)?|be greater than|be more than|be higher than|greater than|more than|in excess of)\b"), Operator.LE),
    (re.compile(r"\bno (?:more|higher|greater) than\b"), Operator.LE),
    (re.compile(r"\bno (?:less|lower) than\b"), Operator.GE),
    (re.compile(r"\bequal (?:to )?or (?:exceed|greater than)\b|\bequal or exceed\b"), Operator.GE),
    (re.compile(r"\bat least\b"), Operator.GE),
    (re.compile(r"\bat most\b"), Operator.LE),
    (re.compile(r"\bmaximum\b"), Operator.LE),
    (re.compile(r"\bminimum\b"), Operator.GE),
    (re.compile(r"\bshall (?:be|remain) (?:less than|below)\b"), Operator.LT),
    (re.compile(r"\bshall (?:be greater than|exceed)\b"), Operator.GT),
]
_MODAL_RE = re.compile(r"\b(shall|must)\b")


def detect_operator(text: str) -> tuple[Operator | None, int]:
    """Return (operator, cue end position) from obligation wording in normalised text."""
    norm = normalize_text(text)
    for pattern, op in _OPERATOR_PATTERNS:
        m = pattern.search(norm)
        if m:
            return op, m.end()
    return None, -1


def detect_frequency(text: str) -> Frequency | None:
    """Testing frequency implied by wording ("in any Financial Year" -> annual, ...)."""
    norm = normalize_text(text)
    if re.search(r"semi-annual|half-year|half year|semi annual", norm):
        return Frequency.SEMI_ANNUAL
    if "financial year" in norm or re.search(r"\bannual(ly)?\b", norm):
        return Frequency.ANNUAL
    if "quarter" in norm:
        return Frequency.QUARTERLY
    return None


def candidate_units(text: str) -> list[tuple[str, bool]]:
    """Split chunk text into (unit, is_table_row) pairs: table rows and sentences."""
    units: list[tuple[str, bool]] = []
    prose: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if prose:
                units.extend((s, False) for s in split_sentences(" ".join(prose)))
                prose = []
            if not re.fullmatch(r"\|(\s*-{3,}\s*\|)+", stripped):
                units.append((stripped, True))
        elif stripped:
            prose.append(stripped)
    if prose:
        units.extend((s, False) for s in split_sentences(" ".join(prose)))
    return units


def match_covenant_unit(
    unit: str, is_row: bool, ctype: CovenantType, default_freq: Frequency
) -> tuple[Operator, float, Frequency] | None:
    """Apply extraction rules to one sentence / table row for one covenant type."""
    norm = normalize_text(unit)
    if not TYPE_KEYWORDS[ctype].search(norm):
        return None
    if is_row:
        cells = [c.strip() for c in unit.strip("|").split("|")]
        if not TYPE_KEYWORDS[ctype].search(normalize_text(cells[0])):
            return None
        op, _ = detect_operator(" ".join(cells[1:]))
        values_text = " ".join(cells[1:])
        cue_end = -1
    else:
        if not _MODAL_RE.search(norm):
            return None
        op, cue_end = detect_operator(unit)
        values_text = unit
    if op is None:
        return None
    if (ctype.direction == "max") != op.is_upper_bound:
        return None
    wanted = {"ratio"} if ctype.expected_unit == Unit.RATIO else {"currency"}
    if is_row and ctype.expected_unit == Unit.RATIO:
        wanted.add("number")
    mentions = [m for m in parse_numbers(values_text) if m.kind in wanted]
    if not mentions:
        return None
    if cue_end >= 0:
        # prefer the first number that follows the obligation cue ("shall not exceed 3.50x")
        mentions = [m for m in mentions if m.start >= cue_end - 2] or mentions
    freq = Frequency.ANNUAL if ctype == CovenantType.MAX_CAPEX else (detect_frequency(unit) or default_freq)
    return op, mentions[0].value, freq


# --------------------------------------------------------------------------------------
# Structured-output LLM client
# --------------------------------------------------------------------------------------
class StructuredOutputError(RuntimeError):
    """The model refused or returned unusable output."""


@dataclass
class StructuredCallResult:
    parsed: BaseModel | None
    attempts: int
    errors: list[str] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema adapted to OpenAI strict mode.

    Every object gets ``additionalProperties: false`` and all properties listed as
    required (nullable fields stay nullable via ``anyOf``); ``$ref`` nodes that carry
    sibling keywords are inlined; ``default`` keywords are dropped.
    """
    schema = copy.deepcopy(model.model_json_schema())
    defs = schema.get("$defs", {})

    def visit(node: Any) -> Any:
        if isinstance(node, list):
            return [visit(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node and len(node) > 1:
            ref = node.pop("$ref").split("/")[-1]
            merged = copy.deepcopy(defs[ref])
            merged.update(node)
            node = merged
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "default":
                continue
            if k in ("properties", "$defs"):
                out[k] = {name: visit(sub) for name, sub in v.items()}
            else:
                out[k] = visit(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out

    return visit(schema)


def _format_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for e in exc.errors()[:12]:
            loc = ".".join(str(x) for x in e.get("loc", ()))
            parts.append(f"- {loc or '<root>'}: {e.get('msg')}")
        return "\n".join(parts)
    return f"- {exc}"


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    return m.group(1).strip() if m else text.strip()


class StructuredLLMClient:
    """Calls an OpenAI-compatible client and returns a validated Pydantic object.

    Retry-with-feedback loop: on a Pydantic ``ValidationError`` (bad JSON, wrong enum,
    failed model validator, ...) the error list is appended to the conversation and the
    model is asked to correct its answer.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        response_mode: str = "parse",
        max_attempts: int = 3,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.response_mode = response_mode
        self.max_attempts = max_attempts
        self.temperature = temperature
        self._last_raw: str | None = None

    def _parse_endpoint(self) -> Any:
        completions = self.client.chat.completions
        if hasattr(completions, "parse"):
            return completions.parse
        return self.client.beta.chat.completions.parse  # older SDKs

    def _call(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> tuple[BaseModel, str]:
        """One API call. Sets ``self._last_raw`` before validating so a failed answer can be echoed back."""
        self._last_raw = None
        if self.response_mode == "parse":
            resp = self._parse_endpoint()(
                model=self.model, messages=messages, response_format=schema, temperature=self.temperature
            )
            msg = resp.choices[0].message
            if getattr(msg, "refusal", None):
                raise StructuredOutputError(f"model refused: {msg.refusal}")
            raw = msg.content or ""
            self._last_raw = raw
            parsed = getattr(msg, "parsed", None)
            return (parsed if parsed is not None else schema.model_validate_json(_strip_fences(raw))), raw
        if self.response_mode == "json_schema":
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": strict_json_schema(schema), "strict": True},
            }
        else:
            response_format = {"type": "json_object"}
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, response_format=response_format, temperature=self.temperature
        )
        raw = resp.choices[0].message.content or ""
        self._last_raw = raw
        return schema.model_validate_json(_strip_fences(raw)), raw

    def complete(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> StructuredCallResult:
        msgs = list(messages)
        errors: list[str] = []
        raws: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                parsed, raw = self._call(msgs, schema)
                raws.append(raw)
                return StructuredCallResult(parsed, attempt, errors, raws)
            except (ValidationError, json.JSONDecodeError, StructuredOutputError) as exc:
                # in .parse() mode the SDK validates internally, so the raw text may be unavailable
                raw = self._last_raw
                err = _format_error(exc)
                errors.append(err)
                LOG.warning("structured output attempt %d failed:\n%s", attempt, err)
                if raw:
                    raws.append(raw)
                    msgs.append({"role": "assistant", "content": raw})
                msgs.append(
                    {
                        "role": "user",
                        "content": "Your previous answer failed schema validation:\n"
                        f"{err}\nFix these problems and return the complete corrected JSON object only.",
                    }
                )
        return StructuredCallResult(None, self.max_attempts, errors, raws)


SYSTEM_PROMPT = """You are a meticulous credit analyst auditing a loan agreement.
Extract ONLY information that is explicitly written in the CONTEXT excerpts.

Rules:
1. Covenant types: max_leverage (Total Debt / EBITDA), min_dscr (EBITDA / Debt Service),
   min_interest_coverage (EBITDA / Net Finance Charges), min_current_ratio
   (Current Assets / Current Liabilities), max_capex (Capital Expenditure per Financial Year),
   min_net_worth (Tangible Net Worth).
2. evidence_quote MUST be copied verbatim from the CONTEXT: the single sentence or the single
   table row that states the threshold. Never paraphrase, never merge sentences.
3. Ignore numbers that are not covenant obligations: ratios stated as current facts,
   margin ratchet grids, baskets, cure amounts, cross-default thresholds, repayment amounts.
4. threshold: ratios as plain numbers (3.5 for "3.50x", "3.5 to 1.0" or "3.50:1.00");
   amounts in full currency units (12500000 for "USD 12.5 million").
5. operator: "<=" for "shall not exceed / no more than / maximum"; "<" for "shall be less than";
   ">=" for "not less than / at least / minimum / shall not fall below"; ">" for
   "shall be greater than / shall exceed".
6. test_frequency: "annual" for limits per Financial Year; otherwise follow the financial
   testing clause ("quarterly" or "semi_annual").
7. source_page: the page shown in the chunk header the quote comes from.
8. Loan terms: rate_pct is the Margin for floating loans or the fixed rate; dates in ISO format.
9. If something is not stated, omit the covenant or use null. Never guess or compute.
Answer with a JSON object only."""


# --------------------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------------------
@dataclass
class ExtractorConfig:
    top_k: int = 5
    max_context_chunks: int = 16
    max_attempts: int = 3


class BaseExtractor(ABC):
    """Shared retrieval + guardrail pipeline; subclasses only propose a payload."""

    name: str = "base"

    def __init__(self, cfg: ExtractorConfig | None = None, guard_cfg: GuardrailConfig | None = None) -> None:
        self.cfg = cfg or ExtractorConfig()
        self.guard_cfg = guard_cfg or GuardrailConfig()

    def retrieve(self, index: HybridIndex) -> tuple[dict[str, list[RetrievalHit]], list[Chunk]]:
        retrieved: dict[str, list[RetrievalHit]] = {}
        for ctype in CovenantType:
            query = covenant_query(ctype, index.cfg.expand_queries)
            retrieved[ctype.value] = index.search(query, top_k=self.cfg.top_k)
        for key, query in LOAN_TERM_QUERIES.items():
            retrieved[key] = index.search(query, top_k=3)
        best: dict[str, tuple[int, int, Chunk]] = {}
        for q_idx, hits in enumerate(retrieved.values()):
            for h in hits:
                prev = best.get(h.chunk.chunk_id)
                if prev is None or (h.rank, q_idx) < prev[:2]:
                    best[h.chunk.chunk_id] = (h.rank, q_idx, h.chunk)
        context = [c for _, _, c in sorted(best.values(), key=lambda x: (x[0], x[1]))][: self.cfg.max_context_chunks]
        context.sort(key=lambda c: c.order)
        return retrieved, context

    @abstractmethod
    def propose(
        self, retrieved: dict[str, list[RetrievalHit]], context: list[Chunk], page_texts: dict[int, str]
    ) -> tuple[ExtractionPayload, int, list[str]]:
        """Return (payload, attempts, errors) from the retrieved context."""

    def extract(self, contract_id: str, index: HybridIndex, page_texts: dict[int, str]) -> ExtractionResult:
        t0 = time.perf_counter()
        retrieved, context = self.retrieve(index)
        errors: list[str] = []
        try:
            payload, attempts, errors = self.propose(retrieved, context, page_texts)
        except Exception as exc:  # provider/network failure must not kill the batch
            LOG.error("extraction failed for %s: %s", contract_id, exc)
            payload, attempts, errors = ExtractionPayload(loan_terms=None, covenants=[]), 1, [f"{type(exc).__name__}: {exc}"]
        sources = sources_from_chunks(context)
        verified = verify_covenants(payload.covenants, sources, page_texts, self.guard_cfg)
        loan_terms, lt_checks = verify_loan_terms(payload.loan_terms, sources, page_texts, self.guard_cfg)
        return ExtractionResult(
            contract_id=contract_id,
            extractor=self.name,
            loan_terms=loan_terms,
            loan_term_checks=lt_checks,
            covenants=verified,
            retrieved={k: [h.chunk.chunk_id for h in v] for k, v in retrieved.items()},
            attempts=attempts,
            errors=errors,
            latency_s=round(time.perf_counter() - t0, 4),
        )


class OfflineExtractor(BaseExtractor):
    """Deterministic rule-based extractor over retrieved chunks (no API key needed)."""

    name = "offline-rules"

    @staticmethod
    def _contract_frequency(hits: list[RetrievalHit]) -> Frequency:
        for h in hits:
            for sent in split_sentences(h.chunk.text.replace("\n", " ")):
                norm = normalize_text(sent)
                if "tested" in norm or "test date" in norm:
                    f = detect_frequency(sent)
                    if f is not None and f != Frequency.ANNUAL:
                        return f
        return Frequency.QUARTERLY

    def _covenants(
        self, retrieved: dict[str, list[RetrievalHit]], page_texts: dict[int, str], default_freq: Frequency
    ) -> list[Covenant]:
        out: list[Covenant] = []
        for ctype in CovenantType:
            found: Covenant | None = None
            for hit in retrieved[ctype.value]:
                for unit, is_row in candidate_units(hit.chunk.text):
                    res = match_covenant_unit(unit, is_row, ctype, default_freq)
                    if res is None:
                        continue
                    op, value, freq = res
                    page = locate_page(unit, page_texts) or hit.chunk.page_start
                    try:
                        found = Covenant(
                            covenant_type=ctype, operator=op, threshold=value, unit=ctype.expected_unit,
                            test_frequency=freq, source_page=page, evidence_quote=unit,
                        )
                    except ValidationError as exc:
                        LOG.debug("rule candidate rejected by schema: %s", exc)
                        continue
                    break
                if found is not None:
                    break
            if found is not None:
                out.append(found)
        return out

    @staticmethod
    def _loan_terms(retrieved: dict[str, list[RetrievalHit]], context: list[Chunk]) -> LoanTerms:
        sentences = []
        seen: set[str] = set()
        for key in ("borrower", "principal", "rate", "maturity"):
            for h in retrieved.get(key, []):
                if h.chunk.chunk_id not in seen:
                    seen.add(h.chunk.chunk_id)
                    sentences.extend(split_sentences(h.chunk.text.replace("\n", " ")))
        for c in context:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                sentences.extend(split_sentences(c.text.replace("\n", " ")))

        data: dict[str, Any] = {k: None for k in LoanTerms.model_fields}
        for s in sentences:
            if data["borrower"] is None and '"Borrower"' in s:
                m = re.search(r"\(1\)\s+(?P<name>[^,()]+?),\s+(?:a|an)\s", s)
                if m:
                    data["borrower"], data["borrower_quote"] = m.group("name").strip(), s
            norm = normalize_text(s)
            if data["principal"] is None and "aggregate amount" in norm and "facility" in norm:
                cur = [m for m in parse_numbers(s) if m.kind == "currency"]
                if cur:
                    data["principal"], data["currency"], data["principal_quote"] = cur[0].value, cur[0].currency, s
            if data["rate_pct"] is None and re.search(r'"margin" means', norm):
                pct = [m for m in parse_numbers(s) if m.kind == "percent"]
                if pct:
                    data["rate_pct"], data["rate_quote"], data["rate_type"] = pct[0].value, s, RateType.FLOATING
            if data["rate_pct"] is None and "fixed rate of" in norm:
                pct = [m for m in parse_numbers(s) if m.kind == "percent"]
                if pct:
                    data["rate_pct"], data["rate_quote"], data["rate_type"] = pct[0].value, s, RateType.FIXED
            if data["base_rate"] is None:
                m = re.search(r"\b(Term SOFR|SOFR|EURIBOR|TONIA|SONIA|LIBOR)\b", s)
                if m and "margin" in norm:
                    data["base_rate"] = m.group(1)
            if data["maturity_date"] is None and '"final maturity date" means' in norm:
                dates = parse_dates(s)
                if dates:
                    data["maturity_date"], data["maturity_quote"] = dates[-1].isoformat(), s
        if data["rate_type"] == RateType.FIXED:
            data["base_rate"] = None
        return LoanTerms.model_validate(data)

    def propose(
        self, retrieved: dict[str, list[RetrievalHit]], context: list[Chunk], page_texts: dict[int, str]
    ) -> tuple[ExtractionPayload, int, list[str]]:
        default_freq = self._contract_frequency(retrieved.get("frequency", []))
        covenants = self._covenants(retrieved, page_texts, default_freq)
        loan_terms = self._loan_terms(retrieved, context)
        return ExtractionPayload(loan_terms=loan_terms, covenants=covenants), 1, []


class LLMExtractor(BaseExtractor):
    """LLM extractor using Structured Outputs through an OpenAI-compatible client."""

    def __init__(
        self,
        provider: ProviderConfig,
        cfg: ExtractorConfig | None = None,
        guard_cfg: GuardrailConfig | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(cfg, guard_cfg)
        self.provider = provider
        self.name = f"{provider.name}:{provider.model}"
        self.llm = StructuredLLMClient(
            client if client is not None else make_client(provider),
            model=provider.model,
            response_mode=provider.response_mode,
            max_attempts=self.cfg.max_attempts,
        )

    def build_messages(self, context: list[Chunk]) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        if self.provider.response_mode == "json_object":
            system += "\n\nThe JSON object must follow this JSON schema:\n" + json.dumps(
                strict_json_schema(ExtractionPayload), separators=(",", ":")
            )
        blocks = [f"[chunk {c.chunk_id} | page {c.page_label} | {c.section}]\n{c.text}" for c in context]
        user = "CONTEXT:\n\n" + "\n\n".join(blocks) + "\n\nExtract the loan terms and every financial covenant as JSON."
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def propose(
        self, retrieved: dict[str, list[RetrievalHit]], context: list[Chunk], page_texts: dict[int, str]
    ) -> tuple[ExtractionPayload, int, list[str]]:
        result = self.llm.complete(self.build_messages(context), ExtractionPayload)
        if result.parsed is None:
            return ExtractionPayload(loan_terms=None, covenants=[]), result.attempts, result.errors
        assert isinstance(result.parsed, ExtractionPayload)
        return result.parsed, result.attempts, result.errors


def build_extractor(
    name: str = "offline",
    cfg: ExtractorConfig | None = None,
    guard_cfg: GuardrailConfig | None = None,
    client: Any | None = None,
) -> BaseExtractor:
    """'offline' or a provider name ('openai' | 'deepseek' | 'groq')."""
    if name in ("offline", "none", "rules"):
        return OfflineExtractor(cfg, guard_cfg)
    return LLMExtractor(resolve_provider(name), cfg, guard_cfg, client=client)


__all__ = [
    "BaseExtractor",
    "ExtractorConfig",
    "LLMExtractor",
    "OfflineExtractor",
    "SourceText",
    "StructuredCallResult",
    "StructuredLLMClient",
    "build_extractor",
    "candidate_units",
    "detect_operator",
    "match_covenant_unit",
    "strict_json_schema",
]
