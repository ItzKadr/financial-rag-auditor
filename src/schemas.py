"""Pydantic v2 models.

Two families of models live here:

* **LLM-facing models** (``Covenant``, ``LoanTerms``, ``ExtractionPayload``): their JSON
  schema is sent to the LLM as a Structured Output contract. Constraints are enforced by
  validators (not by JSON-schema keywords) so that the generated schema stays compatible
  with OpenAI "strict" mode, while Pydantic still rejects bad values; the validation
  error text is fed back to the model on retry.
* **Pipeline models** (``GuardrailCheck``, ``VerifiedCovenant``, ``ExtractionResult``,
  ``AuditFinding``): results produced by the guardrails and the auditor.
"""

from __future__ import annotations

import math
import re
from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CovenantType(str, Enum):
    """Financial maintenance covenants supported by the auditor."""

    MAX_LEVERAGE = "max_leverage"  # Total Debt / LTM EBITDA  <= threshold
    MIN_DSCR = "min_dscr"  # LTM EBITDA / LTM Debt Service >= threshold
    MIN_INTEREST_COVERAGE = "min_interest_coverage"  # LTM EBITDA / LTM Net Finance Charges
    MIN_CURRENT_RATIO = "min_current_ratio"  # Current Assets / Current Liabilities
    MAX_CAPEX = "max_capex"  # Capital Expenditure per Financial Year (amount)
    MIN_NET_WORTH = "min_net_worth"  # Tangible Net Worth (amount)

    @property
    def direction(self) -> str:
        """'max' for ceilings, 'min' for floors."""
        return "max" if self.value.startswith("max") else "min"

    @property
    def expected_unit(self) -> "Unit":
        return Unit.CURRENCY if self in (CovenantType.MAX_CAPEX, CovenantType.MIN_NET_WORTH) else Unit.RATIO


COVENANT_LABELS: dict[CovenantType, str] = {
    CovenantType.MAX_LEVERAGE: "Leverage (Total Debt / EBITDA)",
    CovenantType.MIN_DSCR: "DSCR (EBITDA / Debt Service)",
    CovenantType.MIN_INTEREST_COVERAGE: "Interest Cover (EBITDA / Net Finance Charges)",
    CovenantType.MIN_CURRENT_RATIO: "Current Ratio (Current Assets / Current Liabilities)",
    CovenantType.MAX_CAPEX: "Capital Expenditure per Financial Year",
    CovenantType.MIN_NET_WORTH: "Tangible Net Worth",
}


class Operator(str, Enum):
    LE = "<="
    LT = "<"
    GE = ">="
    GT = ">"

    @property
    def is_upper_bound(self) -> bool:
        return self in (Operator.LE, Operator.LT)

    @property
    def is_strict(self) -> bool:
        return self in (Operator.LT, Operator.GT)


class Unit(str, Enum):
    RATIO = "ratio"
    CURRENCY = "currency"


class Frequency(str, Enum):
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"


class RateType(str, Enum):
    FIXED = "fixed"
    FLOATING = "floating"


def _check_quote(value: str | None) -> str | None:
    if value is None:
        return None
    v = re.sub(r"\s+", " ", value).strip()
    if len(v) < 8:
        raise ValueError("evidence quote must be a verbatim excerpt of at least 8 characters")
    return v


# --------------------------------------------------------------------------------------
# LLM-facing models (structured output contract)
# --------------------------------------------------------------------------------------
class Covenant(BaseModel):
    """One financial covenant exactly as stated in the agreement."""

    model_config = ConfigDict(extra="forbid")

    covenant_type: CovenantType = Field(description="Which financial covenant this is.")
    operator: Operator = Field(
        description="Comparison the borrower must satisfy: actual <operator> threshold "
        "(e.g. leverage 'shall not exceed 3.50x' -> '<=')."
    )
    threshold: float = Field(
        description="Numeric threshold. Ratios as plain numbers (3.5 for '3.50x' or '3.5 to 1.0'); "
        "amounts in full currency units (12500000 for 'USD 12.5 million')."
    )
    unit: Unit = Field(description="'ratio' for ratio covenants, 'currency' for amount covenants.")
    test_frequency: Frequency = Field(description="How often the covenant is tested.")
    source_page: int = Field(description="1-based page number where the clause appears.")
    evidence_quote: str = Field(
        description="Verbatim, contiguous excerpt (one sentence or one table row) copied from the "
        "context that states the threshold."
    )

    @field_validator("threshold")
    @classmethod
    def _positive_finite(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("threshold must be a positive finite number")
        return v

    @field_validator("source_page")
    @classmethod
    def _page_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("source_page is 1-based and must be >= 1")
        return v

    @field_validator("evidence_quote")
    @classmethod
    def _quote(cls, v: str) -> str:
        out = _check_quote(v)
        assert out is not None
        return out

    @model_validator(mode="after")
    def _consistency(self) -> "Covenant":
        ctype = self.covenant_type
        if ctype.direction == "max" and not self.operator.is_upper_bound:
            raise ValueError(f"{ctype.value} is a maximum covenant: operator must be '<=' or '<'")
        if ctype.direction == "min" and self.operator.is_upper_bound:
            raise ValueError(f"{ctype.value} is a minimum covenant: operator must be '>=' or '>'")
        if self.unit != ctype.expected_unit:
            raise ValueError(f"{ctype.value} must use unit '{ctype.expected_unit.value}'")
        if self.unit == Unit.RATIO and self.threshold > 50:
            raise ValueError("ratio thresholds above 50x are implausible; check the number")
        if self.unit == Unit.CURRENCY and self.threshold < 1000:
            raise ValueError("amount thresholds must be expressed in full currency units")
        return self


class LoanTerms(BaseModel):
    """Headline loan terms. Every populated value must be backed by its own quote."""

    model_config = ConfigDict(extra="forbid")

    borrower: str | None = Field(description="Legal name of the borrower.")
    borrower_quote: str | None = Field(description="Verbatim excerpt naming the borrower.")
    principal: float | None = Field(description="Facility amount in full currency units.")
    currency: str | None = Field(description="ISO 4217 code, e.g. USD, EUR, KZT.")
    principal_quote: str | None = Field(description="Verbatim excerpt stating the facility amount.")
    rate_type: RateType | None = Field(description="'fixed' or 'floating'.")
    base_rate: str | None = Field(description="Benchmark for floating loans (e.g. Term SOFR), else null.")
    rate_pct: float | None = Field(description="Margin (floating) or fixed rate, in percent per annum.")
    rate_quote: str | None = Field(description="Verbatim excerpt stating the margin / fixed rate.")
    maturity_date: str | None = Field(description="Final maturity date in ISO format YYYY-MM-DD.")
    maturity_quote: str | None = Field(description="Verbatim excerpt stating the final maturity date.")

    @field_validator("borrower_quote", "principal_quote", "rate_quote", "maturity_quote")
    @classmethod
    def _quotes(cls, v: str | None) -> str | None:
        return _check_quote(v)

    @field_validator("currency")
    @classmethod
    def _currency(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", v):
            raise ValueError("currency must be a 3-letter ISO 4217 code")
        return v

    @field_validator("principal")
    @classmethod
    def _principal(cls, v: float | None) -> float | None:
        if v is not None and (not math.isfinite(v) or v <= 0):
            raise ValueError("principal must be positive")
        return v

    @field_validator("rate_pct")
    @classmethod
    def _rate(cls, v: float | None) -> float | None:
        if v is not None and not (0 < v < 100):
            raise ValueError("rate_pct must be a percentage between 0 and 100")
        return v

    @field_validator("maturity_date")
    @classmethod
    def _iso_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            return date.fromisoformat(v.strip()).isoformat()
        except ValueError as exc:
            raise ValueError("maturity_date must be an ISO date YYYY-MM-DD") from exc

    @model_validator(mode="after")
    def _values_need_quotes(self) -> "LoanTerms":
        pairs = {
            "borrower": "borrower_quote",
            "principal": "principal_quote",
            "rate_pct": "rate_quote",
            "maturity_date": "maturity_quote",
        }
        for field, quote in pairs.items():
            if getattr(self, field) is not None and getattr(self, quote) is None:
                raise ValueError(f"'{field}' is populated but '{quote}' is missing")
        return self


class ExtractionPayload(BaseModel):
    """Top-level Structured Output returned by the extractor (LLM or offline)."""

    model_config = ConfigDict(extra="forbid")

    loan_terms: LoanTerms | None = Field(description="Headline loan terms, or null if absent.")
    covenants: list[Covenant] = Field(description="Every financial covenant found in the context.")


# --------------------------------------------------------------------------------------
# Pipeline models
# --------------------------------------------------------------------------------------
class GuardrailStatus(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"


class GuardrailCheck(BaseModel):
    """Outcome of grounding one extracted value in the retrieved source text."""

    status: GuardrailStatus
    quote_match: str = "none"  # "exact" | "fuzzy" | "none"
    match_score: float = 0.0
    number_grounded: bool = False
    keyword_ok: bool = False
    located_page: int | None = None
    chunk_id: str | None = None
    matched_text: str | None = None
    reasons: list[str] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status == GuardrailStatus.VERIFIED


class VerifiedCovenant(BaseModel):
    covenant: Covenant
    check: GuardrailCheck

    @property
    def accepted(self) -> bool:
        return self.check.accepted

    @property
    def citation_page(self) -> int:
        return self.check.located_page or self.covenant.source_page


class ExtractionResult(BaseModel):
    """Everything the extractor produced for one contract (after guardrails)."""

    contract_id: str
    extractor: str
    loan_terms: LoanTerms | None = None
    loan_term_checks: dict[str, GuardrailCheck] = Field(default_factory=dict)
    covenants: list[VerifiedCovenant] = Field(default_factory=list)
    retrieved: dict[str, list[str]] = Field(default_factory=dict)
    attempts: int = 1
    errors: list[str] = Field(default_factory=list)
    latency_s: float = 0.0

    def accepted_covenants(self) -> list[VerifiedCovenant]:
        return [c for c in self.covenants if c.accepted]

    def rejected_covenants(self) -> list[VerifiedCovenant]:
        return [c for c in self.covenants if not c.accepted]


class FindingStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    BREACH = "BREACH"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class AuditFinding(BaseModel):
    """Result of testing one covenant for one borrower in one test period."""

    contract_id: str
    borrower: str
    period: str
    covenant_type: CovenantType
    operator: Operator
    threshold: float
    unit: Unit
    actual_value: float | None
    headroom_pct: float | None
    status: FindingStatus
    source_page: int
    evidence_quote: str
    note: str = ""

    def key(self) -> tuple[str, str, str]:
        return (self.contract_id, self.covenant_type.value, self.period)

    def to_row(self) -> dict[str, Any]:
        row = self.model_dump(mode="json")
        row["citation"] = f"p.{self.source_page}: \"{self.evidence_quote}\""
        return row
