"""Reconciliation of extracted covenants against the Master Ledger.

For every borrower and quarter the auditor computes the *actual* covenant metrics:

* flows (EBITDA, Net Finance Charges, Debt Service) are taken on a last-twelve-months
  (LTM) basis: :math:`\\mathrm{LTM}_t(x) = \\sum_{i=t-3}^{t} x_i` (all four quarters must
  be present, otherwise the metric is undefined);
* stocks (Total Debt, Current Assets/Liabilities, Net Worth) are point-in-time;
* Capital Expenditure is summed over the Financial Year and tested at year end.

.. math::

    \\text{Leverage}_t = \\frac{\\text{Total Debt}_t}{\\mathrm{LTM}_t(\\text{EBITDA})},\\quad
    \\text{DSCR}_t = \\frac{\\mathrm{LTM}_t(\\text{EBITDA})}{\\mathrm{LTM}_t(\\text{Debt Service})},\\quad
    \\text{ICR}_t = \\frac{\\mathrm{LTM}_t(\\text{EBITDA})}{\\mathrm{LTM}_t(\\text{Interest})}

Each (covenant, test period) yields one :class:`AuditFinding` with status COMPLIANT,
BREACH or INSUFFICIENT_DATA and a citation (page + verbatim quote) back to the contract.
"""

from __future__ import annotations

import difflib
import logging
import math
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .schemas import (
    AuditFinding,
    CovenantType,
    ExtractionResult,
    FindingStatus,
    Frequency,
    Operator,
)

LOG = logging.getLogger(__name__)

# Ledger columns each covenant depends on and how ("point" in time, "ltm" window, "fy").
METRIC_DEPENDENCIES: dict[CovenantType, dict[str, str]] = {
    CovenantType.MAX_LEVERAGE: {"total_debt": "point", "ebitda": "ltm"},
    CovenantType.MIN_DSCR: {"ebitda": "ltm", "debt_service": "ltm"},
    CovenantType.MIN_INTEREST_COVERAGE: {"ebitda": "ltm", "interest_expense": "ltm"},
    CovenantType.MIN_CURRENT_RATIO: {"current_assets": "point", "current_liabilities": "point"},
    CovenantType.MAX_CAPEX: {"capex": "fy"},
    CovenantType.MIN_NET_WORTH: {"net_worth": "point"},
}
METRIC_COLUMN: dict[CovenantType, str] = {
    CovenantType.MAX_LEVERAGE: "leverage",
    CovenantType.MIN_DSCR: "dscr",
    CovenantType.MIN_INTEREST_COVERAGE: "interest_cover",
    CovenantType.MIN_CURRENT_RATIO: "current_ratio",
    CovenantType.MAX_CAPEX: "capex_fy",
    CovenantType.MIN_NET_WORTH: "net_worth_actual",
}
_LEGAL_FORMS = {"llp", "jsc", "ltd", "plc", "bv", "llc", "inc", "limited", "corp", "sa", "ag", "gmbh"}


# --------------------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------------------
def period_label(start_year: int, t: int) -> str:
    """Quarter index ``t`` (0-based from ``start_year`` Q1) -> 'YYYYQn'."""
    return f"{start_year + t // 4}Q{t % 4 + 1}"


def period_end(start_year: int, t: int) -> date:
    year, q = start_year + t // 4, t % 4 + 1
    return {1: date(year, 3, 31), 2: date(year, 6, 30), 3: date(year, 9, 30), 4: date(year, 12, 31)}[q]


def parse_period(label: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})Q([1-4])", label.strip())
    if not m:
        raise ValueError(f"bad period label {label!r}; expected 'YYYYQn'")
    return int(m.group(1)), int(m.group(2))


def is_test_quarter(quarter: int, frequency: Frequency) -> bool:
    """Quarterly: every quarter; semi-annual: Q2 and Q4; annual: Q4 (financial year end)."""
    if frequency == Frequency.QUARTERLY:
        return True
    if frequency == Frequency.SEMI_ANNUAL:
        return quarter in (2, 4)
    return quarter == 4


# --------------------------------------------------------------------------------------
# Actual metrics
# --------------------------------------------------------------------------------------
def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` with explicit semantics for non-positive denominators.

    NaN in either input -> NaN (insufficient data). ``den <= 0`` -> ``+inf`` if
    ``num > 0`` else ``-inf`` (e.g. leverage with non-positive EBITDA is "infinitely"
    levered and therefore breaches any maximum).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan)
    ok = ~np.isnan(num) & ~np.isnan(den)
    pos = ok & (den > 0)
    out[pos] = num[pos] / den[pos]
    nonpos = ok & (den <= 0)
    out[nonpos] = np.where(num[nonpos] > 0, np.inf, -np.inf)
    return out


def compute_actuals(ledger: pd.DataFrame) -> pd.DataFrame:
    """Add LTM aggregates and covenant metrics to a quarterly ledger.

    Required columns: borrower_id, borrower_name, fiscal_year, quarter and the eight
    financial columns. Gaps in a borrower's quarter sequence are re-indexed as NaN rows
    so that LTM windows never silently bridge a missing quarter.
    """
    frames: list[pd.DataFrame] = []
    for bid, g in ledger.groupby("borrower_id", sort=True):
        g = g.copy()
        g["_t"] = g["fiscal_year"].astype(int) * 4 + g["quarter"].astype(int) - 1
        full = pd.DataFrame({"_t": np.arange(g["_t"].min(), g["_t"].max() + 1)})
        g = full.merge(g, on="_t", how="left")
        g["borrower_id"] = bid
        g["borrower_name"] = g["borrower_name"].ffill().bfill()
        g["fiscal_year"] = g["_t"] // 4
        g["quarter"] = g["_t"] % 4 + 1
        g["period"] = [f"{y}Q{q}" for y, q in zip(g["fiscal_year"], g["quarter"])]
        for col in ("ebitda", "interest_expense", "debt_service"):
            g[f"{col}_ltm"] = g[col].rolling(4, min_periods=4).sum()
        g["leverage"] = safe_ratio(g["total_debt"].to_numpy(), g["ebitda_ltm"].to_numpy())
        g["interest_cover"] = safe_ratio(g["ebitda_ltm"].to_numpy(), g["interest_expense_ltm"].to_numpy())
        g["dscr"] = safe_ratio(g["ebitda_ltm"].to_numpy(), g["debt_service_ltm"].to_numpy())
        g["current_ratio"] = safe_ratio(g["current_assets"].to_numpy(), g["current_liabilities"].to_numpy())
        fy_sum = g.groupby("fiscal_year")["capex"].transform(lambda s: s.sum(min_count=4) if s.notna().sum() == 4 else np.nan)
        g["capex_fy"] = np.where(g["quarter"] == 4, fy_sum, np.nan)
        g["net_worth_actual"] = g["net_worth"]
        g["seq"] = g["_t"] - g["_t"].min()
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Comparison semantics
# --------------------------------------------------------------------------------------
def is_compliant(actual: float, operator: Operator, threshold: float, rel_tol: float = 1e-9) -> bool:
    """Evaluate ``actual <operator> threshold``.

    Values within ``rel_tol`` of the threshold are treated as *equal*: equality complies
    with '<=' / '>=' and breaches '<' / '>'. Infinite actuals follow IEEE ordering.
    """
    if math.isfinite(actual) and math.isclose(actual, threshold, rel_tol=rel_tol, abs_tol=1e-12):
        return operator in (Operator.LE, Operator.GE)
    if operator == Operator.LE:
        return actual <= threshold
    if operator == Operator.LT:
        return actual < threshold
    if operator == Operator.GE:
        return actual >= threshold
    return actual > threshold


def headroom_pct(actual: float, operator: Operator, threshold: float) -> float | None:
    """Signed distance to the threshold in % of the threshold (positive = inside the limit)."""
    if not math.isfinite(actual):
        return None
    if operator.is_upper_bound:
        return 100.0 * (threshold - actual) / threshold
    return 100.0 * (actual - threshold) / threshold


# --------------------------------------------------------------------------------------
# Borrower matching
# --------------------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    tokens = re.sub(r"[^a-z0-9 ]+", " ", name.lower().replace("b.v.", "bv")).split()
    return " ".join(t for t in tokens if t not in _LEGAL_FORMS)


def match_borrower(name: str | None, ledger: pd.DataFrame, cutoff: float = 0.85) -> tuple[str | None, float]:
    """Map an extracted borrower name to a ledger ``borrower_id`` (exact, then fuzzy)."""
    if not name:
        return None, 0.0
    table = ledger[["borrower_id", "borrower_name"]].drop_duplicates()
    norm = {normalize_name(n): b for b, n in zip(table["borrower_id"], table["borrower_name"])}
    key = normalize_name(name)
    if key in norm:
        return norm[key], 1.0
    best = difflib.get_close_matches(key, list(norm), n=1, cutoff=cutoff)
    if best:
        return norm[best[0]], difflib.SequenceMatcher(None, key, best[0]).ratio()
    return None, 0.0


# --------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------
@dataclass
class AuditConfig:
    rel_tol: float = 1e-9
    skip_warmup_quarters: int = 3  # first LTM window needs 4 quarters of history
    name_match_cutoff: float = 0.85


def _missing_inputs(rows: pd.DataFrame, idx: int, ctype: CovenantType) -> list[str]:
    row = rows.loc[idx]
    missing: list[str] = []
    for col, kind in METRIC_DEPENDENCIES[ctype].items():
        if kind == "point":
            window = rows.loc[[idx]]
        elif kind == "ltm":
            window = rows[(rows["_t"] <= row["_t"]) & (rows["_t"] >= row["_t"] - 3)]
        else:
            window = rows[rows["fiscal_year"] == row["fiscal_year"]]
        need = 1 if kind == "point" else 4
        if window[col].isna().any() or len(window) < need:
            bad = window.loc[window[col].isna(), "period"].tolist()
            missing.append(f"{col}@{','.join(bad) if bad else 'history'}")
    return missing


def audit_contract(
    extraction: ExtractionResult,
    actuals: pd.DataFrame,
    ledger: pd.DataFrame,
    cfg: AuditConfig | None = None,
) -> list[AuditFinding]:
    """Test every *accepted* covenant of one contract for every applicable period."""
    cfg = cfg or AuditConfig()
    borrower = extraction.loan_terms.borrower if extraction.loan_terms else None
    bid, _score = match_borrower(borrower, ledger, cfg.name_match_cutoff)
    findings: list[AuditFinding] = []
    accepted = extraction.accepted_covenants()
    if bid is None:
        for vc in accepted:
            c = vc.covenant
            findings.append(
                AuditFinding(
                    contract_id=extraction.contract_id,
                    borrower=borrower or "UNKNOWN",
                    period="N/A",
                    covenant_type=c.covenant_type,
                    operator=c.operator,
                    threshold=c.threshold,
                    unit=c.unit,
                    actual_value=None,
                    headroom_pct=None,
                    status=FindingStatus.INSUFFICIENT_DATA,
                    source_page=vc.citation_page,
                    evidence_quote=c.evidence_quote,
                    note="borrower not identified in the Master Ledger",
                )
            )
        return findings

    rows = actuals[actuals["borrower_id"] == bid].reset_index(drop=True)
    name = str(rows["borrower_name"].iloc[0])
    window = rows[rows["seq"] >= cfg.skip_warmup_quarters]
    for vc in accepted:
        c = vc.covenant
        freq = Frequency.ANNUAL if c.covenant_type == CovenantType.MAX_CAPEX else c.test_frequency
        metric = METRIC_COLUMN[c.covenant_type]
        for idx, row in window.iterrows():
            if not is_test_quarter(int(row["quarter"]), freq):
                continue
            actual = float(row[metric])
            if math.isnan(actual):
                status, head, note = FindingStatus.INSUFFICIENT_DATA, None, "missing ledger inputs: " + "; ".join(
                    _missing_inputs(rows, idx, c.covenant_type)
                )
                actual_out: float | None = None
            else:
                ok = is_compliant(actual, c.operator, c.threshold, cfg.rel_tol)
                status = FindingStatus.COMPLIANT if ok else FindingStatus.BREACH
                head = headroom_pct(actual, c.operator, c.threshold)
                note = "" if math.isfinite(actual) else "non-positive denominator"
                actual_out = actual if math.isfinite(actual) else None
            findings.append(
                AuditFinding(
                    contract_id=extraction.contract_id,
                    borrower=name,
                    period=str(row["period"]),
                    covenant_type=c.covenant_type,
                    operator=c.operator,
                    threshold=c.threshold,
                    unit=c.unit,
                    actual_value=actual_out,
                    headroom_pct=head,
                    status=status,
                    source_page=vc.citation_page,
                    evidence_quote=c.evidence_quote,
                    note=note,
                )
            )
    return findings


def audit_portfolio(
    extractions: list[ExtractionResult], ledger: pd.DataFrame, cfg: AuditConfig | None = None
) -> list[AuditFinding]:
    actuals = compute_actuals(ledger)
    findings: list[AuditFinding] = []
    for ex in extractions:
        findings.extend(audit_contract(ex, actuals, ledger, cfg))
    LOG.info("audit produced %d findings for %d contracts", len(findings), len(extractions))
    return findings


def findings_frame(findings: list[AuditFinding]) -> pd.DataFrame:
    cols = [
        "contract_id", "borrower", "period", "covenant_type", "operator", "threshold", "unit",
        "actual_value", "headroom_pct", "status", "source_page", "evidence_quote", "note", "citation",
    ]
    if not findings:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([f.to_row() for f in findings])[cols]


def summarize_findings(findings: list[AuditFinding]) -> dict[str, int]:
    out = {s.value: 0 for s in FindingStatus}
    for f in findings:
        out[f.status.value] += 1
    out["borrowers_in_breach"] = len({f.contract_id for f in findings if f.status == FindingStatus.BREACH})
    return out
