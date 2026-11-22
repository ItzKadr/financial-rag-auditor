"""Synthetic corpus: multi-page credit agreement PDFs, ground truth and a Master Ledger.

Every contract is rendered with PyMuPDF (fitz) from a randomly sampled ``ContractSpec``:

* legal structure: title page, table of contents, parties clause, definitions, articles,
  boilerplate, schedules and a two-column notices / signature block;
* financial covenants written either as prose clauses with varied wording/operators, as a
  ruled table ("Financial Covenant Levels"), or mixed;
* numbers in many surface forms ("3.50x", "3.5 to 1.0", "3.50:1.00", "USD 12,500,000",
  "USD 12.5 million", "twelve percent (12%)");
* distractors: definitions without thresholds, representations quoting *current* ratios,
  a margin ratchet table keyed on leverage, repayment schedule amounts, baskets and cure
  amounts introduced by "not exceeding";
* running headers / footers with page numbers (to be stripped by the parser).

The Master Ledger holds quarterly actual financials per borrower. Values are generated
with headroom against the borrower's covenants and then *deliberately* pushed through the
threshold for a few (covenant, quarter) pairs ("injected breaches"); some cells are blanked
to exercise the INSUFFICIENT_DATA path.

All entities are fictional.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

from .auditor import METRIC_DEPENDENCIES, is_test_quarter, period_label, period_end
from .schemas import CovenantType, Frequency, Unit
from .text_utils import MONTHS, fmt_compact_decimal, fmt_thousands, int_to_words

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Configuration & specs
# --------------------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    n_contracts: int = 24
    seed: int = 42
    n_quarters: int = 12
    start_year: int = 2022
    breach_borrower_frac: float = 0.6
    max_breaches_per_borrower: int = 2
    missing_borrower_frac: float = 0.2
    holdout_phrasing_frac: float = 0.2  # prose covenants written with paraphrases unknown to the rules


@dataclass
class CovenantSpec:
    covenant_type: str
    operator: str
    threshold: float
    unit: str
    test_frequency: str
    surface: str = ""
    presentation: str = "prose"  # "prose" | "table"
    template_id: int = -1
    phrasing: str = "standard"  # "standard" | "holdout" | "table"
    evidence: str = ""
    page: int = 0


@dataclass
class ContractSpec:
    contract_id: str
    borrower_id: str
    borrower: str
    lender: str
    entity_desc: str
    signing_date: date
    currency: str
    principal: float
    rate_type: str
    base_rate: str | None
    rate_pct: float
    maturity_date: date
    tenor_years: int
    test_frequency: str
    layout: str  # "prose" | "table" | "mixed"
    has_ratchet: bool
    governing_law: str
    annual_ebitda: float
    names: dict[str, str]
    covenants: list[CovenantSpec] = field(default_factory=list)


@dataclass
class GeneratedContract:
    contract_id: str
    pdf_bytes: bytes
    ground_truth: dict
    spec: ContractSpec | None = None


@dataclass
class Corpus:
    contracts: list[GeneratedContract]
    ledger: pd.DataFrame
    config: SyntheticConfig

    @property
    def ground_truth(self) -> dict[str, dict]:
        return {c.contract_id: c.ground_truth for c in self.contracts}

    @property
    def pdfs(self) -> dict[str, bytes]:
        return {c.contract_id: c.pdf_bytes for c in self.contracts}


# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------
_PREFIXES = [
    "Altyn", "Steppe", "Caspian", "Nomad", "Tengri", "Aral", "Saryarka", "Irtysh", "Zhetysu",
    "Kokshe", "Ulytau", "Borealis", "Meridian", "Granite", "Silverline", "Northwind",
    "Bluewater", "Ironbridge", "Sunfield", "Emerald", "Falcon", "Orion", "Polaris", "Redstone",
]
_INDUSTRIES = [
    "Agro", "Logistics", "Energy", "Mining", "Retail", "Telecom", "Pharma", "Construction",
    "Foods", "Metals", "Petrochem", "Textiles", "Grain", "Cement",
]
_SUFFIXES = {
    "LLP": "a limited liability partnership incorporated under the laws of the Republic of Kazakhstan",
    "JSC": "a joint stock company incorporated under the laws of the Republic of Kazakhstan",
    "Ltd": "a private limited company incorporated in England and Wales",
    "plc": "a public limited company incorporated in England and Wales",
    "B.V.": "a private limited liability company incorporated in the Netherlands",
    "LLC": "a limited liability company organised under the laws of the State of Delaware",
}
_LENDERS = [
    "Northbridge Commercial Bank", "Meridian Capital Bank", "Steppe Development Bank",
    "Harbour Trust Bank", "Eastgate Investment Bank", "Silk Road Credit Bank",
]
_STREETS = ["Abai", "Dostyk", "Harbour", "Kingsway", "Market", "Tole Bi", "Kabanbay", "Riverside"]
_CITIES = ["Almaty", "Astana", "London", "Amsterdam", "Wilmington", "Shymkent", "Atyrau"]
_CURRENCY_WORDS = {"USD": "US Dollars", "EUR": "euro", "KZT": "Kazakhstan tenge"}
_FX = {"USD": 1.0, "EUR": 0.9, "KZT": 450.0}
_STEP = {"USD": 250_000.0, "EUR": 250_000.0, "KZT": 50_000_000.0}
_BASE_RATE = {"USD": "Term SOFR", "EUR": "EURIBOR", "KZT": "TONIA"}

_RATIO_THRESHOLDS = {
    CovenantType.MAX_LEVERAGE: [2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50],
    CovenantType.MIN_DSCR: [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50],
    CovenantType.MIN_INTEREST_COVERAGE: [2.00, 2.25, 2.50, 2.75, 3.00, 3.50, 4.00],
    CovenantType.MIN_CURRENT_RATIO: [1.00, 1.10, 1.15, 1.20, 1.25, 1.30, 1.50],
}
_COVENANT_PROB = {
    CovenantType.MAX_LEVERAGE: 0.9,
    CovenantType.MIN_INTEREST_COVERAGE: 0.75,
    CovenantType.MIN_DSCR: 0.6,
    CovenantType.MIN_CURRENT_RATIO: 0.5,
    CovenantType.MAX_CAPEX: 0.5,
    CovenantType.MIN_NET_WORTH: 0.5,
}

# (template, operator, forced ratio style)
_TEMPLATES: dict[CovenantType, list[tuple[str, str, str | None]]] = {
    CovenantType.MAX_LEVERAGE: [
        ("The Borrower shall ensure that the {name} in respect of any Relevant Period ending on a Test Date shall not exceed {v}.", "<=", None),
        ('The ratio of Total Debt to EBITDA (the "{name}") shall not, on any Test Date, be greater than {v}.', "<=", None),
        ("The Borrower shall procure that on each Test Date Total Debt does not exceed {v} EBITDA for the Relevant Period ending on that Test Date.", "<=", "times"),
        ("The {name} shall be less than {v} on each Test Date.", "<", None),
        ("The Borrower shall maintain a {name} of no more than {v} at all times.", "<=", None),
    ],
    CovenantType.MIN_DSCR: [
        ("The Borrower shall maintain a {name} of not less than {v} in respect of each Relevant Period.", ">=", None),
        ('Debt service cover (being the ratio of EBITDA to Debt Service, the "{name}") in respect of any Relevant Period shall be at least {v}.', ">=", None),
        ("The {name} for each Relevant Period ending on a Test Date shall be greater than {v}.", ">", None),
        ("The {name} shall not, on any Test Date, fall below {v}.", ">=", None),
    ],
    CovenantType.MIN_INTEREST_COVERAGE: [
        ('The ratio of EBITDA to Net Finance Charges (the "{name}") in respect of any Relevant Period shall not be less than {v}.', ">=", None),
        ("The Borrower shall ensure that the {name} is at least {v} on each Test Date.", ">=", None),
        ("{name} shall not, in respect of any Relevant Period, fall below {v}.", ">=", None),
        ("The {name} in respect of each Relevant Period shall exceed {v}.", ">", None),
    ],
    CovenantType.MIN_CURRENT_RATIO: [
        ("The Borrower shall at all times maintain a {name} of at least {v}.", ">=", None),
        ('The ratio of Current Assets to Current Liabilities (the "{name}") shall not be less than {v} on any Test Date.', ">=", None),
        ("The {name} shall not fall below {v} as at each Test Date.", ">=", None),
    ],
    CovenantType.MAX_CAPEX: [
        ("The aggregate {name} of the Group in any Financial Year shall not exceed {v}.", "<=", None),
        ("The Borrower shall not incur {name} in excess of {v} in aggregate in any Financial Year.", "<=", None),
        ("{name} for each Financial Year shall be limited to a maximum of {v}.", "<=", None),
    ],
    CovenantType.MIN_NET_WORTH: [
        ("The Borrower shall ensure that its {name} is not, at any time, less than {v}.", ">=", None),
        ("{name} shall at all times be at least {v}.", ">=", None),
        ("The Borrower shall maintain a minimum {name} of {v} on each Test Date.", ">=", None),
    ],
}
# Paraphrases the offline rule set was NOT written for (no modal verb, unusual operator
# cues, split concept names). They measure how brittle rules are on unseen wording; an LLM
# extractor is expected to handle them, and the guardrails must keep precision either way.
_HOLDOUT_TEMPLATES: dict[CovenantType, list[tuple[str, str, str | None]]] = {
    CovenantType.MAX_LEVERAGE: [
        ("Leverage must at no time rise above {v}.", "<=", None),
        ("Total Debt of the Group is to be kept at or below {v} EBITDA on every Test Date.", "<=", "times"),
    ],
    CovenantType.MIN_DSCR: [
        ("Debt service cover is to be maintained at or above {v} for each Relevant Period.", ">=", None),
    ],
    CovenantType.MIN_INTEREST_COVERAGE: [
        ("The Borrower undertakes to keep {name} at a level of {v} or higher on each Test Date.", ">=", None),
    ],
    CovenantType.MIN_CURRENT_RATIO: [
        ("Current Assets must at all times amount to at least {v} Current Liabilities.", ">=", "times"),
    ],
    CovenantType.MAX_CAPEX: [
        ("No member of the Group may make {name} above {v} in any Financial Year.", "<=", None),
    ],
    CovenantType.MIN_NET_WORTH: [
        ("{name} must remain above {v} throughout the life of the Facility.", ">", None),
    ],
}
_CLAUSE_LABELS = {
    CovenantType.MAX_LEVERAGE: "Leverage",
    CovenantType.MIN_DSCR: "Debt service cover",
    CovenantType.MIN_INTEREST_COVERAGE: "Interest cover",
    CovenantType.MIN_CURRENT_RATIO: "Liquidity",
    CovenantType.MAX_CAPEX: "Capital expenditure",
    CovenantType.MIN_NET_WORTH: "Net worth",
}
_TABLE_LABELS = {
    CovenantType.MAX_LEVERAGE: "Leverage Ratio (Total Debt to EBITDA)",
    CovenantType.MIN_DSCR: "{dscr} (EBITDA to Debt Service)",
    CovenantType.MIN_INTEREST_COVERAGE: "{icr} (EBITDA to Net Finance Charges)",
    CovenantType.MIN_CURRENT_RATIO: "Current Ratio (Current Assets to Current Liabilities)",
    CovenantType.MAX_CAPEX: "Capital Expenditure (per Financial Year)",
    CovenantType.MIN_NET_WORTH: "Tangible Net Worth",
}
_FREQ_WORDS = {"quarterly": "Quarterly", "semi_annual": "Semi-annually", "annual": "Annually"}

_BOILERPLATE = {
    "construction": [
        "Unless a contrary indication appears, any reference in this Agreement to a person shall be construed so as to include its successors in title, permitted assigns and permitted transferees.",
        "Section, Clause and Schedule headings are for ease of reference only and shall not affect the interpretation of this Agreement.",
        "A Default is continuing if it has not been remedied or waived, and an Event of Default is continuing if it has not been waived in writing by the Lender.",
    ],
    "purpose": [
        "The Borrower shall apply all amounts borrowed by it under the Facility towards the refinancing of existing indebtedness and the general corporate purposes of the Group.",
        "The Lender is not bound to monitor or verify the application of any amount borrowed pursuant to this Agreement.",
    ],
    "utilisation": [
        "The Borrower may not deliver a Utilisation Request unless the Lender has received all of the documents and other evidence listed in Schedule 1 (Conditions Precedent) in form and substance satisfactory to the Lender.",
        "The Lender will only be obliged to comply with a Utilisation Request if on the date of the Utilisation Request and on the proposed Utilisation Date no Default is continuing or would result from the proposed Loan.",
    ],
    "tax": [
        "The Borrower shall make all payments to be made by it without any Tax Deduction, unless a Tax Deduction is required by law.",
        "If a Tax Deduction is required by law to be made by the Borrower, the amount of the payment due from the Borrower shall be increased to an amount which leaves an amount equal to the payment which would have been due if no Tax Deduction had been required.",
    ],
    "payments": [
        "On each date on which the Borrower is required to make a payment under this Agreement, the Borrower shall make the same available to the Lender for value on the due date at the time and in such funds as the Lender may specify.",
        "Any payment which is due to be made on a day that is not a Business Day shall be made on the next Business Day in the same calendar month (if there is one) or the preceding Business Day (if there is not).",
    ],
    "negative_pledge": [
        "The Borrower shall not, and shall procure that no other member of the Group will, create or permit to subsist any Security over any of its assets other than Permitted Security.",
    ],
    "reps": [
        "The Borrower is a limited liability entity, duly incorporated and validly existing under the law of its jurisdiction of incorporation.",
        "The obligations expressed to be assumed by the Borrower in this Agreement are legal, valid, binding and enforceable obligations.",
        "No litigation, arbitration or administrative proceedings of or before any court, arbitral body or agency which, if adversely determined, might reasonably be expected to have a Material Adverse Effect have been started or threatened against the Borrower.",
    ],
}


# --------------------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------------------
def fmt_ratio(value: float, style: str) -> str:
    if style == "x":
        return f"{value:.2f}x"
    if style == "colon":
        return f"{value:.2f}:1.00"
    if style == "to":
        return f"{fmt_compact_decimal(value)} to 1.0"
    if style == "times":
        return f"{value:.2f} times"
    if style == "to1":
        return f"{value:.2f} to 1"
    raise ValueError(f"unknown ratio style {style!r}")


def _fmt_millions(amount: float) -> str:
    m = amount / 1e6
    if float(m).is_integer():
        return f"{int(m):,}"
    return f"{m:,.2f}".rstrip("0").rstrip(".")


def fmt_amount(amount: float, currency: str, style: str) -> str:
    if style == "code":
        return f"{currency} {fmt_thousands(amount)}"
    if style == "code_words":
        return f"{currency} {fmt_thousands(amount)} ({int_to_words(int(amount))} {_CURRENCY_WORDS[currency]})"
    if style == "million":
        return f"{currency} {_fmt_millions(amount)} million"
    if style == "symbol":
        return f"US${fmt_thousands(amount)}"
    if style == "suffix":
        return f"{fmt_thousands(amount)} {_CURRENCY_WORDS[currency]}"
    raise ValueError(f"unknown amount style {style!r}")


def fmt_date(d: date, style: str) -> str:
    if style == "dmy":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if style == "mdy":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    return d.isoformat()


def _round_to(x: float, step: float) -> float:
    return float(max(step, round(x / step) * step))


# --------------------------------------------------------------------------------------
# PDF layout engine
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=65536)
def _text_width(text: str, font: str, size: float) -> float:
    return pymupdf.get_text_length(text, fontname=font, fontsize=size)


class PdfWriter:
    """Minimal flow-layout engine on top of PyMuPDF (fitz) Base-14 fonts.

    Paragraphs and tables are never split across pages, so every clause has exactly one
    page number (used for citations and ground truth).
    """

    def __init__(self, width: float = 595.0, height: float = 842.0, margin: float = 72.0) -> None:
        self.doc = pymupdf.open()
        self.width, self.height, self.margin = width, height, margin
        self.text_width = width - 2 * margin
        self.page: pymupdf.Page | None = None
        self.y = margin
        self.new_page()

    @property
    def page_no(self) -> int:
        return len(self.doc)

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=self.width, height=self.height)
        self.y = self.margin

    def _ensure(self, needed: float) -> None:
        if self.y + needed > self.height - self.margin and self.y > self.margin + 1:
            self.new_page()

    @staticmethod
    def wrap(text: str, font: str, size: float, width: float) -> list[str]:
        """Greedy word wrap. Base-14 widths are additive (no kerning), so word widths are cached."""
        space = _text_width(" ", font, size)
        lines: list[str] = []
        current: list[str] = []
        cur_w = 0.0
        for word in text.split():
            w = _text_width(word, font, size)
            trial_w = w if not current else cur_w + space + w
            if trial_w <= width or not current:
                current.append(word)
                cur_w = trial_w
            else:
                lines.append(" ".join(current))
                current, cur_w = [word], w
        if current:
            lines.append(" ".join(current))
        return lines

    def spacer(self, h: float) -> None:
        self.y += h

    def heading(self, text: str, size: float = 11.5, space_before: float = 10.0) -> int:
        self._ensure(space_before + size * 1.5 + 4 * 13.5)
        if self.y > self.margin + 1:
            self.y += space_before
        assert self.page is not None
        self.page.insert_text((self.margin, self.y + size), text, fontname="tibo", fontsize=size)
        self.y += size * 1.9
        return self.page_no

    def paragraph(
        self,
        text: str,
        size: float = 10.5,
        font: str = "tiro",
        indent: float = 0.0,
        align: str = "left",
        space_after: float = 6.0,
    ) -> int:
        lh = size * 1.3
        lines = self.wrap(text, font, size, self.text_width - indent)
        self._ensure(len(lines) * lh)
        assert self.page is not None
        for line in lines:
            x = self.margin + indent
            if align == "center":
                x = self.margin + (self.text_width - pymupdf.get_text_length(line, fontname=font, fontsize=size)) / 2
            self.page.insert_text((x, self.y + size), line, fontname=font, fontsize=size)
            self.y += lh
        self.y += space_after
        return self.page_no

    def table(self, rows: list[list[str]], col_fracs: list[float], size: float = 9.0) -> int:
        widths = [f * self.text_width for f in col_fracs]
        lh = size * 1.25
        wrapped = [
            [self.wrap(cell, "tibo" if r == 0 else "tiro", size, w - 6) or [""] for cell, w in zip(row, widths)]
            for r, row in enumerate(rows)
        ]
        heights = [max(len(c) for c in row) * lh + 6 for row in wrapped]
        self._ensure(sum(heights) + 10)
        assert self.page is not None
        y = self.y + 2
        for r, (row, h) in enumerate(zip(wrapped, heights)):
            x = self.margin
            for lines, w in zip(row, widths):
                rect = pymupdf.Rect(x, y, x + w, y + h)
                self.page.draw_rect(rect, color=(0, 0, 0), fill=(0.88, 0.88, 0.88) if r == 0 else None, width=0.6)
                for k, line in enumerate(lines):
                    self.page.insert_text(
                        (x + 3, y + 3 + size + k * lh), line, fontname="tibo" if r == 0 else "tiro", fontsize=size
                    )
                x += w
            y += h
        self.y = y + 10
        return self.page_no

    def two_columns(self, left: list[str], right: list[str], size: float = 10.0) -> int:
        lh = size * 1.35
        self._ensure(max(len(left), len(right)) * lh + 6)
        assert self.page is not None
        col_x = self.margin + self.text_width / 2 + 10
        for k, line in enumerate(left):
            self.page.insert_text((self.margin, self.y + size + k * lh), line, fontname="tiro", fontsize=size)
        for k, line in enumerate(right):
            self.page.insert_text((col_x, self.y + size + k * lh), line, fontname="tiro", fontsize=size)
        self.y += max(len(left), len(right)) * lh + 8
        return self.page_no

    def finalize(self, header_left: str, header_right: str, footer_left: str) -> bytes:
        n = len(self.doc)
        for i, page in enumerate(self.doc, start=1):
            page.insert_text((self.margin, 40), header_left, fontname="helv", fontsize=8)
            hr_w = pymupdf.get_text_length(header_right, fontname="helv", fontsize=8)
            page.insert_text((self.width - self.margin - hr_w, 40), header_right, fontname="helv", fontsize=8)
            footer = f"Page {i} of {n}"
            fw = pymupdf.get_text_length(footer, fontname="helv", fontsize=8)
            page.insert_text(((self.width - fw) / 2, self.height - 36), footer, fontname="helv", fontsize=8)
            page.insert_text((self.margin, self.height - 36), footer_left, fontname="helv", fontsize=8)
        data = self.doc.tobytes(garbage=3, deflate=True)
        self.doc.close()
        return data


# --------------------------------------------------------------------------------------
# Spec sampling
# --------------------------------------------------------------------------------------
def _unique_names(n: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    """Unique fictional borrower names as ``(name, legal_form)`` pairs."""
    combos = [(p, i) for p in _PREFIXES for i in _INDUSTRIES]
    order = rng.permutation(len(combos))
    suffixes = list(_SUFFIXES)
    names: list[tuple[str, str]] = []
    for k in range(n):
        p, ind = combos[int(order[k % len(combos)])]
        suffix = suffixes[int(rng.integers(len(suffixes)))]
        series = "" if k < len(combos) else f" {k // len(combos) + 1}"
        names.append((f"{p} {ind}{series} {suffix}", suffix))
    return names


def _sample_spec(
    idx: int, borrower: str, suffix: str, rng: np.random.Generator, holdout_frac: float = 0.0
) -> ContractSpec:
    currency = str(rng.choice(["USD", "EUR", "KZT"], p=[0.6, 0.2, 0.2]))
    fx, step = _FX[currency], _STEP[currency]
    annual_ebitda = float(rng.uniform(8e6, 60e6)) * fx
    principal = _round_to(annual_ebitda * rng.uniform(1.5, 3.0), step)
    signing = date.fromordinal(date(2021, 1, 4).toordinal() + int(rng.integers(0, 340)))
    tenor = int(rng.choice([3, 4, 5, 7]))
    maturity = date(signing.year + tenor, signing.month, min(signing.day, 28))
    rate_type = "floating" if rng.random() < 0.7 else "fixed"
    if rate_type == "floating":
        rate_pct = float(rng.choice([2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5]))
        base_rate: str | None = _BASE_RATE[currency]
    else:
        choices = [14.0, 15.0, 16.0, 17.0, 18.0] if currency == "KZT" else [8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0]
        rate_pct = float(rng.choice(choices))
        base_rate = None
    freq = "quarterly" if rng.random() < 0.75 else "semi_annual"
    names = {
        "leverage": "Leverage Ratio",
        "icr": str(rng.choice(["Interest Cover", "Interest Coverage Ratio", "Interest Cover Ratio"])),
        "dscr": str(rng.choice(["Debt Service Cover Ratio", "Debt Service Coverage Ratio", "DSCR"])),
        "cr": "Current Ratio",
        "capex": "Capital Expenditure",
        "nw": "Tangible Net Worth",
    }

    chosen = [t for t, p in _COVENANT_PROB.items() if rng.random() < p]
    remaining = [t for t in CovenantType if t not in chosen]
    while len(chosen) < 3:
        pick = remaining.pop(int(rng.integers(len(remaining))))
        chosen.append(pick)
    chosen = [t for t in CovenantType if t in chosen]  # canonical order

    layout = str(rng.choice(["prose", "table", "mixed"], p=[0.4, 0.35, 0.25]))
    covenants: list[CovenantSpec] = []
    for ctype in chosen:
        if ctype in _RATIO_THRESHOLDS:
            threshold = float(rng.choice(_RATIO_THRESHOLDS[ctype]))
            unit = Unit.RATIO.value
        elif ctype == CovenantType.MAX_CAPEX:
            threshold = _round_to(annual_ebitda * rng.uniform(0.25, 0.5), step)
            unit = Unit.CURRENCY.value
        else:
            threshold = _round_to(annual_ebitda * rng.uniform(1.2, 2.5), 2 * step)
            unit = Unit.CURRENCY.value
        if layout == "table":
            presentation = "table"
        elif layout == "mixed":
            presentation = "table" if unit == Unit.RATIO.value else "prose"
        else:
            presentation = "prose"
        phrasing = "table"
        if presentation == "prose":
            phrasing = "holdout" if rng.random() < holdout_frac else "standard"
        templates = (_HOLDOUT_TEMPLATES if phrasing == "holdout" else _TEMPLATES)[ctype]
        template_id = int(rng.integers(len(templates))) if presentation == "prose" else -1
        operator = templates[template_id][1] if presentation == "prose" else ("<=" if ctype.direction == "max" else ">=")
        cov_freq = "annual" if ctype == CovenantType.MAX_CAPEX else freq
        covenants.append(
            CovenantSpec(
                covenant_type=ctype.value,
                operator=operator,
                threshold=threshold,
                unit=unit,
                test_frequency=cov_freq,
                presentation=presentation,
                template_id=template_id,
                phrasing=phrasing,
            )
        )

    return ContractSpec(
        contract_id=f"CA{idx:03d}",
        borrower_id=f"B{idx:03d}",
        borrower=borrower,
        lender=str(rng.choice(_LENDERS)),
        entity_desc=_SUFFIXES[suffix],
        signing_date=signing,
        currency=currency,
        principal=principal,
        rate_type=rate_type,
        base_rate=base_rate,
        rate_pct=rate_pct,
        maturity_date=maturity,
        tenor_years=tenor,
        test_frequency=freq,
        layout=layout,
        has_ratchet=rate_type == "floating" and bool(rng.random() < 0.5),
        governing_law=str(rng.choice(["English law", "the law of the Astana International Financial Centre"])),
        annual_ebitda=annual_ebitda,
        names=names,
        covenants=covenants,
    )


# --------------------------------------------------------------------------------------
# Contract rendering
# --------------------------------------------------------------------------------------
def _amount_style(currency: str, rng: np.random.Generator) -> str:
    styles = ["code", "code_words", "million", "suffix"] + (["symbol"] if currency == "USD" else [])
    return str(rng.choice(styles))


def _covenant_name(spec: ContractSpec, ctype: CovenantType) -> str:
    key = {
        CovenantType.MAX_LEVERAGE: "leverage",
        CovenantType.MIN_DSCR: "dscr",
        CovenantType.MIN_INTEREST_COVERAGE: "icr",
        CovenantType.MIN_CURRENT_RATIO: "cr",
        CovenantType.MAX_CAPEX: "capex",
        CovenantType.MIN_NET_WORTH: "nw",
    }[ctype]
    return spec.names[key]


def _surface(spec: ContractSpec, cov: CovenantSpec, rng: np.random.Generator, forced_style: str | None) -> str:
    if cov.unit == Unit.RATIO.value:
        style = forced_style or str(rng.choice(["x", "colon", "to", "times", "to1"]))
        return fmt_ratio(cov.threshold, style)
    return fmt_amount(cov.threshold, spec.currency, _amount_style(spec.currency, rng))


def _definitions(spec: ContractSpec, rng: np.random.Generator, maturity_text: str, margin_text: str | None) -> list[str]:
    n = spec.names
    test_date = "each Quarter Date" if spec.test_frequency == "quarterly" else "each Half-Year Date"
    basket = fmt_amount(_round_to(spec.principal * rng.uniform(0.04, 0.1), _STEP[spec.currency]), spec.currency, "code")
    defs = {
        "Business Day": '"Business Day" means a day (other than a Saturday or Sunday) on which banks are open for general business in London and Almaty.',
        "Capital Expenditure": '"Capital Expenditure" means any expenditure which, in accordance with the Accounting Principles, is treated as capital expenditure.',
        "Compliance Certificate": '"Compliance Certificate" means a certificate substantially in the form set out in Schedule 3 (Form of Compliance Certificate).',
        "Current Assets": '"Current Assets" means the aggregate of all assets of the Group which are treated as current assets in accordance with the Accounting Principles.',
        "Current Liabilities": '"Current Liabilities" means the aggregate of all liabilities of the Group falling due within one year.',
        "Current Ratio": f'"{n["cr"]}" means the ratio of Current Assets to Current Liabilities on a Test Date.',
        "Debt Service": '"Debt Service" means, in respect of any Relevant Period, the aggregate of scheduled principal repayments and Net Finance Charges.',
        "DSCR": f'"{n["dscr"]}" means, in respect of any Relevant Period, the ratio of EBITDA to Debt Service for that Relevant Period.',
        "EBITDA": '"EBITDA" means, in respect of any Relevant Period, the consolidated operating profit of the Group before taxation, Net Finance Charges, depreciation and amortisation.',
        "Final Maturity Date": f'"Final Maturity Date" means {maturity_text}.',
        "Financial Year": '"Financial Year" means the annual accounting period of the Group ending on 31 December in each year.',
        "Interest Cover": f'"{n["icr"]}" means the ratio of EBITDA to Net Finance Charges in respect of any Relevant Period.',
        "Leverage Ratio": '"Leverage Ratio" means, in respect of any Relevant Period, the ratio of Total Debt on the last day of that Relevant Period to EBITDA in respect of that Relevant Period.',
        "Net Finance Charges": '"Net Finance Charges" means, for any Relevant Period, the aggregate amount of interest, commission, fees and other finance payments payable by the Group, less interest income.',
        "Permitted Financial Indebtedness": f'"Permitted Financial Indebtedness" means Financial Indebtedness under the Finance Documents and any other Financial Indebtedness not exceeding {basket} in aggregate at any time.',
        "Relevant Period": '"Relevant Period" means each period of twelve months ending on or about the last day of the Financial Year and each period of twelve months ending on or about each Test Date.',
        "Tangible Net Worth": f'"{n["nw"]}" means the aggregate of the paid up share capital and reserves of the Group less goodwill and other intangible assets.',
        "Test Date": f'"Test Date" means {test_date}.',
        "Total Debt": '"Total Debt" means the aggregate amount of all obligations of the Group for or in respect of Financial Indebtedness.',
    }
    if spec.test_frequency == "quarterly":
        defs["Quarter Date"] = '"Quarter Date" means each of 31 March, 30 June, 30 September and 31 December.'
    else:
        defs["Half-Year Date"] = '"Half-Year Date" means each of 30 June and 31 December.'
    if margin_text is not None:
        defs["Margin"] = f'"Margin" means {margin_text}.'
    return [defs[k] for k in sorted(defs)]


def render_contract(spec: ContractSpec, rng: np.random.Generator) -> tuple[bytes, dict]:
    """Render a contract PDF and return ``(pdf_bytes, ground_truth)``."""
    w = PdfWriter()
    cur = spec.currency
    principal_text = fmt_amount(spec.principal, cur, str(rng.choice(["code", "code_words", "million"])))
    signing_text = fmt_date(spec.signing_date, "dmy")
    maturity_style = str(rng.choice(["dmy", "mdy", "iso"], p=[0.6, 0.3, 0.1]))
    maturity_text = (
        f"the date falling {spec.tenor_years * 12} months after the date of this Agreement, being "
        f"{fmt_date(spec.maturity_date, maturity_style)}"
    )
    # ---- article plan (optional articles shift the numbering) ----
    articles = ["definitions", "facility"]
    if rng.random() < 0.7:
        articles.append("utilisation")
    articles += ["repayment", "interest", "fees"]
    if rng.random() < 0.6:
        articles.append("tax")
    articles += ["reps", "information", "covenants", "undertakings", "default"]
    if rng.random() < 0.5:
        articles.append("payments")
    articles += ["notices", "law"]
    num = {name: i for i, name in enumerate(articles, start=1)}
    titles = {
        "definitions": "DEFINITIONS AND INTERPRETATION",
        "facility": "THE FACILITY",
        "utilisation": "CONDITIONS OF UTILISATION",
        "repayment": "REPAYMENT",
        "interest": "INTEREST",
        "fees": "FEES",
        "tax": "TAX GROSS-UP",
        "reps": "REPRESENTATIONS",
        "information": "INFORMATION UNDERTAKINGS",
        "covenants": "FINANCIAL COVENANTS",
        "undertakings": "GENERAL UNDERTAKINGS",
        "default": "EVENTS OF DEFAULT",
        "payments": "PAYMENT MECHANICS",
        "notices": "NOTICES",
        "law": "GOVERNING LAW",
    }
    fc = num["covenants"]
    margin_text: str | None = None
    if spec.rate_type == "floating":
        margin_text = f"{spec.rate_pct:.2f} per cent. per annum"
        if spec.has_ratchet:
            margin_text = f"subject to Clause {num['interest']}.3 (Margin ratchet), {margin_text}"

    gt_cov: list[CovenantSpec] = []

    # ---- title page ----
    w.spacer(120)
    w.paragraph("CREDIT AGREEMENT", size=20, font="tibo", align="center", space_after=18)
    w.paragraph(f"dated {signing_text}", align="center", space_after=24)
    w.paragraph(f"{fmt_amount(spec.principal, cur, 'code')} TERM LOAN FACILITY", size=12, font="tibo", align="center", space_after=24)
    w.paragraph("for", align="center")
    w.paragraph(spec.borrower.upper(), size=12, font="tibo", align="center")
    w.paragraph("as Borrower", align="center", space_after=18)
    w.paragraph("provided by", align="center")
    w.paragraph(spec.lender.upper(), size=12, font="tibo", align="center")
    w.paragraph("as Lender", align="center", space_after=40)
    w.heading("TABLE OF CONTENTS", size=11)
    for name in articles:
        w.paragraph(f"{num[name]}. {titles[name].title()}", size=9.5, space_after=1)
    w.new_page()

    # ---- parties ----
    w.paragraph(
        f"THIS AGREEMENT is dated {signing_text} and is made between: (1) {spec.borrower}, {spec.entity_desc} "
        f'with business identification number {int(rng.integers(10**11, 10**12 - 1))} (the "Borrower"); and '
        f'(2) {spec.lender}, acting through its head office (the "Lender").'
    )
    w.paragraph("IT IS AGREED as follows:", font="tibo")

    def art(name: str) -> None:
        w.heading(f"ARTICLE {num[name]} - {titles[name]}")

    def sub(name: str, k: int, title: str) -> None:
        w.heading(f"{num[name]}.{k} {title}", size=10.5, space_before=4)

    # ---- definitions ----
    art("definitions")
    sub("definitions", 1, "Definitions")
    w.paragraph("In this Agreement:")
    for d in _definitions(spec, rng, maturity_text, margin_text):
        w.paragraph(d, indent=12)
    sub("definitions", 2, "Construction")
    for p in _BOILERPLATE["construction"]:
        w.paragraph(p)

    # ---- facility ----
    art("facility")
    sub("facility", 1, "The Facility")
    w.paragraph(
        "Subject to the terms of this Agreement, the Lender makes available to the Borrower a term loan facility "
        f"in an aggregate amount equal to {principal_text}."
    )
    sub("facility", 2, "Purpose")
    for p in _BOILERPLATE["purpose"]:
        w.paragraph(p)

    if "utilisation" in num:
        art("utilisation")
        for p in _BOILERPLATE["utilisation"]:
            w.paragraph(p)

    # ---- repayment (distractor amounts) ----
    art("repayment")
    n_inst = int(rng.integers(4, 8))
    w.paragraph(
        f"The Borrower shall repay the Loan in {int_to_words(n_inst)} semi-annual instalments on each Repayment Date "
        "in the amounts set out in the table below, with the final instalment payable on the Final Maturity Date."
    )
    inst_step = _STEP[cur] / 5
    inst = _round_to(spec.principal * 0.6 / max(1, n_inst - 1), inst_step)
    outstanding = spec.principal
    rows = [["Repayment Date", "Repayment Instalment", "Outstanding after repayment"]]
    for k in range(n_inst):
        months_before = (n_inst - 1 - k) * 6
        ym = spec.maturity_date.year * 12 + spec.maturity_date.month - 1 - months_before
        d = date(ym // 12, ym % 12 + 1, min(spec.maturity_date.day, 28))
        amount = inst if k < n_inst - 1 else outstanding
        outstanding = max(0.0, outstanding - amount)
        rows.append([fmt_date(d, "dmy"), fmt_amount(amount, cur, "code"), fmt_amount(outstanding, cur, "code")])
    w.table(rows, [0.3, 0.35, 0.35])
    w.paragraph("The Borrower may not reborrow any part of the Facility which is repaid.")

    # ---- interest ----
    art("interest")
    sub("interest", 1, "Calculation of interest")
    if spec.rate_type == "floating":
        w.paragraph(
            "The rate of interest on the Loan for each Interest Period is the percentage rate per annum which is the "
            f"aggregate of the applicable Margin and {spec.base_rate}."
        )
    else:
        r = spec.rate_pct
        rate_words = f"{int_to_words(int(r))} percent ({int(r)}%)" if float(r).is_integer() else f"{r:.2f} per cent."
        w.paragraph(f"The Loan shall bear interest at a fixed rate of {rate_words} per annum, payable quarterly in arrears.")
    sub("interest", 2, "Default interest")
    w.paragraph(
        "If the Borrower fails to pay any amount payable by it on its due date, interest shall accrue on the overdue "
        f"amount at a rate which is {float(rng.choice([1.0, 2.0, 3.0])):.2f} per cent. per annum higher than the rate "
        "which would otherwise apply."
    )
    if spec.has_ratchet:
        sub("interest", 3, "Margin ratchet")
        a = float(rng.choice([2.5, 3.0, 3.5]))
        b = a - 1.0
        m = spec.rate_pct
        w.paragraph(
            "If the Leverage Ratio in respect of the most recently completed Relevant Period is within a range set out "
            "below, the Margin shall be the percentage per annum set out opposite that range:"
        )
        w.table(
            [
                ["Leverage Ratio", "Margin (% p.a.)"],
                [f"Greater than {a:.2f}x", f"{m + 0.5:.2f}"],
                [f"Less than or equal to {a:.2f}x but greater than {b:.2f}x", f"{m:.2f}"],
                [f"Less than or equal to {b:.2f}x", f"{max(0.5, m - 0.5):.2f}"],
            ],
            [0.6, 0.4],
        )

    # ---- fees ----
    art("fees")
    w.paragraph(
        f"The Borrower shall pay to the Lender an arrangement fee in the amount of {float(rng.choice([0.5, 0.75, 1.0, 1.25])):.2f} "
        "per cent. of the Total Commitments, payable on the first Utilisation Date."
    )
    w.paragraph(
        "The Borrower shall pay to the Lender a commitment fee computed at the rate of 35 per cent. of the applicable "
        "Margin on the undrawn amount of the Facility during the Availability Period."
    )

    if "tax" in num:
        art("tax")
        for p in _BOILERPLATE["tax"]:
            w.paragraph(p)

    # ---- representations (distractor: current ratio levels, no obligation) ----
    art("reps")
    for p in _BOILERPLATE["reps"]:
        w.paragraph(p)
    lev = next((c for c in spec.covenants if c.covenant_type == CovenantType.MAX_LEVERAGE.value), None)
    icr = next((c for c in spec.covenants if c.covenant_type == CovenantType.MIN_INTEREST_COVERAGE.value), None)
    lev_now = round((lev.threshold if lev else 3.0) * float(rng.uniform(0.45, 0.8)), 2)
    icr_now = round((icr.threshold if icr else 2.5) * float(rng.uniform(1.4, 2.2)), 2)
    w.paragraph(
        f"As at the date of this Agreement, the Leverage Ratio is {lev_now:.2f}x and the {spec.names['icr']} is "
        f"{icr_now:.2f}x, in each case calculated by reference to the Original Financial Statements."
    )

    # ---- information undertakings ----
    art("information")
    w.paragraph(
        "The Borrower shall supply to the Lender as soon as they become available, but in any event within 120 days "
        "after the end of each Financial Year, its audited consolidated financial statements for that Financial Year."
    )
    w.paragraph(
        "The Borrower shall supply to the Lender within 60 days after each Test Date its unaudited consolidated "
        f"financial statements, together with a Compliance Certificate setting out computations as to compliance "
        f"with Clause {fc} (Financial Covenants)."
    )

    # ---- financial covenants ----
    art("covenants")
    sub("covenants", 1, "Financial definitions")
    w.paragraph(f"In this Clause {fc}, terms defined in Clause 1.1 (Definitions) have the same meaning when used herein.")
    sub("covenants", 2, "Financial condition")
    prose = [c for c in spec.covenants if c.presentation == "prose"]
    table_covs = [c for c in spec.covenants if c.presentation == "table"]
    if prose:
        w.paragraph("The Borrower shall ensure that:")
        for letter, cov in zip("abcdefgh", prose):
            ctype = CovenantType(cov.covenant_type)
            bank = _HOLDOUT_TEMPLATES if cov.phrasing == "holdout" else _TEMPLATES
            template, _, forced = bank[ctype][cov.template_id]
            cov.surface = _surface(spec, cov, rng, forced)
            sentence = template.format(name=_covenant_name(spec, ctype), v=cov.surface)
            cov.evidence = sentence
            cov.page = w.paragraph(f"({letter}) {_CLAUSE_LABELS[ctype]}. {sentence}", indent=12)
            gt_cov.append(cov)
    if table_covs:
        w.paragraph(
            "The Borrower shall ensure that each financial covenant set out in Schedule 2 (Financial Covenant Levels) "
            "is satisfied on each Test Date."
        )
    sub("covenants", 3, "Financial testing")
    freq_text = "quarterly on each Quarter Date" if spec.test_frequency == "quarterly" else "semi-annually on each Half-Year Date"
    w.paragraph(
        f"The financial covenants set out in Clause {fc}.2 shall be calculated in accordance with the Accounting "
        f"Principles and tested {freq_text} by reference to the financial statements and the related Compliance Certificate."
    )
    sub("covenants", 4, "Equity cure")
    cure = fmt_amount(_round_to(spec.principal * rng.uniform(0.05, 0.12), _STEP[cur]), cur, "code")
    w.paragraph(
        f"If the requirements of Clause {fc}.2 are not met, the Borrower may, within 20 Business Days, remedy such "
        f"failure by procuring new equity in an amount not exceeding {cure}. The cure right may not be exercised "
        "more than twice during the life of the Facility."
    )

    # ---- general undertakings (distractor: disposals basket) ----
    art("undertakings")
    for p in _BOILERPLATE["negative_pledge"]:
        w.paragraph(p)
    disposal = fmt_amount(_round_to(spec.principal * rng.uniform(0.02, 0.06), _STEP[cur]), cur, "code")
    w.paragraph(
        f"The Borrower shall not dispose of any asset with an aggregate value exceeding {disposal} in any Financial Year."
    )
    w.paragraph("The Borrower shall not incur any Financial Indebtedness other than Permitted Financial Indebtedness.")

    # ---- events of default ----
    art("default")
    cross = fmt_amount(_round_to(spec.principal * rng.uniform(0.05, 0.1), _STEP[cur]), cur, "code")
    w.paragraph(
        "Non-payment: the Borrower does not pay on the due date any amount payable pursuant to a Finance Document "
        "unless its failure to pay is caused by administrative or technical error and payment is made within "
        "3 Business Days of its due date."
    )
    w.paragraph(
        f"Financial covenants: any requirement of Clause {fc} (Financial Covenants) is not satisfied, subject to "
        f"Clause {fc}.4 (Equity cure)."
    )
    w.paragraph(
        f"Cross default: any Financial Indebtedness of any member of the Group in an aggregate amount exceeding "
        f"{cross} is not paid when due or is declared due and payable prior to its specified maturity."
    )

    if "payments" in num:
        art("payments")
        for p in _BOILERPLATE["payments"]:
            w.paragraph(p)

    # ---- notices (two-column block) ----
    art("notices")
    w.paragraph("The address for any communication to be made under or in connection with this Agreement is:")
    city_b, city_l = rng.choice(_CITIES, 2, replace=False)
    w.two_columns(
        [
            "Borrower:",
            spec.borrower,
            f"{int(rng.integers(1, 200))} {rng.choice(_STREETS)} Street, {city_b}",
            "Attention: Chief Financial Officer",
        ],
        [
            "Lender:",
            spec.lender,
            f"{int(rng.integers(1, 200))} {rng.choice(_STREETS)} Street, {city_l}",
            "Attention: Loan Administration",
        ],
    )

    art("law")
    w.paragraph(
        f"This Agreement and any non-contractual obligations arising out of or in connection with it are governed by {spec.governing_law}."
    )

    # ---- schedules ----
    if table_covs:
        w.new_page()
        w.heading("SCHEDULE 2 - FINANCIAL COVENANT LEVELS")
        w.paragraph(
            f"The financial covenants referred to in Clause {fc}.2 (Financial condition) are set out below. Each "
            "level applies on each Test Date unless otherwise indicated."
        )
        rows = [["Financial Covenant", "Requirement", "Level", "Testing"]]
        ratio_style = str(rng.choice(["x", "colon", "to"]))
        for cov in table_covs:
            ctype = CovenantType(cov.covenant_type)
            label = _TABLE_LABELS[ctype].format(dscr=spec.names["dscr"], icr=spec.names["icr"])
            requirement = "Maximum" if ctype.direction == "max" else "Minimum"
            if cov.unit == Unit.RATIO.value:
                cov.surface = fmt_ratio(cov.threshold, ratio_style)
            else:
                cov.surface = fmt_amount(cov.threshold, cur, str(rng.choice(["code", "million"])))
            cells = [label, requirement, cov.surface, _FREQ_WORDS[cov.test_frequency]]
            rows.append(cells)
            cov.evidence = " | ".join(cells)
            gt_cov.append(cov)
        page = w.table(rows, [0.42, 0.16, 0.24, 0.18])
        for cov in table_covs:
            cov.page = page

    # ---- signatures (two columns) ----
    w.paragraph(
        "IN WITNESS WHEREOF this Agreement has been entered into on the date stated at the beginning of this Agreement.",
        space_after=12,
    )
    w.two_columns(
        ["THE BORROWER", spec.borrower, "By: ____________________", "Name: Authorised Signatory"],
        ["THE LENDER", spec.lender, "By: ____________________", "Name: Authorised Signatory"],
    )

    pdf = w.finalize(
        header_left="CONFIDENTIAL - EXECUTION VERSION",
        header_right=f"{spec.borrower} - Credit Agreement",
        footer_left=f"Ref. {spec.contract_id}",
    )
    order = {t.value: i for i, t in enumerate(CovenantType)}
    gt_cov.sort(key=lambda c: order[c.covenant_type])
    ground_truth = {
        "contract_id": spec.contract_id,
        "borrower_id": spec.borrower_id,
        "borrower": spec.borrower,
        "lender": spec.lender,
        "layout": spec.layout,
        "has_ratchet": spec.has_ratchet,
        "test_frequency": spec.test_frequency,
        "loan_terms": {
            "borrower": spec.borrower,
            "principal": spec.principal,
            "currency": spec.currency,
            "rate_type": spec.rate_type,
            "base_rate": spec.base_rate,
            "rate_pct": spec.rate_pct,
            "maturity_date": spec.maturity_date.isoformat(),
        },
        "covenants": [asdict(c) for c in gt_cov],
    }
    return pdf, ground_truth


# --------------------------------------------------------------------------------------
# Master Ledger
# --------------------------------------------------------------------------------------
LEDGER_COLUMNS = [
    "ebitda", "total_debt", "interest_expense", "debt_service",
    "current_assets", "current_liabilities", "capex", "net_worth",
]


def _affects(column: str, t_missing: int, ctype: CovenantType, t_test: int) -> bool:
    kind = METRIC_DEPENDENCIES[ctype].get(column)
    if kind == "point":
        return t_missing == t_test
    if kind == "ltm":
        return t_test - 3 <= t_missing <= t_test
    if kind == "fy":
        return t_missing // 4 == t_test // 4
    return False


def generate_ledger_rows(
    spec: ContractSpec, rng: np.random.Generator, cfg: SyntheticConfig
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Quarterly actuals for one borrower, with injected breaches and missing cells."""
    T = cfg.n_quarters
    e_annual = spec.annual_ebitda
    growth = rng.uniform(-0.02, 0.08)
    seasonal = rng.normal(0, 0.05, 4)
    t_idx = np.arange(T)
    ebitda = e_annual / 4 * (1 + growth) ** (t_idx / 4) * (1 + seasonal[t_idx % 4]) * rng.normal(1, 0.04, T)
    ltm = np.array([ebitda[t - 3 : t + 1].sum() if t >= 3 else ebitda[t] * 4 for t in range(T)])
    thr = {CovenantType(c.covenant_type): c.threshold for c in spec.covenants}

    lev = thr.get(CovenantType.MAX_LEVERAGE)
    total_debt = ltm * ((lev or 2.0) * rng.uniform(0.55, 0.85, T))
    interest = ebitda / ((thr.get(CovenantType.MIN_INTEREST_COVERAGE) or 3.0) * rng.uniform(1.3, 1.9, T))
    debt_service = ebitda / ((thr.get(CovenantType.MIN_DSCR) or 1.3) * rng.uniform(1.25, 1.7, T))
    current_liabilities = e_annual * rng.uniform(0.3, 0.5, T)
    current_assets = current_liabilities * ((thr.get(CovenantType.MIN_CURRENT_RATIO) or 1.2) * rng.uniform(1.12, 1.45, T))
    capex_limit = thr.get(CovenantType.MAX_CAPEX) or e_annual * 0.4
    capex = capex_limit / 4 * rng.uniform(0.5, 0.85, T)
    nw_floor = thr.get(CovenantType.MIN_NET_WORTH) or e_annual * 1.5
    net_worth = nw_floor * rng.uniform(1.12, 1.5, T)

    injected: list[dict] = []
    if spec.covenants and rng.random() < cfg.breach_borrower_frac:
        k = int(rng.integers(1, cfg.max_breaches_per_borrower + 1))
        picks = rng.choice(len(spec.covenants), size=min(k, len(spec.covenants)), replace=False)
        for i in picks:
            cov = spec.covenants[int(i)]
            ctype = CovenantType(cov.covenant_type)
            freq = Frequency(cov.test_frequency)
            candidates = [t for t in range(3, T) if is_test_quarter(t % 4 + 1, freq)]
            t = int(rng.choice(candidates))
            th = cov.threshold
            if ctype == CovenantType.MAX_LEVERAGE:
                total_debt[t] = ltm[t] * th * rng.uniform(1.05, 1.30)
            elif ctype == CovenantType.MIN_INTEREST_COVERAGE:
                target = ltm[t] / (th * rng.uniform(0.70, 0.95))
                interest[t] += max(0.0, target - interest[t - 3 : t + 1].sum())
            elif ctype == CovenantType.MIN_DSCR:
                target = ltm[t] / (th * rng.uniform(0.70, 0.95))
                debt_service[t] += max(0.0, target - debt_service[t - 3 : t + 1].sum())
            elif ctype == CovenantType.MIN_CURRENT_RATIO:
                current_liabilities[t] = current_assets[t] / (th * rng.uniform(0.75, 0.95))
            elif ctype == CovenantType.MAX_CAPEX:
                fy = slice(t - 3, t + 1)
                capex[t] += th * rng.uniform(1.05, 1.35) - capex[fy].sum()
            elif ctype == CovenantType.MIN_NET_WORTH:
                net_worth[t] = th * rng.uniform(0.75, 0.95)
            injected.append({"contract_id": spec.contract_id, "covenant_type": ctype.value, "t": t,
                             "period": period_label(cfg.start_year, t)})

    frame = pd.DataFrame(
        {
            "ebitda": ebitda,
            "total_debt": total_debt,
            "interest_expense": interest,
            "debt_service": debt_service,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "capex": capex,
            "net_worth": net_worth,
        }
    ).round(0)

    missing: list[dict] = []
    if rng.random() < cfg.missing_borrower_frac:
        for _ in range(20):
            col = str(rng.choice(LEDGER_COLUMNS))
            tm = int(rng.integers(3, T))
            conflict = any(
                _affects(col, tm, CovenantType(ev["covenant_type"]), ev["t"]) for ev in injected
            )
            if not conflict:
                frame.loc[tm, col] = np.nan
                missing.append({"contract_id": spec.contract_id, "column": col, "period": period_label(cfg.start_year, tm)})
                break

    frame.insert(0, "borrower_id", spec.borrower_id)
    frame.insert(1, "borrower_name", spec.borrower)
    frame.insert(2, "currency", spec.currency)
    frame.insert(3, "period", [period_label(cfg.start_year, t) for t in range(T)])
    frame.insert(4, "period_end", [period_end(cfg.start_year, t).isoformat() for t in range(T)])
    frame.insert(5, "fiscal_year", [cfg.start_year + t // 4 for t in range(T)])
    frame.insert(6, "quarter", [t % 4 + 1 for t in range(T)])
    return frame, injected, missing


# --------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------
def generate_corpus(cfg: SyntheticConfig, out_dir: Path | None = None) -> Corpus:
    """Generate ``cfg.n_contracts`` contracts and the Master Ledger (optionally on disk).

    Layout on disk::

        out_dir/contracts/CA001.pdf
        out_dir/ground_truth/CA001.json
        out_dir/master_ledger.csv
        out_dir/manifest.json
    """
    rng = np.random.default_rng(cfg.seed)
    names = _unique_names(cfg.n_contracts, rng)
    contracts: list[GeneratedContract] = []
    ledgers: list[pd.DataFrame] = []
    for i, (borrower, suffix) in enumerate(names, start=1):
        spec = _sample_spec(i, borrower, suffix, rng, cfg.holdout_phrasing_frac)
        pdf, gt = render_contract(spec, rng)
        ledger, injected, missing = generate_ledger_rows(spec, rng, cfg)
        gt["injected_breaches"] = injected
        gt["missing_data"] = missing
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        gt["n_pages"] = len(doc)
        doc.close()
        gt["file"] = f"contracts/{spec.contract_id}.pdf"
        contracts.append(GeneratedContract(contract_id=spec.contract_id, pdf_bytes=pdf, ground_truth=gt, spec=spec))
        ledgers.append(ledger)
    ledger_df = pd.concat(ledgers, ignore_index=True)
    corpus = Corpus(contracts=contracts, ledger=ledger_df, config=cfg)
    if out_dir is not None:
        save_corpus(corpus, out_dir)
    LOG.info(
        "generated %d contracts (%d pages), %d ledger rows, %d injected breaches",
        len(contracts),
        sum(c.ground_truth["n_pages"] for c in contracts),
        len(ledger_df),
        sum(len(c.ground_truth["injected_breaches"]) for c in contracts),
    )
    return corpus


def save_corpus(corpus: Corpus, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / "contracts").mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
    for c in corpus.contracts:
        (out_dir / "contracts" / f"{c.contract_id}.pdf").write_bytes(c.pdf_bytes)
        with open(out_dir / "ground_truth" / f"{c.contract_id}.json", "w", encoding="utf-8") as fh:
            json.dump(c.ground_truth, fh, indent=2)
    corpus.ledger.to_csv(out_dir / "master_ledger.csv", index=False)
    manifest = {
        "config": asdict(corpus.config),
        "contracts": [c.contract_id for c in corpus.contracts],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_corpus(out_dir: Path) -> Corpus:
    """Reload a corpus written by :func:`save_corpus`."""
    out_dir = Path(out_dir)
    with open(out_dir / "manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    cfg = SyntheticConfig(**manifest["config"])
    contracts: list[GeneratedContract] = []
    for cid in manifest["contracts"]:
        with open(out_dir / "ground_truth" / f"{cid}.json", encoding="utf-8") as fh:
            gt = json.load(fh)
        pdf = (out_dir / "contracts" / f"{cid}.pdf").read_bytes()
        contracts.append(GeneratedContract(contract_id=cid, pdf_bytes=pdf, ground_truth=gt))
    ledger = pd.read_csv(out_dir / "master_ledger.csv")
    return Corpus(contracts=contracts, ledger=ledger, config=cfg)
