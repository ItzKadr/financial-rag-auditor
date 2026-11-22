"""Text normalisation, sentence splitting, number/date parsing and number formatting.

These helpers are shared by the synthetic generator (formatting numbers in many surface
forms), the chunker (sentence boundaries), the offline extractor (parsing thresholds) and
the guardrails (checking that a number really occurs in a quote).
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date

# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
_TRANSLATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
    }
)


def normalize_text(text: str) -> str:
    """Canonical form used for quote matching.

    Lower-cases, unifies typographic quotes/dashes, turns Markdown table pipes and
    separator rows into whitespace and collapses all whitespace runs to one space.
    """
    t = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    t = t.replace("|", " ")
    t = re.sub(r"-{3,}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def compact(text: str) -> str:
    """Alphanumeric-only fingerprint (robust to spacing / punctuation / table pipes)."""
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


_WORD_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Cheap tokenizer-free token count (words + punctuation marks).

    It over-counts slightly versus BPE tokenizers for legal prose, which makes it a
    conservative budget for chunking.
    """
    return len(_WORD_TOKEN_RE.findall(text))


# --------------------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------------------
_ABBREVIATIONS = {
    "no", "nos", "cl", "art", "sec", "co", "ltd", "inc", "llc", "llp", "jsc", "plc",
    "b.v", "u.s", "e.g", "i.e", "etc", "mr", "ms", "dr", "st", "approx", "vs", "para",
    "pp", "p", "corp", "s.a",
}
_BOUNDARY_RE = re.compile(r"([.!?][\"')\]]?)\s+(?=[A-Z(\"\d])")


def split_sentences(text: str) -> list[str]:
    """Split legal prose into sentences.

    A boundary is ``[.!?]`` followed by whitespace and an upper-case letter, digit, quote
    or opening parenthesis. Decimal numbers ("3.50x") never match because no whitespace
    follows the dot; abbreviations ("B.V.", "Ltd.", "e.g.") and single initials are kept.
    """
    text = text.strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        head = text[start : m.start(1)]
        prev = re.search(r"([A-Za-z][A-Za-z.]*)$", head)
        if prev:
            word = prev.group(1)
            if word.lower().rstrip(".") in _ABBREVIATIONS:
                continue
            if len(word) == 1 and word.isupper():
                continue
        end = m.start(1) + len(m.group(1))
        piece = text[start:end].strip()
        if piece:
            sentences.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# --------------------------------------------------------------------------------------
# Number parsing
# --------------------------------------------------------------------------------------
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SCALES = {
    "thousand": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
}
_CURRENCY_ALIASES = {
    "usd": "USD",
    "us$": "USD",
    "$": "USD",
    "us dollars": "USD",
    "united states dollars": "USD",
    "dollars": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "kzt": "KZT",
    "tenge": "KZT",
    "kazakhstan tenge": "KZT",
}

_CUR_PREFIX_RE = re.compile(
    rf"(?P<cur>US\$|USD|EUR|KZT|\$)\s?(?P<num>{_NUM})(?:\s?(?P<scale>million|billion|thousand|mn|bn|m)\b)?",
    re.IGNORECASE,
)
_CUR_SUFFIX_RE = re.compile(
    rf"(?P<num>{_NUM})(?:\s(?P<scale>million|billion|thousand))?\s"
    r"(?P<cur>United States Dollars|US Dollars|Dollars|USD|EUR|euros?|KZT|Kazakhstan tenge|tenge)\b",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s?(?:x\b|times\b|:\s?1(?:\.0+)?\b|to\s1(?:\.0+)?\b)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s?(?:%|per\s?cent\b\.?|percent\b)",
    re.IGNORECASE,
)
_PLAIN_RE = re.compile(_NUM)


@dataclass(frozen=True)
class NumberMention:
    """A number found in text together with its semantic kind."""

    value: float
    kind: str  # "currency" | "ratio" | "percent" | "number"
    start: int
    end: int
    text: str
    currency: str | None = None


def _to_float(num: str) -> float:
    return float(num.replace(",", ""))


def parse_numbers(text: str) -> list[NumberMention]:
    """Extract typed numeric mentions from ``text``.

    Priority when spans overlap: currency > ratio > percent > plain number. Scale words
    are applied ("USD 12.5 million" -> 12_500_000).
    """
    candidates: list[tuple[int, NumberMention]] = []
    for m in _CUR_PREFIX_RE.finditer(text):
        scale = _SCALES.get((m.group("scale") or "").lower(), 1.0)
        cur = _CURRENCY_ALIASES.get(m.group("cur").lower(), m.group("cur").upper())
        candidates.append(
            (0, NumberMention(_to_float(m.group("num")) * scale, "currency", m.start(), m.end(), m.group(0), cur))
        )
    for m in _CUR_SUFFIX_RE.finditer(text):
        scale = _SCALES.get((m.group("scale") or "").lower(), 1.0)
        cur = _CURRENCY_ALIASES.get(m.group("cur").lower(), m.group("cur").upper())
        candidates.append(
            (0, NumberMention(_to_float(m.group("num")) * scale, "currency", m.start(), m.end(), m.group(0), cur))
        )
    for m in _RATIO_RE.finditer(text):
        candidates.append((1, NumberMention(float(m.group("num")), "ratio", m.start(), m.end(), m.group(0))))
    for m in _PERCENT_RE.finditer(text):
        candidates.append((2, NumberMention(float(m.group("num")), "percent", m.start(), m.end(), m.group(0))))

    candidates.sort(key=lambda c: (c[0], -(c[1].end - c[1].start), c[1].start))
    chosen: list[NumberMention] = []
    for _, cand in candidates:
        if all(cand.end <= c.start or cand.start >= c.end for c in chosen):
            chosen.append(cand)
    for m in _PLAIN_RE.finditer(text):
        if all(m.end() <= c.start or m.start() >= c.end for c in chosen):
            chosen.append(NumberMention(_to_float(m.group(0)), "number", m.start(), m.end(), m.group(0)))
    chosen.sort(key=lambda c: c.start)
    return chosen


def numeric_values(text: str) -> list[float]:
    """All numeric values in ``text``: typed mentions plus every raw digit group."""
    values = [m.value for m in parse_numbers(text)]
    values.extend(_to_float(m.group(0)) for m in _PLAIN_RE.finditer(text))
    return values


def contains_value(text: str, value: float, rel_tol: float = 1e-6) -> bool:
    """True if ``value`` occurs (in any supported surface form) inside ``text``."""
    return any(math.isclose(v, value, rel_tol=rel_tol, abs_tol=1e-9) for v in numeric_values(text))


# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_IDX = {m.lower(): i for i, m in enumerate(MONTHS, start=1)}
_MONTH_ALT = "|".join(MONTHS)
_DATE_DMY_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\s+(\d{{4}})\b", re.IGNORECASE)
_DATE_MDY_RE = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{1,2}}),\s*(\d{{4}})\b", re.IGNORECASE)
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def parse_dates(text: str) -> list[date]:
    """Parse '15 March 2028', 'March 15, 2028', '2028-03-15' and '15/03/2028' (D/M/Y)."""
    found: list[tuple[int, date]] = []

    def _add(pos: int, y: int, mo: int, d: int) -> None:
        try:
            found.append((pos, date(y, mo, d)))
        except ValueError:
            return

    for m in _DATE_DMY_RE.finditer(text):
        _add(m.start(), int(m.group(3)), _MONTH_IDX[m.group(2).lower()], int(m.group(1)))
    for m in _DATE_MDY_RE.finditer(text):
        _add(m.start(), int(m.group(3)), _MONTH_IDX[m.group(1).lower()], int(m.group(2)))
    for m in _DATE_ISO_RE.finditer(text):
        _add(m.start(), int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in _DATE_SLASH_RE.finditer(text):
        _add(m.start(), int(m.group(3)), int(m.group(2)), int(m.group(1)))
    found.sort(key=lambda x: x[0])
    return [d for _, d in found]


# --------------------------------------------------------------------------------------
# Formatting (used by the synthetic generator)
# --------------------------------------------------------------------------------------
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def int_to_words(n: int) -> str:
    """English words for a non-negative integer (short scale)."""
    if n < 0:
        raise ValueError("only non-negative integers are supported")
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" and {int_to_words(rest)}" if rest else "")
    for scale_value, scale_name in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        if n >= scale_value:
            head, rest = divmod(n, scale_value)
            words = f"{int_to_words(head)} {scale_name}"
            if rest:
                words += (" " if rest >= 100 else " and ") + int_to_words(rest)
            return words
    raise AssertionError("unreachable")


def fmt_thousands(value: float) -> str:
    """12500000 -> '12,500,000' (drops a zero fractional part)."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def fmt_compact_decimal(value: float) -> str:
    """3.50 -> '3.5', 3.25 -> '3.25', 2.0 -> '2.0'."""
    s = f"{value:.2f}".rstrip("0")
    return s + "0" if s.endswith(".") else s
