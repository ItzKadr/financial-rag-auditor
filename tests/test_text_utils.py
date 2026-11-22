from datetime import date

import pytest

from src.text_utils import (
    compact,
    contains_value,
    count_tokens,
    int_to_words,
    normalize_text,
    parse_dates,
    parse_numbers,
    split_sentences,
)


@pytest.mark.parametrize(
    "text,value,kind",
    [
        ("shall not exceed 3.50x.", 3.5, "ratio"),
        ("shall not be less than 3.5 to 1.0", 3.5, "ratio"),
        ("of at least 1.25:1.00 on each Test Date", 1.25, "ratio"),
        ("does not exceed 4.00 times EBITDA", 4.0, "ratio"),
        ("a maximum of USD 12,500,000 in any Financial Year", 12_500_000, "currency"),
        ("at least USD 12.5 million", 12_500_000, "currency"),
        ("US$7,250,000 in aggregate", 7_250_000, "currency"),
        ("not less than 48,000,000 US Dollars", 48_000_000, "currency"),
        ("KZT 9,500 million", 9_500_000_000, "currency"),
        ("twelve percent (12%) per annum", 12.0, "percent"),
        ("3.25 per cent. per annum", 3.25, "percent"),
    ],
)
def test_parse_numbers_surface_forms(text: str, value: float, kind: str) -> None:
    mentions = [m for m in parse_numbers(text) if m.kind == kind]
    assert mentions, f"no {kind} found in {text!r}"
    assert mentions[0].value == pytest.approx(value)


def test_ratio_denominator_is_not_a_separate_mention() -> None:
    mentions = parse_numbers("not less than 1.25:1.00")
    assert [m.value for m in mentions] == [1.25]
    # but the raw-number view still sees it (used only for non-typed checks)
    assert contains_value("1.25:1.00", 1.0)


def test_currency_code_is_recorded() -> None:
    m = parse_numbers("aggregate amount equal to EUR 40,000,000")[0]
    assert m.currency == "EUR" and m.value == 40_000_000


def test_split_sentences_keeps_decimals_and_abbreviations() -> None:
    text = (
        "(a) Leverage. The Leverage Ratio shall not exceed 3.50x. "
        "Irtysh Logistics B.V. (the \"Borrower\") shall comply. Margin is 3.25 per cent. per annum."
    )
    sents = split_sentences(text)
    assert sents[0] == "(a) Leverage."
    assert sents[1] == "The Leverage Ratio shall not exceed 3.50x."
    assert any("B.V. (the" in s for s in sents)
    assert sents[-1].endswith("per cent. per annum.")


def test_int_to_words() -> None:
    assert int_to_words(12) == "twelve"
    assert int_to_words(12_500_000) == "twelve million five hundred thousand"
    assert int_to_words(48_000_000) == "forty-eight million"
    assert int_to_words(101) == "one hundred and one"


def test_parse_dates_formats() -> None:
    target = date(2028, 3, 15)
    for text in ("being 15 March 2028.", "being March 15, 2028.", "being 2028-03-15.", "on 15/03/2028"):
        assert parse_dates(text) == [target]


def test_normalize_and_compact() -> None:
    row = "|Leverage Ratio|Maximum|3.50x|Quarterly|"
    assert normalize_text(row) == "leverage ratio maximum 3.50x quarterly"
    assert compact(row) == compact("Leverage Ratio | Maximum | 3.50x | Quarterly")
    assert normalize_text("“Borrower”  \n means") == '"borrower" means'


def test_count_tokens() -> None:
    assert count_tokens("The Borrower shall not exceed 3.50x.") == 9
