from src.guardrails import SourceText, find_quote, value_grounded, verify_covenant, verify_loan_terms
from src.schemas import Covenant, LoanTerms

SOURCE = [
    SourceText(
        "(a) Leverage. The Borrower shall ensure that the Leverage Ratio in respect of any Relevant Period "
        "ending on a Test Date shall not exceed 3.50x. (b) Interest cover. The Interest Cover shall not, "
        "in respect of any Relevant Period, fall below 2.75 to 1.0.",
        chunk_id="C1-0001", page=5,
    ),
    SourceText("|Debt Service Cover Ratio (EBITDA to Debt Service)|Minimum|1.25:1.00|Quarterly|", chunk_id="C1-0002", page=7),
    SourceText(
        'This Agreement is made between: (1) Altyn Agro LLP, a limited liability partnership (the "Borrower"). '
        "The Lender makes available a term loan facility in an aggregate amount equal to USD 50,000,000.",
        chunk_id="C1-0003", page=2,
    ),
]
PAGES = {2: SOURCE[2].text, 5: SOURCE[0].text, 7: SOURCE[1].text}


def _cov(**kw) -> Covenant:
    base = dict(covenant_type="max_leverage", operator="<=", threshold=3.5, unit="ratio", test_frequency="quarterly",
                source_page=5, evidence_quote="the Leverage Ratio in respect of any Relevant Period ending on a Test Date shall not exceed 3.50x.")
    return Covenant(**(base | kw))


def test_genuine_covenant_is_verified() -> None:
    check = verify_covenant(_cov(), SOURCE, PAGES)
    assert check.accepted and check.quote_match == "exact" and check.located_page == 5 and check.chunk_id == "C1-0001"


def test_hallucinated_value_is_rejected() -> None:
    # the model copied the right sentence but reported a number that is not in it
    check = verify_covenant(_cov(threshold=4.0), SOURCE, PAGES)
    assert not check.accepted
    assert not check.number_grounded
    assert any("does not occur" in r for r in check.reasons)


def test_near_copy_quote_with_swapped_number_is_rejected() -> None:
    # 97% identical to the real sentence -> fuzzy match succeeds, but the *source* says 3.50x
    fake = "the Leverage Ratio in respect of any Relevant Period ending on a Test Date shall not exceed 4.25x."
    check = verify_covenant(_cov(threshold=4.25, evidence_quote=fake), SOURCE, PAGES)
    assert check.quote_match == "fuzzy"
    assert not check.accepted and not check.number_grounded


def test_fabricated_quote_is_rejected() -> None:
    fake = "The Borrower covenants that leverage will never be above 5.00x during the term."
    check = verify_covenant(_cov(threshold=5.0, evidence_quote=fake), SOURCE, PAGES)
    assert not check.accepted and check.quote_match == "none"


def test_wrong_covenant_type_is_rejected() -> None:
    quote = "The Interest Cover shall not, in respect of any Relevant Period, fall below 2.75 to 1.0."
    check = verify_covenant(_cov(covenant_type="min_dscr", operator=">=", threshold=2.75, evidence_quote=quote), SOURCE, PAGES)
    assert check.number_grounded and not check.keyword_ok and not check.accepted


def test_fuzzy_match_tolerates_spacing_and_pipes() -> None:
    quote = "Debt Service Cover Ratio (EBITDA to Debt Service) | Minimum | 1.25:1.00 | Quarterly"
    cov = _cov(covenant_type="min_dscr", operator=">=", threshold=1.25, evidence_quote=quote, source_page=3)
    check = verify_covenant(cov, SOURCE, PAGES)
    assert check.accepted and check.located_page == 7
    assert any("corrected to 7" in r for r in check.reasons)


def test_ratio_denominator_cannot_ground_a_fake_threshold() -> None:
    assert value_grounded("minimum 1.25:1.00", 1.25, {"ratio", "number"})
    assert not value_grounded("minimum 1.25:1.00", 1.0, {"ratio", "number"})


def test_find_quote_none_for_short_or_absent() -> None:
    assert find_quote("3.5x", SOURCE).kind == "none"
    assert find_quote("completely unrelated sentence about weather patterns", SOURCE).kind == "none"


def test_loan_terms_hallucinated_field_is_nulled() -> None:
    lt = LoanTerms(
        borrower="Altyn Agro LLP", borrower_quote="(1) Altyn Agro LLP, a limited liability partnership",
        principal=60_000_000, currency="USD",
        principal_quote="a term loan facility in an aggregate amount equal to USD 50,000,000.",
        rate_type=None, base_rate=None, rate_pct=None, rate_quote=None, maturity_date=None, maturity_quote=None,
    )
    verified, checks = verify_loan_terms(lt, SOURCE, PAGES)
    assert checks["borrower"].accepted and verified.borrower == "Altyn Agro LLP"
    assert not checks["principal"].accepted
    assert verified.principal is None and verified.currency is None
