import pytest
from pydantic import ValidationError

from src.extractor import strict_json_schema
from src.schemas import Covenant, ExtractionPayload, LoanTerms

GOOD = dict(
    covenant_type="max_leverage", operator="<=", threshold=3.5, unit="ratio", test_frequency="quarterly",
    source_page=4, evidence_quote="The Leverage Ratio shall not exceed 3.50x.",
)


def test_valid_covenant() -> None:
    c = Covenant(**GOOD)
    assert c.operator.is_upper_bound and c.covenant_type.direction == "max"


@pytest.mark.parametrize(
    "update,msg",
    [
        ({"operator": ">="}, "maximum covenant"),
        ({"unit": "currency"}, "must use unit"),
        ({"threshold": -1.0}, "positive"),
        ({"threshold": 350.0}, "implausible"),
        ({"source_page": 0}, ">= 1"),
        ({"evidence_quote": "3.5x"}, "at least 8"),
        ({"covenant_type": "min_dscr"}, "minimum covenant"),
        ({"covenant_type": "max_capex", "unit": "currency", "threshold": 12.5}, "full currency units"),
    ],
)
def test_covenant_validators(update: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        Covenant(**(GOOD | update))


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Covenant(**(GOOD | {"confidence": 0.9}))


def _loan(**kw):
    base = {k: None for k in LoanTerms.model_fields}
    return LoanTerms(**(base | kw))


def test_loan_terms_require_quotes_and_iso_dates() -> None:
    with pytest.raises(ValidationError, match="principal_quote"):
        _loan(principal=1e6, currency="USD")
    with pytest.raises(ValidationError, match="ISO"):
        _loan(maturity_date="15 March 2028", maturity_quote="being 15 March 2028")
    lt = _loan(currency="usd", maturity_date="2028-03-15", maturity_quote="being 15 March 2028")
    assert lt.currency == "USD"


def test_strict_json_schema_for_structured_outputs() -> None:
    schema = strict_json_schema(ExtractionPayload)

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    objs = list(objects(schema))
    assert len(objs) >= 3  # payload, Covenant, LoanTerms
    for o in objs:
        assert o["additionalProperties"] is False
        assert set(o["required"]) == set(o["properties"])
    assert "default" not in str(schema)


def test_schema_accepted_by_openai_sdk_converter() -> None:
    converter = pytest.importorskip("openai.lib._pydantic")
    schema = converter.to_strict_json_schema(ExtractionPayload)
    assert schema["additionalProperties"] is False
