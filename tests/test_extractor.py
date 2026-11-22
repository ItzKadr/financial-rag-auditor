"""Extractor tests - fully offline. The OpenAI client is replaced by a duck-typed fake."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from src.evaluation import extraction_metrics, loan_terms_metrics, recall_by_phrasing
from src.extractor import (
    LLMExtractor,
    OfflineExtractor,
    StructuredLLMClient,
    candidate_units,
    detect_operator,
    match_covenant_unit,
)
from src.providers import ProviderConfig
from src.schemas import CovenantType, ExtractionPayload, Frequency, Operator


class FakeCompletions:
    """Mimics ``client.chat.completions`` (create + parse) with scripted outputs."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def _next(self, kwargs: dict) -> str:
        rest = copy.deepcopy({k: v for k, v in kwargs.items() if k != "response_format"})
        self.calls.append(rest | {"response_format": kwargs.get("response_format")})
        return self.outputs.pop(0)

    def create(self, **kwargs):
        content = self._next(kwargs)
        msg = SimpleNamespace(content=content, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    def parse(self, **kwargs):
        content = self._next(kwargs)
        # like the real SDK: validate against the Pydantic response_format (raises ValidationError)
        parsed = kwargs["response_format"].model_validate_json(content)
        msg = SimpleNamespace(content=content, refusal=None, parsed=parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def fake_client(outputs: list[str]) -> tuple[SimpleNamespace, FakeCompletions]:
    comp = FakeCompletions(outputs)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


def payload_json(covenants: list[dict], loan_terms: dict | None = None) -> str:
    return json.dumps({"loan_terms": loan_terms, "covenants": covenants})


LEV = {"covenant_type": "max_leverage", "operator": "<=", "threshold": 3.5, "unit": "ratio",
       "test_frequency": "quarterly", "source_page": 5, "evidence_quote": "The Leverage Ratio shall not exceed 3.50x."}
MESSAGES = [{"role": "system", "content": "extract"}, {"role": "user", "content": "CONTEXT: ..."}]


# --------------------------------------------------------------------------------------
# Structured-output client
# --------------------------------------------------------------------------------------
def test_parse_mode_returns_validated_model() -> None:
    client, comp = fake_client([payload_json([LEV])])
    llm = StructuredLLMClient(client, model="gpt-test", response_mode="parse")
    res = llm.complete(MESSAGES, ExtractionPayload)
    assert res.attempts == 1 and not res.errors
    assert isinstance(res.parsed, ExtractionPayload)
    assert res.parsed.covenants[0].operator == Operator.LE
    call = comp.calls[0]
    assert call["response_format"] is ExtractionPayload and call["temperature"] == 0.0 and call["model"] == "gpt-test"


def test_retry_feeds_back_validation_errors() -> None:
    bad = dict(LEV, operator=">=")  # inconsistent with a maximum covenant -> model_validator error
    client, comp = fake_client([payload_json([bad]), payload_json([LEV])])
    llm = StructuredLLMClient(client, model="gpt-test", response_mode="parse", max_attempts=3)
    res = llm.complete(MESSAGES, ExtractionPayload)
    assert res.parsed is not None and res.attempts == 2
    assert len(res.errors) == 1 and "maximum covenant" in res.errors[0]
    second_messages = comp.calls[1]["messages"]
    assert len(second_messages) == len(MESSAGES) + 1
    assert second_messages[-1]["role"] == "user" and "failed schema validation" in second_messages[-1]["content"]
    assert "covenants.0" in second_messages[-1]["content"]


def test_json_object_mode_handles_fences_and_bad_json() -> None:
    fenced = "```json\n" + payload_json([LEV]) + "\n```"
    client, comp = fake_client(["{not json", fenced])
    llm = StructuredLLMClient(client, model="deepseek-chat", response_mode="json_object")
    res = llm.complete(MESSAGES, ExtractionPayload)
    assert res.parsed is not None and res.attempts == 2
    assert comp.calls[0]["response_format"] == {"type": "json_object"}
    # the invalid raw answer is echoed back as an assistant turn before the correction request
    assert comp.calls[1]["messages"][-2] == {"role": "assistant", "content": "{not json"}


def test_json_schema_mode_sends_strict_schema() -> None:
    client, comp = fake_client([payload_json([LEV])])
    StructuredLLMClient(client, model="m", response_mode="json_schema").complete(MESSAGES, ExtractionPayload)
    rf = comp.calls[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["additionalProperties"] is False


def test_gives_up_after_max_attempts() -> None:
    bad = payload_json([dict(LEV, threshold=-1)])
    client, comp = fake_client([bad, bad])
    res = StructuredLLMClient(client, model="m", response_mode="parse", max_attempts=2).complete(MESSAGES, ExtractionPayload)
    assert res.parsed is None and res.attempts == 2 and len(res.errors) == 2 and len(comp.calls) == 2


# --------------------------------------------------------------------------------------
# LLM extractor end-to-end with guardrails (mocked client, real retrieval)
# --------------------------------------------------------------------------------------
def test_llm_extractor_guardrails_reject_hallucination(corpus, processed) -> None:
    cid = next(c.contract_id for c in corpus.contracts
               if any(x["covenant_type"] == "max_leverage" for x in c.ground_truth["covenants"]))
    gt = corpus.ground_truth[cid]
    true_cov = next(x for x in gt["covenants"] if x["covenant_type"] == "max_leverage")
    genuine = {"covenant_type": "max_leverage", "operator": true_cov["operator"], "threshold": true_cov["threshold"],
               "unit": "ratio", "test_frequency": true_cov["test_frequency"], "source_page": true_cov["page"],
               "evidence_quote": true_cov["evidence"]}
    hallucinated = {"covenant_type": "min_net_worth", "operator": ">=", "threshold": 77_000_000, "unit": "currency",
                    "test_frequency": "quarterly", "source_page": 3,
                    "evidence_quote": "The Borrower shall maintain a Tangible Net Worth of at least USD 77,000,000."}
    client, comp = fake_client([payload_json([genuine, hallucinated])])
    provider = ProviderConfig(name="openai", api_key="test", base_url=None, model="gpt-test", response_mode="parse")
    extractor = LLMExtractor(provider, client=client)
    p = processed[cid]
    result = extractor.extract(cid, p.index, p.page_texts)
    assert result.extractor == "openai:gpt-test"
    status = {vc.covenant.covenant_type: vc.accepted for vc in result.covenants}
    assert status[CovenantType.MAX_LEVERAGE] is True
    assert status[CovenantType.MIN_NET_WORTH] is False
    # the prompt contained the retrieved context with chunk ids and pages
    user_msg = comp.calls[0]["messages"][1]["content"]
    assert "[chunk" in user_msg and "page" in user_msg


def test_llm_extractor_survives_provider_errors(processed, first_id) -> None:
    class Boom:
        def parse(self, **kwargs):
            raise ConnectionError("network down")

    client = SimpleNamespace(chat=SimpleNamespace(completions=Boom()))
    provider = ProviderConfig(name="openai", api_key="x", base_url=None, model="m", response_mode="parse")
    p = processed[first_id]
    result = LLMExtractor(provider, client=client).extract(first_id, p.index, p.page_texts)
    assert result.covenants == [] and result.errors and "network down" in result.errors[0]


# --------------------------------------------------------------------------------------
# Offline rules
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,op",
    [
        ("The Leverage Ratio shall not exceed 3.50x.", Operator.LE),
        ('The ratio (the "Leverage Ratio") shall not, on any Test Date, be greater than 3.50x.', Operator.LE),
        ("The Leverage Ratio shall be less than 3.50x on each Test Date.", Operator.LT),
        ("The DSCR shall not, on any Test Date, fall below 1.35 times.", Operator.GE),
        ("Interest Cover in respect of each Relevant Period shall exceed 2.50x.", Operator.GT),
        ("The DSCR for each Relevant Period shall be greater than 1.20x.", Operator.GT),
        ("Capital Expenditure shall be limited to a maximum of USD 5,000,000.", Operator.LE),
        ("Tangible Net Worth is not, at any time, less than USD 9,000,000.", Operator.GE),
    ],
)
def test_detect_operator(text: str, op: Operator) -> None:
    assert detect_operator(text)[0] == op


def test_rules_ignore_distractors() -> None:
    q = Frequency.QUARTERLY
    # a statement of fact (no obligation), a ratchet row, and a cure amount
    assert match_covenant_unit("As at the date of this Agreement, the Leverage Ratio is 2.11x.", False, CovenantType.MAX_LEVERAGE, q) is None
    assert match_covenant_unit("|Greater than 3.00x|3.75|", True, CovenantType.MAX_LEVERAGE, q) is None
    assert match_covenant_unit(
        "If the Leverage Ratio is breached the Borrower may cure it by equity not exceeding USD 3,000,000.",
        False, CovenantType.MAX_LEVERAGE, q,
    ) is None
    hit = match_covenant_unit("|Leverage Ratio (Total Debt to EBITDA)|Maximum|3.50x|Quarterly|", True, CovenantType.MAX_LEVERAGE, q)
    assert hit == (Operator.LE, 3.5, Frequency.QUARTERLY)
    capex = match_covenant_unit("The Borrower shall not incur Capital Expenditure in excess of USD 12.5 million in aggregate in any Financial Year.",
                                False, CovenantType.MAX_CAPEX, q)
    assert capex == (Operator.LE, 12_500_000, Frequency.ANNUAL)


def test_candidate_units_split_rows_and_sentences() -> None:
    text = "Intro sentence one. Intro two.\n|A|B|\n|---|---|\n|Leverage|3.50x|\nTail sentence."
    units = candidate_units(text)
    assert units == [("Intro sentence one.", False), ("Intro two.", False), ("|A|B|", True),
                     ("|Leverage|3.50x|", True), ("Tail sentence.", False)]


def test_offline_extractor_on_fixture_corpus(corpus, processed) -> None:
    extractor = OfflineExtractor()
    results = [extractor.extract(cid, p.index, p.page_texts) for cid, p in processed.items()]
    gt = corpus.ground_truth
    m = extraction_metrics(results, gt)
    assert m["hallucination_rate"] == 0.0
    assert m["field_precision"] == 1.0
    by_phrasing = recall_by_phrasing(results, gt)
    assert all(v == 1.0 for k, v in by_phrasing.items() if k.startswith("recall_"))
    assert m["covenant_recall"] == 1.0  # fixture has no hold-out phrasing
    assert loan_terms_metrics(results, gt)["loan_terms_acc"] == 1.0
