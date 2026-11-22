import math

import numpy as np
import pandas as pd
import pytest

from src.auditor import (
    audit_contract,
    compute_actuals,
    headroom_pct,
    is_compliant,
    is_test_quarter,
    match_borrower,
    safe_ratio,
)
from src.schemas import (
    Covenant,
    ExtractionResult,
    FindingStatus,
    Frequency,
    GuardrailCheck,
    GuardrailStatus,
    LoanTerms,
    Operator,
    VerifiedCovenant,
)

# --------------------------------------------------------------------------------------
# Operator semantics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "actual,op,threshold,ok",
    [
        (3.50, Operator.LE, 3.50, True),   # equality complies with "shall not exceed"
        (3.50, Operator.LT, 3.50, False),  # ...but breaches "shall be less than"
        (3.49, Operator.LT, 3.50, True),
        (3.51, Operator.LE, 3.50, False),
        (1.25, Operator.GE, 1.25, True),
        (1.25, Operator.GT, 1.25, False),
        (1.26, Operator.GT, 1.25, True),
        (1.24, Operator.GE, 1.25, False),
        (3.5 + 1e-12, Operator.LE, 3.5, True),  # float noise is treated as equality
        (math.inf, Operator.LE, 3.5, False),   # non-positive EBITDA -> infinite leverage -> breach
        (math.inf, Operator.GE, 2.0, True),    # zero interest -> infinite cover -> compliant
    ],
)
def test_operator_semantics(actual: float, op: Operator, threshold: float, ok: bool) -> None:
    assert is_compliant(actual, op, threshold) is ok


def test_headroom() -> None:
    assert headroom_pct(3.0, Operator.LE, 4.0) == pytest.approx(25.0)
    assert headroom_pct(5.0, Operator.LE, 4.0) == pytest.approx(-25.0)
    assert headroom_pct(1.5, Operator.GE, 1.2) == pytest.approx(25.0)
    assert headroom_pct(math.inf, Operator.LE, 4.0) is None


def test_safe_ratio() -> None:
    out = safe_ratio(np.array([10.0, 10.0, -1.0, np.nan]), np.array([4.0, 0.0, 0.0, 1.0]))
    assert out[0] == pytest.approx(2.5) and out[1] == math.inf and out[2] == -math.inf and math.isnan(out[3])


def test_test_quarters() -> None:
    assert [q for q in range(1, 5) if is_test_quarter(q, Frequency.QUARTERLY)] == [1, 2, 3, 4]
    assert [q for q in range(1, 5) if is_test_quarter(q, Frequency.SEMI_ANNUAL)] == [2, 4]
    assert [q for q in range(1, 5) if is_test_quarter(q, Frequency.ANNUAL)] == [4]


# --------------------------------------------------------------------------------------
# Ledger maths
# --------------------------------------------------------------------------------------
def _ledger(n: int = 8, **overrides) -> pd.DataFrame:
    t = np.arange(n)
    data = {
        "borrower_id": "B1", "borrower_name": "Altyn Agro LLP", "currency": "USD",
        "period": [f"{2022 + i // 4}Q{i % 4 + 1}" for i in t], "fiscal_year": 2022 + t // 4, "quarter": t % 4 + 1,
        "ebitda": np.full(n, 10.0), "total_debt": np.full(n, 120.0), "interest_expense": np.full(n, 2.0),
        "debt_service": np.full(n, 5.0), "current_assets": np.full(n, 30.0), "current_liabilities": np.full(n, 20.0),
        "capex": np.full(n, 3.0), "net_worth": np.full(n, 100.0),
    }
    df = pd.DataFrame(data)
    for col, values in overrides.items():
        df[col] = values
    return df


def test_compute_actuals_hand_computed() -> None:
    act = compute_actuals(_ledger())
    row = act[act["period"] == "2022Q4"].iloc[0]
    assert row["ebitda_ltm"] == pytest.approx(40.0)
    assert row["leverage"] == pytest.approx(120 / 40)          # 3.0x
    assert row["interest_cover"] == pytest.approx(40 / 8)      # 5.0x
    assert row["dscr"] == pytest.approx(40 / 20)               # 2.0x
    assert row["current_ratio"] == pytest.approx(1.5)
    assert row["capex_fy"] == pytest.approx(12.0)
    assert math.isnan(act[act["period"] == "2022Q3"].iloc[0]["leverage"])  # < 4 quarters of history
    assert math.isnan(act[act["period"] == "2023Q2"].iloc[0]["capex_fy"])   # FY capex only at Q4


def test_missing_quarter_row_is_not_bridged() -> None:
    led = _ledger().drop(index=5)  # 2023Q2 missing entirely
    act = compute_actuals(led)
    assert len(act) == 8
    assert math.isnan(act[act["period"] == "2023Q3"].iloc[0]["ebitda_ltm"])


def _extraction(cov: Covenant, borrower: str = "Altyn Agro LLP") -> ExtractionResult:
    check = GuardrailCheck(status=GuardrailStatus.VERIFIED, located_page=7)
    lt = LoanTerms(borrower=borrower, borrower_quote=f"(1) {borrower}, a company", principal=None, currency=None,
                   principal_quote=None, rate_type=None, base_rate=None, rate_pct=None, rate_quote=None,
                   maturity_date=None, maturity_quote=None)
    return ExtractionResult(contract_id="CA1", extractor="test", loan_terms=lt, covenants=[VerifiedCovenant(covenant=cov, check=check)])


LEV = Covenant(covenant_type="max_leverage", operator="<=", threshold=3.0, unit="ratio", test_frequency="quarterly",
               source_page=7, evidence_quote="The Leverage Ratio shall not exceed 3.00x.")


def test_audit_statuses_and_citations() -> None:
    debt = np.full(8, 120.0)
    debt[6] = 130.0  # 2023Q3 leverage = 3.25x > 3.00x
    led = _ledger(total_debt=debt)
    findings = audit_contract(_extraction(LEV), compute_actuals(led), led)
    by_period = {f.period: f for f in findings}
    assert set(by_period) == {"2022Q4", "2023Q1", "2023Q2", "2023Q3", "2023Q4"}  # warm-up skipped
    assert by_period["2022Q4"].status == FindingStatus.COMPLIANT   # exactly 3.00x with '<='
    assert by_period["2023Q3"].status == FindingStatus.BREACH
    assert by_period["2023Q3"].actual_value == pytest.approx(3.25)
    assert by_period["2023Q3"].source_page == 7 and "3.00x" in by_period["2023Q3"].evidence_quote

    strict = LEV.model_copy(update={"operator": Operator.LT})
    f_strict = {f.period: f for f in audit_contract(_extraction(strict), compute_actuals(led), led)}
    assert f_strict["2022Q4"].status == FindingStatus.BREACH  # 3.00x is not "< 3.00x"


def test_insufficient_data_and_unknown_borrower() -> None:
    ebitda = np.full(8, 10.0)
    ebitda[5] = np.nan
    led = _ledger(ebitda=ebitda)
    findings = {f.period: f for f in audit_contract(_extraction(LEV), compute_actuals(led), led)}
    for p in ("2023Q2", "2023Q3", "2023Q4"):  # every LTM window containing 2023Q2
        assert findings[p].status == FindingStatus.INSUFFICIENT_DATA and "ebitda@2023Q2" in findings[p].note
    assert findings["2023Q1"].status == FindingStatus.COMPLIANT

    ghost = audit_contract(_extraction(LEV, borrower="Unknown Holdings Ltd"), compute_actuals(led), led)
    assert len(ghost) == 1 and ghost[0].status == FindingStatus.INSUFFICIENT_DATA and ghost[0].period == "N/A"


def test_semi_annual_and_capex_periods() -> None:
    led = _ledger()
    semi = LEV.model_copy(update={"test_frequency": Frequency.SEMI_ANNUAL})
    assert [f.period for f in audit_contract(_extraction(semi), compute_actuals(led), led)] == ["2022Q4", "2023Q2", "2023Q4"]
    led = _ledger(capex=np.full(8, 3_000.0))
    capex = Covenant(covenant_type="max_capex", operator="<=", threshold=11_000, unit="currency", test_frequency="quarterly",
                     source_page=7, evidence_quote="Capital Expenditure shall not exceed USD 11,000 in any Financial Year.")
    f = audit_contract(_extraction(capex), compute_actuals(led), led)
    assert [x.period for x in f] == ["2022Q4", "2023Q4"]  # FY test regardless of the stated frequency
    assert all(x.status == FindingStatus.BREACH for x in f)  # 4 x 3,000 = 12,000 > 11,000
    assert f[0].actual_value == pytest.approx(12_000)


def test_borrower_matching() -> None:
    led = _ledger()
    assert match_borrower("ALTYN AGRO LLP", led)[0] == "B1"
    assert match_borrower("Altyn Agro", led)[0] == "B1"          # legal form ignored
    assert match_borrower("Altyn Agr0 LLP", led)[0] == "B1"      # fuzzy
    assert match_borrower("Northwind Mining plc", led)[0] is None
