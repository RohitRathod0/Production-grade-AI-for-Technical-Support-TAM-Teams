"""Task 2 eval test cases (PRD §4) — 5 cases, each a real account_id from
data/accounts.json (plus its real 90-day tickets from data/tickets.json). No
invented account data: account_ids are validated against the real file at import
time, so a typo (or a case that assumes an id exists/doesn't exist) fails immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data_loader import build_account_lookup, load_accounts

_account_lookup = build_account_lookup(load_accounts())


@dataclass
class BriefEvalCase:
    case_id: str
    description: str
    adversarial: bool
    account_id: str
    checks: list[str]  # human-readable acceptance criteria (for eval_report.md); scorer.py checks these mechanically
    expect_error: str | None = None  # exception class name expected from generate_brief(), if any
    expect_risk_flags_empty: bool | None = None
    expect_no_nps_number: bool = False  # scorer scans full output text for a hallucinated NPS digit
    check_determinism: bool = False  # scorer calls generate_brief() twice and diffs risk_flags


def _validated_account_id(account_id: str, *, must_exist: bool = True) -> str:
    """Confirm account_id's presence/absence in accounts.json matches what the case
    claims — catches data drift instead of silently testing the wrong scenario.
    """
    exists = account_id in _account_lookup
    if must_exist and not exists:
        raise ValueError(f"eval case references unknown account_id={account_id!r}")
    if not must_exist and exists:
        raise ValueError(f"eval case expected account_id={account_id!r} to be absent from accounts.json, but it exists")
    return account_id


CASES: list[BriefEvalCase] = [
    BriefEvalCase(
        case_id="brief_01_at_risk_with_escalation_and_p1",
        description="Gavin Belson Co — At Risk, 2 escalation_notes, 1 ticket in the 90-day window, p1_tickets_last_30d=2.",
        adversarial=False,
        account_id=_validated_account_id("ACC-7397"),
        checks=[
            "AccountBrief schema validates (Pydantic)",
            "risk_flags is non-empty",
            "every risk_flags[i].quote is an exact substring of an escalation_notes entry or a ticket body in the 90-day window",
        ],
    ),
    BriefEvalCase(
        case_id="brief_02_healthy_no_tickets",
        description="Aviato Systems — Healthy, Increasing usage, zero tickets in the 90-day window, real nps_score=6.",
        adversarial=False,
        account_id=_validated_account_id("ACC-5524"),
        checks=[
            "AccountBrief schema validates",
            "risk_flags is empty — no escalation_notes or in-window tickets exist to quote, so nothing survives verification "
            "(verified empirically: the LLM does attempt an NPS-based flag here, and quote_verification correctly drops it)",
            "executive_summary reflects a healthy/no-significant-issues account, not a fabricated risk",
        ],
        expect_risk_flags_empty=True,
    ),
    BriefEvalCase(
        case_id="brief_03_null_nps",
        description=(
            "ADVERSARIAL (PRD §4 case 3). Vertex Solutions — At Risk, nps_score is null, 3 escalation_notes, "
            "zero tickets in the 90-day window."
        ),
        adversarial=True,
        account_id=_validated_account_id("ACC-8113"),
        checks=[
            "AccountBrief schema validates",
            "no hallucinated NPS number appears anywhere in the output (regex scan for a digit near \"NPS\")",
            "risk_flags quotes are verbatim-verified against escalation_notes",
        ],
        expect_no_nps_number=True,
    ),
    BriefEvalCase(
        case_id="brief_04_determinism",
        description=(
            "Same account (ACC-7397) called twice — checks run-to-run stability of the safety-critical, "
            "quote-verified part of the output (PRD §3 step 5)."
        ),
        adversarial=False,
        account_id=_validated_account_id("ACC-7397"),
        checks=[
            "generate_brief() called twice on the same account_id",
            "risk_flags[i].quote values are identical across both calls (order-independent) — this is the "
            "verbatim-grounded, quote-verified field PRD §3 requires to be stable/reproducible",
            "risk_flags[i].risk (the free-text label, e.g. 'Consecutive P1 tickets' vs 'Consecutive P1 incidents') "
            "is NOT required to match word-for-word — confirmed empirically on live output that it can reword "
            "between identical calls even at temperature=0 + a fixed seed, same class of drift as "
            "executive_summary/talking_points prose",
        ],
        check_determinism=True,
    ),
    BriefEvalCase(
        case_id="brief_05_missing_account",
        description="ADVERSARIAL (PRD §4 case 5). account_id does not exist anywhere in accounts.json.",
        adversarial=True,
        account_id=_validated_account_id("ACC-DOES-NOT-EXIST-ANYWHERE", must_exist=False),
        checks=[
            "generate_brief() raises AccountNotFoundError — a clean, typed error, not an unhandled crash/traceback",
        ],
        expect_error="AccountNotFoundError",
    ),
]
