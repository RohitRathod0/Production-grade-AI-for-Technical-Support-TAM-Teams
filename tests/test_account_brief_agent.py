"""Task 2 agent tests against real accounts.json/tickets.json — no mocked data.
Makes live LLM calls at temperature=0; requires a configured GROQ_API_KEY in .env.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.account_brief.agent import AccountNotFoundError, generate_brief
from src.data_loader import build_account_lookup, get_account_tickets, load_accounts, load_tickets

# Real ids drawn from data/, verified against the shipped dataset:
ACCOUNT_AT_RISK_WITH_ESCALATION = "ACC-7397"  # At Risk, 2 escalation notes, 1 ticket in the 90d window
ACCOUNT_HEALTHY_NO_TICKETS = "ACC-5524"  # Healthy, zero tickets in the 90d window, real nps_score=6
ACCOUNT_NULL_NPS = "ACC-8113"  # At Risk, nps_score is null, 3 escalation notes, zero tickets in window


def _quotable_sources(account_id: str) -> list[str]:
    lookup = build_account_lookup(load_accounts())
    account = lookup[account_id]
    tickets = get_account_tickets(account_id, load_tickets())
    sources = [str(note) for note in (account.get("escalation_notes") or [])]
    sources += [t["body"] for t in tickets]
    return sources


def test_at_risk_account_risk_flags_are_verbatim_quotes():
    brief = generate_brief(ACCOUNT_AT_RISK_WITH_ESCALATION)
    assert brief.risk_flags, "expected at least one risk flag for an At Risk account with escalation notes"
    sources = _quotable_sources(ACCOUNT_AT_RISK_WITH_ESCALATION)
    for flag in brief.risk_flags:
        assert any(flag.quote in source for source in sources), f"quote not verbatim in source data: {flag.quote!r}"


def test_healthy_account_with_no_tickets_reports_no_significant_issues():
    brief = generate_brief(ACCOUNT_HEALTHY_NO_TICKETS)
    # No escalation notes and no tickets in window means nothing is quotable, so even
    # though risk_rules.py may flag a candidate (e.g. borderline NPS), verification
    # must strip anything the LLM can't ground — final risk_flags must be empty.
    assert brief.risk_flags == []
    assert brief.executive_summary


def test_null_nps_account_never_hallucinates_a_number():
    brief = generate_brief(ACCOUNT_NULL_NPS)
    full_text = " ".join(
        [brief.executive_summary, *brief.talking_points, *(f"{f.risk} {f.quote}" for f in brief.risk_flags)]
    )
    assert not re.search(r"NPS\D{0,20}?\d", full_text, re.I), f"possible hallucinated NPS number in: {full_text!r}"


def test_missing_account_raises_clean_error_not_crash():
    try:
        generate_brief("ACC-DOES-NOT-EXIST-ANYWHERE")
        assert False, "expected AccountNotFoundError"
    except AccountNotFoundError as exc:
        assert exc.account_id == "ACC-DOES-NOT-EXIST-ANYWHERE"


def test_determinism_same_account_twice():
    """Determinism (PRD §3 step 5): temperature=0 + a fixed seed are both set in
    agent.py, but empirically Groq's free-text prose (executive_summary,
    talking_points) still drifts word-for-word between identical calls — a real,
    observed provider-level limitation, not something canonicalization can paper
    over. What canonicalization *does* guarantee, and what's asserted here, is that
    the safety-critical, quote-verified part — risk_flags, the only field with a
    verbatim-grounding requirement — is byte-identical across runs.
    """
    brief1 = generate_brief(ACCOUNT_AT_RISK_WITH_ESCALATION)
    brief2 = generate_brief(ACCOUNT_AT_RISK_WITH_ESCALATION)
    assert brief1.risk_flags == brief2.risk_flags
