"""generate_brief(account_id) -> AccountBrief — Task 2 pipeline: deterministic risk
pre-pass (risk_rules.py) -> LLM synthesis (temp=0) -> quote verification -> canonicalization.

Quote verification is the trust boundary: every risk_flags[i].quote the LLM returns
must be an exact substring of a real escalation_notes entry or a real ticket body in
the account's 90-day window. A flag whose quote fails that check is dropped and
logged (never silently kept, never silently discarded without a trace) — see
_verify_quotes().
"""

from __future__ import annotations

import json
import logging

from src import llm_client
from src.account_brief.prompts import build_brief_prompt
from src.account_brief.risk_rules import candidate_risk_signals
from src.account_brief.schema import AccountBrief, RiskFlag
from src.data_loader import (
    build_account_lookup,
    get_account,
    get_account_tickets,
    load_accounts,
    load_tickets,
)

logger = logging.getLogger(__name__)

_MAX_SYNTHESIS_ATTEMPTS = 2
_SYNTHESIS_SEED = 0  # fixed seed (PRD §3 step 5) — best-effort determinism on top of temperature=0


class AccountNotFoundError(Exception):
    """Raised by generate_brief() when account_id has no record in accounts.json —
    callers (API layer, CLI) catch this for a clean error response, never a raw crash.
    """

    def __init__(self, account_id: str):
        super().__init__(f"No account found for account_id={account_id!r}")
        self.account_id = account_id


def _quotable_sources(account: dict, tickets: list[dict]) -> list[str]:
    """Every verbatim string a risk_flags quote is allowed to be a substring of:
    escalation_notes entries and ticket bodies in the account's 90-day window.
    """
    sources = [str(note) for note in (account.get("escalation_notes") or [])]
    sources += [t["body"] for t in tickets if t.get("body")]
    return sources


def _verify_quotes(risk_flags: list[RiskFlag], sources: list[str]) -> list[RiskFlag]:
    """Keep only risk_flags whose quote is an exact substring of some source string.
    Every drop is logged with the offending risk/quote — never a silent discard.
    """
    verified = []
    for flag in risk_flags:
        if any(flag.quote in source for source in sources):
            verified.append(flag)
        else:
            logger.warning(
                "Dropping unverifiable risk flag (quote not found in escalation_notes "
                "or ticket bodies): risk=%r quote=%r",
                flag.risk,
                flag.quote,
            )
    return verified


def _canonicalize(brief: AccountBrief) -> AccountBrief:
    """Strip whitespace and sort risk_flags for run-to-run determinism. RiskFlag (see
    schema.py) carries no ticket_id field, so flags are sorted by (risk, quote) instead
    — a different key than PRD's suggested "sort by ticket_id" example, but the same
    determinism goal: a stable, content-derived order regardless of LLM output order.
    """
    risk_flags = sorted(
        (RiskFlag(risk=f.risk.strip(), quote=f.quote.strip()) for f in brief.risk_flags),
        key=lambda f: (f.risk, f.quote),
    )
    return AccountBrief(
        executive_summary=brief.executive_summary.strip(),
        risk_flags=risk_flags,
        talking_points=[p.strip() for p in brief.talking_points],
    )


def _synthesize(account: dict, tickets: list[dict], candidates: list[RiskFlag]) -> AccountBrief:
    """Call the LLM at temperature=0 and validate its JSON against AccountBrief,
    retrying once on a malformed/invalid response before letting the error propagate.
    """
    messages = build_brief_prompt(account, tickets, candidates)
    last_error: Exception | None = None
    for _ in range(_MAX_SYNTHESIS_ATTEMPTS):
        raw = llm_client.chat(messages, json_mode=True, temperature=0, seed=_SYNTHESIS_SEED)
        try:
            return AccountBrief.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"LLM brief synthesis failed after {_MAX_SYNTHESIS_ATTEMPTS} attempts: {last_error}")


def generate_brief(account_id: str) -> AccountBrief:
    """Run the Task 2 pipeline for one account_id: look up the account, pull its
    90-day tickets, compute grounded risk candidates, synthesize the brief, verify
    every risk_flags quote against real source text, then canonicalize for
    determinism. Raises AccountNotFoundError (never an unhandled crash) if account_id
    has no record in accounts.json.
    """
    lookup = build_account_lookup(load_accounts())
    account = get_account(account_id, lookup)
    if account is None:
        raise AccountNotFoundError(account_id)

    tickets = get_account_tickets(account_id, load_tickets())
    candidates = candidate_risk_signals(account, tickets)

    brief = _synthesize(account, tickets, candidates)
    verified_flags = _verify_quotes(brief.risk_flags, _quotable_sources(account, tickets))
    brief = AccountBrief(
        executive_summary=brief.executive_summary,
        risk_flags=verified_flags,
        talking_points=brief.talking_points,
    )
    return _canonicalize(brief)
