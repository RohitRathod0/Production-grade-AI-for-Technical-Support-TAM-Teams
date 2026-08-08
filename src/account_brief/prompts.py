"""Versioned prompt templates for Task 2 account-health brief synthesis."""

from __future__ import annotations

from src.account_brief.schema import RiskFlag

PROMPT_VERSION = "brief_v1"

_SYSTEM_TEMPLATE = """You are a TAM (Technical Account Manager) assistant. You write a 3-section \
account-health brief from data already provided below — you have no other information \
about this customer and must not use outside knowledge.

Return a single JSON object with exactly these keys: executive_summary, risk_flags, \
talking_points.

- executive_summary: 3-5 sentences summarizing account health, grounded only in the facts \
below. If no risk signals were detected and no tickets fall in the 90-day window, say so \
plainly (e.g. "no significant issues") — never invent a problem that isn't in the data.
- risk_flags: a list of {"risk": "...", "quote": "..."} objects, one per genuine risk signal \
you can back with real evidence. "risk" is a short description. "quote" MUST be copied \
character-for-character (exact substring — no paraphrasing, no ellipses, no added \
punctuation) from one of the escalation notes or ticket bodies listed below. If a candidate \
risk signal below has no escalation note or ticket body that actually contains quotable text \
supporting it, DO NOT invent a quote — omit that risk_flags entry entirely (you may still \
mention the underlying fact in executive_summary without a quote).
- talking_points: 2-5 short recommended talking points for the TAM's next customer \
conversation, grounded in the facts below.

Never state a specific NPS number unless the NPS score given below is a real, non-null \
number — if NPS is unavailable, say it's unavailable and do not guess a value."""


def _format_ticket(ticket: dict) -> str:
    return (
        f"- ticket_id={ticket.get('ticket_id')} urgency={ticket.get('urgency')} "
        f"category={ticket.get('category')} created_at={ticket.get('created_at')}\n"
        f"  subject: {ticket.get('subject')}\n"
        f"  body: {ticket.get('body')}"
    )


def build_brief_prompt(account: dict, tickets: list[dict], candidates: list[RiskFlag]) -> list[dict]:
    """Chat messages for the brief-synthesis LLM call.

    `candidates` (from risk_rules.candidate_risk_signals) bounds which risk *topics*
    the LLM may surface — it must ground each one in real quotable text below or leave
    it out of risk_flags. It never invents a risk topic that isn't in `candidates`.
    """
    candidate_lines = "\n".join(f"- {c.risk} (evidence: {c.quote})" for c in candidates) or "(none detected)"
    escalation_notes = account.get("escalation_notes") or []
    notes_lines = "\n".join(f"- {note}" for note in escalation_notes) or "(none)"
    ticket_lines = "\n".join(_format_ticket(t) for t in tickets) or "(no tickets in the last 90 days)"

    nps_score = account.get("nps_score")
    nps_line = "unavailable (null)" if nps_score is None else str(nps_score)

    user = f"""Account: {account.get('company')} ({account.get('account_id')})
plan_tier: {account.get('plan_tier')}  health_status: {account.get('health_status')}  \
usage_trend: {account.get('usage_trend')}  open_tickets: {account.get('open_tickets')}  \
nps_score: {nps_line}

Rule-detected candidate risk signals (topics only — you must still find real quotable \
evidence for each below, or omit it from risk_flags):
{candidate_lines}

Escalation notes (verbatim, quotable):
{notes_lines}

Tickets in the last 90 days (verbatim bodies, quotable):
{ticket_lines}

Required JSON keys: executive_summary, risk_flags, talking_points."""

    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE},
        {"role": "user", "content": user},
    ]
