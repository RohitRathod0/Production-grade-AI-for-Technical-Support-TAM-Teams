"""Task 1 eval test cases (PRD §4) — 5 cases, each a real ticket pulled directly from
data/tickets.json by ticket_id. No invented/synthetic ticket content: `input` is always
the ticket's verbatim subject/body, resolved from the real file at import time (a typo
in a ticket_id fails immediately with KeyError, not silently).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data_loader import build_account_lookup, load_accounts, load_tickets

_tickets_by_id = {t["ticket_id"]: t for t in load_tickets()}
_account_lookup = build_account_lookup(load_accounts())


@dataclass
class TriageEvalCase:
    case_id: str
    description: str
    adversarial: bool
    ticket_id: str
    input: dict  # {"subject": str, "body": str} — verbatim from tickets.json
    checks: list[str]  # human-readable acceptance criteria (for eval_report.md); scorer.py checks these mechanically
    expect_kb_doc_path_in: list[str] | None = None  # acceptable kb_match.doc_path values, if deterministic
    expect_kb_match_none: bool = False


def _case_input(ticket_id: str, *, account_id_missing: bool | None = None) -> dict:
    """Look up a real ticket and return its {subject, body}. If account_id_missing is
    set, asserts the ticket's account_id presence/absence in accounts.json matches
    what the case claims — catches data drift instead of silently testing the wrong thing.
    """
    ticket = _tickets_by_id[ticket_id]
    if account_id_missing is not None:
        is_missing = ticket["account_id"] not in _account_lookup
        if is_missing != account_id_missing:
            raise ValueError(
                f"{ticket_id}: expected account_id missing-from-accounts.json={account_id_missing}, actual={is_missing}"
            )
    return {"subject": ticket["subject"], "body": ticket["body"]}


CASES: list[TriageEvalCase] = [
    TriageEvalCase(
        case_id="triage_01_clean_baseline",
        description="Baseline happy path: DataBridge Pro bulk-archive request, single clear issue, no dual signals.",
        adversarial=False,
        ticket_id="TKT-10000",
        input=_case_input("TKT-10000"),
        checks=[
            "TriageOutput schema validates (Pydantic)",
            "product/category/urgency are each exactly one valid enum value",
            "routing.team is derived consistently from the returned category (auditable, not LLM-chosen)",
            "routing.escalation is True iff urgency == 'P1'",
            "kb_match, if present, has a doc_path that actually exists under knowledge_base/",
            "draft_response is non-empty",
        ],
    ),
    TriageEvalCase(
        case_id="triage_02_ambiguous_dual_signal",
        description=(
            "ADVERSARIAL (PRD §4 case 3). Dual-signal ticket: CloudSync page loads are slow "
            "AND integration operations are timing out — no single category is unambiguously "
            "correct from the text alone."
        ),
        adversarial=True,
        ticket_id="TKT-10214",
        input=_case_input("TKT-10214"),
        checks=[
            "Does not crash and does not return a blank or multi-value category",
            "category is exactly one valid enum value",
            "urgency is exactly one valid enum value",
            "reasoning is non-empty (sane-reasoning check is LLM-judge scored, not a hard rule)",
        ],
    ),
    TriageEvalCase(
        case_id="triage_03_no_kb_match",
        description=(
            "AnalyticsHub webhook-delivery ticket. Verified empirically (Hours 4-6 testing) that "
            "retrieve_kb_match() returns None for this exact ticket — no KB doc in the 9-doc corpus "
            "substantively covers this specific failure."
        ),
        adversarial=False,
        ticket_id="TKT-10385",
        input=_case_input("TKT-10385"),
        checks=[
            "kb_match is null — must not hallucinate a doc when there is no good match",
            "draft_response is still produced (acknowledges the issue without inventing KB steps)",
        ],
        expect_kb_match_none=True,
    ),
    TriageEvalCase(
        case_id="triage_04_missing_account_id",
        description=(
            "ADVERSARIAL (PRD §4 case 4). account_id=ACC-1188 on this ticket has no record in "
            "accounts.json — pipeline must degrade gracefully."
        ),
        adversarial=True,
        ticket_id="TKT-10003",
        input=_case_input("TKT-10003", account_id_missing=True),
        checks=[
            "Pipeline does not crash — classify_ticket() never joins against accounts.json at all",
            "TriageOutput schema validates as normal, identical contract to any other ticket",
        ],
    ),
    TriageEvalCase(
        case_id="triage_05_known_kb_doc_match",
        description=(
            "SecureVault SSO ticket quoting AUTH_TOKEN_EXPIRED verbatim. Both the cross-product SSO "
            "troubleshooting doc and the SecureVault product doc legitimately cover this error "
            "(verified empirically in Hours 4-6 testing) — either is an acceptable top match."
        ),
        adversarial=False,
        ticket_id="TKT-10107",
        input=_case_input("TKT-10107"),
        checks=[
            "kb_match is not null",
            "kb_match.doc_path is one of the two legitimately relevant docs",
        ],
        expect_kb_doc_path_in=["troubleshooting/authentication-sso.md", "products/securevault.md"],
    ),
]
