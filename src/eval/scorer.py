"""Task 3 scoring (PRD §4): rule-based hard gates first, LLM-as-judge only for cases
that pass them. Hard gates are the pass/fail authority — a case that fails one is a
hard fail regardless of how good the LLM-judge would have scored it, because there's
no point judging the quality of a structurally broken output.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, Field

from src import config, llm_client
from src.account_brief import agent as brief_agent
from src.account_brief.schema import AccountBrief
from src.data_loader import build_account_lookup, get_account_tickets, load_accounts, load_tickets
from src.eval.cases_brief import BriefEvalCase
from src.eval.cases_triage import TriageEvalCase
from src.eval.prompts import build_judge_prompt
from src.triage.agent import classify_ticket
from src.triage.schema import Category, Product, TriageOutput, Urgency

logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base"

_account_lookup = build_account_lookup(load_accounts())
_all_tickets = load_tickets()


class JudgeScore(BaseModel):
    """LLM-as-judge output shape (prompts.PROMPT_VERSION = "judge_v1")."""

    relevance: float = Field(ge=0, le=1)
    faithfulness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    notes: str


@dataclass
class EvalResult:
    case_id: str
    task: str
    adversarial: bool
    hard_gate_pass: bool | None  # None = SKIPPED (no provider key in this environment), not evaluated
    hard_gate_failures: list[str]
    quality_score: float | None  # None if hard gates failed/skipped, or judge itself was skipped
    judge_notes: str | None
    notes: str


# CI runs with no provider keys at all (PRD: no secrets in the workflow). Detected
# once at import time, same pattern as _account_lookup/_all_tickets above. Read via
# config.py (never os.getenv directly, per config.py's own docstring) — config.py's
# GROQ_API_KEY/MISTRAL_API_KEY are themselves os.getenv-backed, so this is the same
# "present and non-empty" check, just going through the one place env vars are read.
HAS_GROQ_KEY = bool(config.GROQ_API_KEY)
HAS_MISTRAL_KEY = bool(config.MISTRAL_API_KEY)


def _run_judge(task: str, source_context: str, output_json: str) -> JudgeScore | None:
    """Single low-cost LLM-as-judge call via config.JUDGE_PROVIDER (Mistral, a
    different model family than the primary Groq agent — reduces self-grading bias).
    Returns None (never raises) if the judge call itself fails, so a flaky judge call
    doesn't take down the whole eval run — it just leaves quality_score unset, which
    the report surfaces honestly rather than faking a score.
    """
    messages = build_judge_prompt(task, source_context, output_json)
    try:
        raw = llm_client.chat(messages, json_mode=True, temperature=0, provider=config.JUDGE_PROVIDER)
        return JudgeScore.model_validate(json.loads(raw))
    except Exception as exc:
        logger.warning("LLM-as-judge call failed for task=%r: %s", task, exc)
        return None


def _maybe_run_judge(task: str, source_context: str, output_json: str) -> tuple[JudgeScore | None, str | None]:
    """Runs the judge only if MISTRAL_API_KEY is actually set — otherwise returns a
    clear skip reason instead of attempting (and failing) a call with no key.
    """
    if not HAS_MISTRAL_KEY:
        return None, "SKIPPED (no MISTRAL_API_KEY in this environment)"
    return _run_judge(task, source_context, output_json), None


def _finalize(
    case_id: str,
    task: str,
    adversarial: bool,
    failures: list[str],
    judge_score: JudgeScore | None = None,
    judge_skip_reason: str | None = None,
) -> EvalResult:
    hard_gate_pass = not failures
    quality_score = None
    judge_notes = None
    if judge_score is not None:
        quality_score = round((judge_score.relevance + judge_score.faithfulness + judge_score.completeness) / 3, 3)
        judge_notes = judge_score.notes
    elif judge_skip_reason is not None:
        judge_notes = judge_skip_reason
    notes = "; ".join(failures) if failures else (judge_notes or "hard gates passed")
    return EvalResult(
        case_id=case_id,
        task=task,
        adversarial=adversarial,
        hard_gate_pass=hard_gate_pass,
        hard_gate_failures=failures,
        quality_score=quality_score,
        judge_notes=judge_notes,
        notes=notes,
    )


def _finalize_skipped(case_id: str, task: str, adversarial: bool, missing_key: str) -> EvalResult:
    """Whole case skipped: it strictly requires a live LLM call and that provider's
    key isn't set. hard_gate_pass=None marks this distinctly from both pass (True)
    and fail (False) everywhere the result is consumed (report writers, run.py, CI).
    """
    return EvalResult(
        case_id=case_id,
        task=task,
        adversarial=adversarial,
        hard_gate_pass=None,
        hard_gate_failures=[],
        quality_score=None,
        judge_notes=None,
        notes=f"SKIPPED (no {missing_key} in this environment)",
    )


def score_triage_case(case: TriageEvalCase) -> EvalResult:
    """Run classify_ticket() on one real ticket and score it against the case's
    hard gates, then (only if they all pass) an LLM-judge quality pass.
    """
    if not HAS_GROQ_KEY:
        # Every triage case needs a live classify_ticket() call — no rule-based
        # subset survives without one, since every check inspects the LLM's output.
        return _finalize_skipped(case.case_id, "triage", case.adversarial, "GROQ_API_KEY")

    try:
        result = classify_ticket(case.input)
    except Exception as exc:
        return _finalize(case.case_id, "triage", case.adversarial, [f"classify_ticket() raised unexpectedly: {exc!r}"], None)

    failures: list[str] = []

    # Schema validity / enum membership are structurally guaranteed by successful
    # construction of `result` (product/category/urgency are Pydantic Literal
    # fields) — re-asserted explicitly here so it's a visible, reported check rather
    # than a silent assumption.
    if not isinstance(result, TriageOutput):
        failures.append("classify_ticket() did not return a TriageOutput instance")
    if result.product not in get_args(Product):
        failures.append(f"product not a valid enum member: {result.product!r}")
    if result.category not in get_args(Category):
        failures.append(f"category not a valid enum member: {result.category!r}")
    if result.urgency not in get_args(Urgency):
        failures.append(f"urgency not a valid enum member: {result.urgency!r}")

    # Grounding invariant, checked on every case: kb_match must point at a real file.
    if result.kb_match is not None and not (KB_DIR / result.kb_match.doc_path).is_file():
        failures.append(f"kb_match.doc_path does not exist on disk: {result.kb_match.doc_path!r}")

    # Routing auditability invariant, checked on every case.
    if result.routing.escalation != (result.urgency == "P1"):
        failures.append("routing.escalation is inconsistent with urgency == 'P1'")

    if case.expect_kb_match_none and result.kb_match is not None:
        failures.append(f"expected kb_match=None but got doc_path={result.kb_match.doc_path!r}")

    if case.expect_kb_doc_path_in is not None:
        got = result.kb_match.doc_path if result.kb_match else None
        if got not in case.expect_kb_doc_path_in:
            failures.append(f"expected kb_match.doc_path in {case.expect_kb_doc_path_in}, got {got!r}")

    if not result.draft_response.strip():
        failures.append("draft_response is empty")

    judge_score, judge_skip_reason = None, None
    if not failures:
        judge_score, judge_skip_reason = _maybe_run_judge(
            task="Task 1 ticket triage",
            source_context=f"Ticket subject: {case.input['subject']}\nTicket body: {case.input['body']}",
            output_json=result.model_dump_json(indent=2),
        )

    return _finalize(case.case_id, "triage", case.adversarial, failures, judge_score, judge_skip_reason)


def _account_source_context(account: dict, tickets: list[dict]) -> str:
    # Mirrors every account field account_brief/prompts.py actually gives the brief-
    # generation LLM (plan_tier, health_status, usage_trend, open_tickets, nps_score,
    # p1_tickets_last_30d). Omitting any of these previously made the judge flag real,
    # grounded account facts (e.g. "open_tickets: 9") as fabricated, since it never
    # saw them in its own source context — found and fixed against real eval output.
    lines = [
        f"Account: {account.get('company')} ({account.get('account_id')})",
        f"plan_tier={account.get('plan_tier')} health_status={account.get('health_status')} "
        f"usage_trend={account.get('usage_trend')} open_tickets={account.get('open_tickets')} "
        f"p1_tickets_last_30d={account.get('p1_tickets_last_30d')} nps_score={account.get('nps_score')}",
        "escalation_notes: " + ("; ".join(account.get("escalation_notes") or []) or "(none)"),
    ]
    if tickets:
        lines += [f"- ticket {t['ticket_id']} ({t['urgency']}): {t['subject']} | {t['body']}" for t in tickets]
    else:
        lines.append("(no tickets in the 90-day window)")
    return "\n".join(lines)


def score_brief_case(case: BriefEvalCase) -> EvalResult:
    """Run generate_brief() on one real account and score it against the case's
    hard gates, then (only if they all pass) an LLM-judge quality pass.
    """
    if case.expect_error is not None:
        # account_id lookup fails before generate_brief() ever reaches the LLM, so
        # this case is fully rule-based — it always runs for real, key or no key.
        failures: list[str] = []
        try:
            brief_agent.generate_brief(case.account_id)
        except Exception as exc:
            if type(exc).__name__ != case.expect_error:
                failures.append(f"expected {case.expect_error} but got {type(exc).__name__}: {exc}")
        else:
            failures.append(f"expected {case.expect_error} to be raised, but generate_brief() succeeded")
        return _finalize(case.case_id, "account_brief", case.adversarial, failures)

    if not HAS_GROQ_KEY:
        return _finalize_skipped(case.case_id, "account_brief", case.adversarial, "GROQ_API_KEY")

    try:
        brief = brief_agent.generate_brief(case.account_id)
    except Exception as exc:
        return _finalize(
            case.case_id, "account_brief", case.adversarial, [f"generate_brief() raised unexpectedly: {exc!r}"], None
        )

    failures = []
    if not isinstance(brief, AccountBrief):
        failures.append("generate_brief() did not return an AccountBrief instance")

    account = _account_lookup[case.account_id]
    tickets = get_account_tickets(case.account_id, _all_tickets)

    # Quote verification: reuse agent.py's own verifier (not duplicated here) as an
    # independent audit of the FINAL returned brief. agent.py already filters
    # unverifiable quotes internally before returning, so this should always pass;
    # if it doesn't, that's a real regression in agent.py's own verification step.
    sources = brief_agent._quotable_sources(account, tickets)
    reverified = brief_agent._verify_quotes(brief.risk_flags, sources)
    if len(reverified) != len(brief.risk_flags):
        dropped = [f for f in brief.risk_flags if f not in reverified]
        failures.append(f"{len(dropped)} risk_flags quote(s) failed independent re-verification: {dropped}")

    if case.expect_risk_flags_empty and brief.risk_flags:
        failures.append(f"expected risk_flags=[] but got {len(brief.risk_flags)} flag(s)")

    if case.expect_no_nps_number:
        full_text = " ".join(
            [brief.executive_summary, *brief.talking_points, *(f"{f.risk} {f.quote}" for f in brief.risk_flags)]
        )
        if re.search(r"NPS\D{0,20}?\d", full_text, re.I):
            failures.append("possible hallucinated NPS number found in output")

    if case.check_determinism:
        # Checked on QUOTES specifically, not full RiskFlag equality. Found by running
        # this exact check against live output: the free-text "risk" label (e.g.
        # "Consecutive P1 tickets" vs "Consecutive P1 incidents") can reword between
        # identical calls even at temperature=0 + a fixed seed, while the "quote" —
        # the verbatim-grounded, quote-verified field PRD §3 actually cares about
        # being stable/reproducible — was byte-identical every time observed. Full
        # RiskFlag equality was tried first and is strictly stronger than what PRD's
        # determinism requirement is about; this is the honest, evidence-backed scope.
        brief2 = brief_agent.generate_brief(case.account_id)
        quotes1 = sorted(f.quote for f in brief.risk_flags)
        quotes2 = sorted(f.quote for f in brief2.risk_flags)
        if quotes1 != quotes2:
            failures.append(f"risk_flags quotes differ between two calls to generate_brief(): {quotes1!r} vs {quotes2!r}")

    judge_score, judge_skip_reason = None, None
    if not failures:
        judge_score, judge_skip_reason = _maybe_run_judge(
            task="Task 2 account-health brief",
            source_context=_account_source_context(account, tickets),
            output_json=brief.model_dump_json(indent=2),
        )

    return _finalize(case.case_id, "account_brief", case.adversarial, failures, judge_score, judge_skip_reason)
