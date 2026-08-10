"""Task 1 pipeline: normalize, retrieve, classify, draft, and route a ticket."""

from __future__ import annotations

import json
from collections.abc import Callable

from src import llm_client
from src.triage.prompts import build_classify_prompt
from src.triage.retriever import retrieve_kb_match
from src.triage.schema import Routing, TriageClassification, TriageOutput

_ROUTING_TEAM_BY_CATEGORY = {
    "Bug": "Product Engineering",
    "Feature Request": "Product Engineering",
    "How-To": "Support/Docs team",
    "Performance": "Product Engineering",
    "Billing": "Billing team",
    "Integration": "Integrations/Platform team",
    "Onboarding": "Customer Success team",
    "Data Loss": "Product Engineering",
}
_MAX_CLASSIFY_ATTEMPTS = 2


class _DraftResponseExtractor:
    """Extract displayable draft text from raw streamed JSON without trusting it yet."""

    _KEY = '"draft_response"'
    _ESCAPES = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

    def __init__(self) -> None:
        self._prefix = ""
        self._seen_key = False
        self._in_value = False
        self._finished = False
        self._escaped = False
        self._unicode_digits = ""

    def feed(self, chunk: str) -> list[str]:
        """Return decoded draft-response fragments from one raw JSON stream chunk."""
        emitted: list[str] = []
        for char in chunk:
            if self._finished:
                continue
            if not self._seen_key:
                self._prefix = (self._prefix + char)[-len(self._KEY) :]
                if self._prefix == self._KEY:
                    self._seen_key = True
                    self._prefix = ""
                continue
            if not self._in_value:
                if char == '"':
                    self._in_value = True
                continue
            if self._unicode_digits:
                self._unicode_digits += char
                if len(self._unicode_digits) == 4:
                    emitted.append(chr(int(self._unicode_digits, 16)))
                    self._unicode_digits = ""
                continue
            if self._escaped:
                self._escaped = False
                if char == "u":
                    self._unicode_digits = ""
                else:
                    emitted.append(self._ESCAPES.get(char, char))
                continue
            if char == "\\":
                self._escaped = True
            elif char == '"':
                self._finished = True
            else:
                emitted.append(char)
        return emitted


def _normalize(ticket: dict | str) -> tuple[str, str]:
    """Accept either a ticket dict or raw ticket text."""
    if isinstance(ticket, str):
        return "", ticket
    return ticket.get("subject") or "", ticket.get("body") or ""


def _route(category: str, urgency: str) -> Routing:
    """Compute auditable routing from the LLM's validated category and urgency."""
    team = _ROUTING_TEAM_BY_CATEGORY.get(category, "General Support")
    escalation = urgency == "P1"
    reasoning = f"category='{category}' routes to {team}."
    if escalation:
        reasoning += " urgency=P1 also flags Escalation/On-call."
    return Routing(team=team, escalation=escalation, reasoning=reasoning)


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing ``` (optionally ```json) fence if present.

    llm_client.chat's streaming path omits response_format=json_object (Groq buffers
    the whole completion server-side before emitting anything when json_mode+stream
    are combined, defeating real token-by-token streaming — see llm_client._groq_call
    for the measured evidence). Without provider-enforced JSON mode, Groq wraps the
    JSON object in a markdown code fence on every call observed live, even though the
    prompt explicitly says not to — so this strip is a required second line of
    defense, not a nicety, for the streaming path to ever produce parseable JSON.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    stripped = stripped[3:]
    if stripped[:4].lower() == "json":
        stripped = stripped[4:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _classify(
    subject: str,
    body: str,
    kb_snippet: str | None,
    on_draft_token: Callable[[str], None] | None = None,
) -> TriageClassification:
    """Classify once complete; optionally expose draft-response tokens while buffering."""
    messages = build_classify_prompt(subject=subject, body=body, kb_snippet=kb_snippet)
    last_error: Exception | None = None
    for _ in range(_MAX_CLASSIFY_ATTEMPTS):
        if on_draft_token is None:
            raw = llm_client.chat(messages, json_mode=True)
        else:
            extractor = _DraftResponseExtractor()
            raw_parts: list[str] = []
            for chunk in llm_client.chat(messages, json_mode=True, stream=True):
                raw_parts.append(chunk)
                for token in extractor.feed(chunk):
                    on_draft_token(token)
            raw = "".join(raw_parts)
        try:
            return TriageClassification.model_validate(json.loads(_strip_code_fence(raw)))
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"LLM classification failed after {_MAX_CLASSIFY_ATTEMPTS} attempts: {last_error}")


def classify_ticket(
    ticket: dict,
    on_draft_token: Callable[[str], None] | None = None,
) -> TriageOutput:
    """Run the validated triage pipeline, streaming only the customer draft if asked."""
    subject, body = _normalize(ticket)
    kb_match = retrieve_kb_match(f"{subject} {body}".strip())
    classification = _classify(
        subject,
        body,
        kb_snippet=kb_match.snippet if kb_match else None,
        on_draft_token=on_draft_token,
    )
    routing = _route(classification.category, classification.urgency)
    return TriageOutput(**classification.model_dump(), kb_match=kb_match, routing=routing)
