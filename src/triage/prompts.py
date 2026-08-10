"""Versioned prompt templates for Task 1 ticket triage."""

from __future__ import annotations

from typing import get_args

from src.triage.schema import Category, Product, TriageClassification, Urgency

PROMPT_VERSION = "triage_v1"

_SYSTEM_TEMPLATE = """You are a support-ticket triage assistant for a B2B SaaS company with five \
products: {products}.

Given a raw support ticket, classify it and draft a short first response. Respond with a \
single JSON object with exactly these keys: product, product_area, category, urgency, \
reasoning, draft_response. Output only the raw JSON object itself — no markdown code \
fences (no ``` before or after it) and no commentary of any kind, before or after it.

- product: exactly one of {products}.
- product_area: the module/feature area within that product the ticket concerns (short \
free-text, e.g. "SSO", "Connectors", "Billing", "Dashboard").
- category: exactly one of {categories}.
- urgency: exactly one of {urgencies} ({urgency_meaning}).
- reasoning: 1-2 sentences explaining the product/product_area/category/urgency choices. If \
the ticket has signals for more than one category, say so and explain which one wins and why.
- draft_response: a short (3-5 sentence), professional first response to the customer. If a \
knowledge-base excerpt is provided below, ground the response in it and reference its steps; \
otherwise acknowledge the issue and say the team is investigating. Never invent product \
behaviour, error codes, or steps that aren't in the ticket or the provided excerpt.

Always return a single best-guess classification, even if the ticket is ambiguous or sparse — \
never leave a field blank or return anything other than the JSON object."""

_URGENCY_MEANING = "P1=critical/business stopped, P2=major impact, P3=moderate, P4=low/cosmetic"


def build_classify_prompt(subject: str, body: str, kb_snippet: str | None = None) -> list[dict]:
    """Chat messages for the single classify+draft LLM call. `kb_snippet` (if any)
    is the verbatim KB chunk text retrieved by retriever.py — the LLM only ever
    *reads* it to write draft_response; it never chooses which doc to cite.
    """
    system = _SYSTEM_TEMPLATE.format(
        products=", ".join(get_args(Product)),
        categories=", ".join(get_args(Category)),
        urgencies=", ".join(get_args(Urgency)),
        urgency_meaning=_URGENCY_MEANING,
    )

    user_parts = [f"Subject: {subject}", "", f"Body: {body}"]
    if kb_snippet:
        user_parts += ["", "Relevant knowledge-base excerpt:", kb_snippet]
    else:
        user_parts += ["", "(No matching knowledge-base excerpt was found.)"]
    user_parts += ["", f"Required JSON keys: {', '.join(TriageClassification.model_fields.keys())}"]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
