"""Pydantic schema for Task 1 triage output (PRD §2).

TriageClassification is what the LLM itself must produce. kb_match and routing are
computed deterministically in Python (see retriever.py / agent.py) and are never
delegated to the LLM — this is what makes KB citations non-hallucinatable and
routing auditable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Product = Literal["DataBridge Pro", "CloudSync", "AnalyticsHub", "SecureVault", "WorkflowEngine"]
Category = Literal[
    "Bug",
    "Feature Request",
    "How-To",
    "Performance",
    "Billing",
    "Integration",
    "Onboarding",
    "Data Loss",
]
Urgency = Literal["P1", "P2", "P3", "P4"]


class KBMatch(BaseModel):
    """A single retrieved knowledge-base chunk, built from a real kb_index.py chunk —
    doc_path/heading/snippet are always verbatim substrings of the source file.
    """

    doc_path: str = Field(..., description="Path relative to knowledge_base/")
    heading: str = Field(..., description="Most specific heading covering the matched chunk")
    snippet: str = Field(..., description="Verbatim excerpt from the matched chunk")


class Routing(BaseModel):
    """Recommended responder team. Rule-derived from category/urgency (see agent.py's
    _route()), not LLM-decided — this is what makes it auditable.
    """

    team: str
    escalation: bool = Field(..., description="True if this also needs Escalation/On-call (P1 tickets)")
    reasoning: str


class TriageClassification(BaseModel):
    """The fields the LLM call itself must return."""

    product: Product
    product_area: str
    category: Category
    urgency: Urgency
    reasoning: str = Field(..., description="Short explanation for the product/product_area/category/urgency classification")
    draft_response: str


class TriageOutput(TriageClassification):
    """Full Task 1 result: LLM classification + deterministically computed kb_match/routing."""

    kb_match: KBMatch | None = Field(None, description="Best matching KB chunk, or null if no good match was found")
    routing: Routing
