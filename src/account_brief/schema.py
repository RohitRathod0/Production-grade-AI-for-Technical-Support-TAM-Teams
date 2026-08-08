"""Structured output contracts for the account-brief workflow."""

from pydantic import BaseModel, ConfigDict, Field


class RiskFlag(BaseModel):
    """A deterministic or model-generated risk observation with source evidence."""

    model_config = ConfigDict(extra="forbid")

    risk: str = Field(description="Short description of the account risk signal.")
    quote: str = Field(description="Verbatim account or ticket evidence for the signal.")


class AccountBrief(BaseModel):
    """The LLM-facing account-brief response shape defined in PRD section 3."""

    model_config = ConfigDict(extra="forbid")

    executive_summary: str
    risk_flags: list[RiskFlag]
    talking_points: list[str]
