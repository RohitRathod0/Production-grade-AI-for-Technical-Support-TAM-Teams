"""Deterministic candidate-risk rules for account briefs.

These rules deliberately use the account record as the source of truth for account
health metrics. ``tickets`` is the already-filtered account ticket list from
``data_loader.get_account_tickets`` and may be empty when the data sets do not join.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schema import RiskFlag


LOW_NPS_SCORE = 6
HIGH_OPEN_TICKETS_PER_LICENSED_SEAT = 0.01


def candidate_risk_signals(
    account: Mapping[str, Any], tickets: Sequence[Mapping[str, Any]] | None,
) -> list[RiskFlag]:
    """Return deterministic risk candidates for one account.

    ``tickets`` is accepted as contextual input from the 90-day data-loader join;
    it can be empty without changing the account-level assessment. This prevents
    missing ticket joins from hiding risks stored on the account record itself.
    """
    # Materialize once so callers can pass any sequence, including an empty one. The
    # count is retained as evidence context but is not treated as a risk by itself.
    matched_ticket_count = len(tickets or ())
    signals: list[RiskFlag] = []

    health_status = account.get("health_status")
    if health_status != "Healthy":
        signals.append(
            RiskFlag(
                risk="Account health is not Healthy",
                quote=f"health_status: {health_status!r}",
            )
        )

    usage_trend = account.get("usage_trend")
    if usage_trend in {"Declining", "Inactive"}:
        signals.append(
            RiskFlag(
                risk="Usage trend is declining or inactive",
                quote=f"usage_trend: {usage_trend!r}",
            )
        )

    p1_tickets = account.get("p1_tickets_last_30d") or 0
    if p1_tickets > 0:
        signals.append(
            RiskFlag(
                risk="Recent P1 support incidents",
                quote=(
                    f"p1_tickets_last_30d: {p1_tickets}; "
                    f"matched tickets in supplied window: {matched_ticket_count}"
                ),
            )
        )

    escalation_notes = account.get("escalation_notes") or []
    if isinstance(escalation_notes, str):
        escalation_notes = [escalation_notes] if escalation_notes.strip() else []
    if escalation_notes:
        signals.append(
            RiskFlag(
                risk="Escalation notes are present",
                quote="; ".join(str(note) for note in escalation_notes),
            )
        )

    nps_score = account.get("nps_score")
    if nps_score is None or nps_score <= LOW_NPS_SCORE:
        nps_evidence = "null" if nps_score is None else str(nps_score)
        signals.append(
            RiskFlag(
                risk="NPS score is low or unavailable",
                quote=f"nps_score: {nps_evidence}",
            )
        )

    open_tickets = account.get("open_tickets") or 0
    seats_licensed = account.get("seats_licensed") or 0
    if seats_licensed > 0 and open_tickets / seats_licensed >= HIGH_OPEN_TICKETS_PER_LICENSED_SEAT:
        signals.append(
            RiskFlag(
                risk="Open-ticket volume is high for account size",
                quote=(
                    f"open_tickets: {open_tickets}; seats_licensed: {seats_licensed}; "
                    f"open-ticket rate: {open_tickets / seats_licensed:.1%}"
                ),
            )
        )

    return signals
