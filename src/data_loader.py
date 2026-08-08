"""Load tickets.json / accounts.json read-only, join by account_id, and window tickets
to the last N days for account-health analysis.

"Today" for the 90-day window (PRD §11 point 3): the dataset's real-world "today" is
ambiguous, so we anchor to max(created_at) across ALL tickets in tickets.json rather
than wall-clock time. This makes the window reproducible on any machine, on any date —
re-running this code next year against the same data/tickets.json yields the same
90-day cutoff.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKETS_PATH = DATA_DIR / "tickets.json"
ACCOUNTS_PATH = DATA_DIR / "accounts.json"


def load_tickets(path: Path = TICKETS_PATH) -> list[dict]:
    """Read tickets.json verbatim. Read-only — never mutates or augments records."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_accounts(path: Path = ACCOUNTS_PATH) -> list[dict]:
    """Read accounts.json verbatim. Read-only — never mutates or augments records."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_account_lookup(accounts: list[dict]) -> dict[str, dict]:
    """account_id -> account record, for O(1) joins from tickets.json."""
    return {a["account_id"]: a for a in accounts}


def get_account(account_id: str, account_lookup: dict[str, dict]) -> dict | None:
    """Look up an account by id. Returns None (never raises) if account_id is not
    present — tickets.json references account_ids that don't always exist in
    accounts.json (DATA_SCHEMA.md), and that's an expected, not exceptional, case.
    """
    return account_lookup.get(account_id)


def _parse_created_at(ticket: dict) -> datetime:
    return datetime.fromisoformat(ticket["created_at"].replace("Z", "+00:00"))


def latest_ticket_date(tickets: list[dict]) -> datetime | None:
    """Max created_at across all tickets — the anchor used as "today" for windowing.
    Returns None if `tickets` is empty.
    """
    if not tickets:
        return None
    return max(_parse_created_at(t) for t in tickets)


def get_account_tickets(
    account_id: str,
    tickets: list[dict],
    days: int = 90,
    anchor: datetime | None = None,
) -> list[dict]:
    """Tickets for account_id created within the last `days` days of `anchor`.

    `anchor` defaults to latest_ticket_date(tickets) (see module docstring). Returns
    [] — never raises — if account_id has no tickets in the window, or no tickets at
    all, or `tickets` is empty.
    """
    if anchor is None:
        anchor = latest_ticket_date(tickets)
    if anchor is None:
        return []
    cutoff = anchor - timedelta(days=days)
    return [
        t
        for t in tickets
        if t.get("account_id") == account_id and _parse_created_at(t) > cutoff
    ]
