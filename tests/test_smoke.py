"""Import and exercise data_loader.py and kb_index.py against the real data/ and
knowledge_base/ files — no mocked data. Every public function in both modules is
called from here at least once.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    build_account_lookup,
    get_account,
    get_account_tickets,
    latest_ticket_date,
    load_accounts,
    load_tickets,
)
from src.kb_index import build_retriever, load_kb_chunks

# Real ids drawn from data/, verified against the shipped dataset:
#   ACC-7397 — present in accounts.json AND has a ticket in tickets.json
#   ACC-1003 — appears as a ticket account_id but has NO record in accounts.json
KNOWN_ACCOUNT_WITH_TICKET = "ACC-7397"
ACCOUNT_ID_MISSING_FROM_ACCOUNTS = "ACC-1003"


def test_load_tickets_and_accounts():
    tickets = load_tickets()
    accounts = load_accounts()
    assert len(tickets) == 500
    assert len(accounts) == 50
    assert {"ticket_id", "account_id", "subject", "body", "created_at"} <= tickets[0].keys()
    assert {"account_id", "company", "health_status"} <= accounts[0].keys()


def test_account_lookup_hit_and_miss():
    accounts = load_accounts()
    lookup = build_account_lookup(accounts)
    assert len(lookup) == len(accounts)

    hit = get_account(KNOWN_ACCOUNT_WITH_TICKET, lookup)
    assert hit is not None
    assert hit["account_id"] == KNOWN_ACCOUNT_WITH_TICKET

    miss = get_account("ACC-NOT-A-REAL-ID", lookup)
    assert miss is None


def test_latest_ticket_date_anchors_to_max_created_at():
    tickets = load_tickets()
    anchor = latest_ticket_date(tickets)
    assert anchor is not None
    assert anchor == max(
        datetime.fromisoformat(t["created_at"].replace("Z", "+00:00")) for t in tickets
    )
    assert latest_ticket_date([]) is None


def test_get_account_tickets_returns_real_ticket_in_window():
    tickets = load_tickets()
    result = get_account_tickets(KNOWN_ACCOUNT_WITH_TICKET, tickets)
    assert len(result) >= 1
    assert all(t["account_id"] == KNOWN_ACCOUNT_WITH_TICKET for t in result)

    # Push the anchor 10 years past the latest ticket: the 90-day cutoff now sits
    # far in the future relative to every real created_at, so the window is empty.
    anchor = latest_ticket_date(tickets)
    narrow = get_account_tickets(KNOWN_ACCOUNT_WITH_TICKET, tickets, days=90, anchor=anchor + timedelta(days=3650))
    assert narrow == []


def test_get_account_tickets_graceful_on_missing_account():
    tickets = load_tickets()
    accounts = load_accounts()
    lookup = build_account_lookup(accounts)

    assert get_account(ACCOUNT_ID_MISSING_FROM_ACCOUNTS, lookup) is None
    result = get_account_tickets(ACCOUNT_ID_MISSING_FROM_ACCOUNTS, tickets)
    assert isinstance(result, list)  # never raises, even though the account doesn't exist

    assert get_account_tickets("ACC-DOES-NOT-EXIST-ANYWHERE", tickets) == []


def test_load_kb_chunks_covers_all_nine_docs():
    chunks = load_kb_chunks()
    doc_paths = {c.doc_path for c in chunks}
    assert len(doc_paths) == 9
    assert "troubleshooting/authentication-sso.md" in doc_paths
    # every chunk is a non-empty, verbatim substring of its source file
    for chunk in chunks:
        source = (Path(__file__).resolve().parent.parent / "knowledge_base" / chunk.doc_path).read_text(encoding="utf-8")
        assert chunk.text in source
        assert chunk.doc_title  # heading hierarchy metadata was captured


def test_kb_retriever_finds_real_match_for_real_ticket():
    # This ticket (real, from tickets.json) is a SecureVault SSO failure quoting the
    # AUTH_TOKEN_EXPIRED error verbatim. Both the cross-product SSO troubleshooting doc
    # and the SecureVault product doc cover that error, so either is a legitimately
    # relevant top match — the retriever must not return something unrelated (e.g. the
    # billing or onboarding docs).
    tickets = load_tickets()
    ticket = next(t for t in tickets if t["account_id"] == "ACC-5511")
    query = f"{ticket['subject']} {ticket['body']}"

    retriever = build_retriever()
    results = retriever.search(query, top_k=3)
    assert results
    top_chunk, top_score = results[0]
    assert top_score > 0
    assert top_chunk.doc_path in {
        "troubleshooting/authentication-sso.md",
        "products/securevault.md",
    }


def test_kb_retriever_returns_empty_for_no_good_match():
    retriever = build_retriever()
    assert retriever.search("zzqxvbnmqwxjklp asdfghjklqwertyzxcvbnm plkjhgfdsamnbvcxz") == []
