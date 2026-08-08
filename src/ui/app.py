"""Streamlit interface for the triage and account-brief workflows."""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

import streamlit as st

from src.account_brief.agent import AccountNotFoundError, generate_brief
from src.data_loader import load_accounts, load_tickets
from src.triage.agent import classify_ticket


@st.cache_data
def _tickets() -> list[dict[str, Any]]:
    return load_tickets()


@st.cache_data
def _accounts() -> list[dict[str, Any]]:
    return load_accounts()


def _show_triage_result(result: Any, include_draft_response: bool = True) -> None:
    """Render triage output as readable sections rather than a raw JSON blob."""
    st.subheader("Classification")
    product, area, category, urgency = st.columns(4)
    product.metric("Product", result.product)
    area.metric("Product area", result.product_area)
    category.metric("Category", result.category)
    urgency.metric("Urgency", result.urgency)

    st.markdown("#### Why this classification")
    st.write(result.reasoning)
    if include_draft_response:
        st.markdown("#### Suggested response")
        st.info(result.draft_response)

    st.markdown("#### Routing")
    routing = result.routing
    if routing.escalation:
        st.warning(f"{routing.team} — escalation/on-call required")
    else:
        st.success(routing.team)
    st.caption(routing.reasoning)

    st.markdown("#### Knowledge-base match")
    if result.kb_match is None:
        st.caption("No matching knowledge-base excerpt was found.")
    else:
        st.caption(f"{result.kb_match.doc_path} · {result.kb_match.heading}")
        st.code(result.kb_match.snippet, language="markdown")


def _show_account_brief(brief: Any) -> None:
    """Render the AccountBrief's three sections without exposing raw JSON."""
    st.subheader("Executive summary")
    st.write(brief.executive_summary)

    st.subheader("Risk flags")
    if not brief.risk_flags:
        st.success("No verified risk flags were returned for this account.")
    for flag in brief.risk_flags:
        with st.container(border=True):
            st.markdown(f"**{flag.risk}**")
            st.caption(f'Evidence: “{flag.quote}”')

    st.subheader("Talking points")
    if brief.talking_points:
        for point in brief.talking_points:
            st.markdown(f"- {point}")
    else:
        st.caption("No talking points were returned.")


def _triage_tab() -> None:
    st.header("Triage a ticket")
    tickets = _tickets()
    input_mode = st.radio("Ticket source", ("Choose a real ticket", "Paste ticket JSON"), horizontal=True)

    ticket: dict[str, Any] | None = None
    if input_mode == "Choose a real ticket":
        ticket_by_id = {ticket["ticket_id"]: ticket for ticket in tickets}
        ticket_id = st.selectbox(
            "Ticket",
            options=list(ticket_by_id),
            format_func=lambda value: f"{value} — {ticket_by_id[value]['subject']}",
        )
        ticket = ticket_by_id[ticket_id]
        st.caption(f"{ticket['company']} · {ticket['product']} · {ticket['urgency']}")
    else:
        pasted_ticket = st.text_area(
            "Ticket JSON",
            placeholder='{"subject": "Cannot connect", "body": "Production is failing..."}',
            height=180,
        )
        if pasted_ticket.strip():
            try:
                candidate = json.loads(pasted_ticket)
                if not isinstance(candidate, dict):
                    raise ValueError("Ticket JSON must be an object.")
                ticket = candidate
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"Enter a valid ticket JSON object: {exc}")

    if st.button("Triage ticket", type="primary", disabled=ticket is None):
        try:
            streamed: queue.Queue[str | None] = queue.Queue()
            state: dict[str, Any] = {}

            def run_triage() -> None:
                try:
                    state["result"] = classify_ticket(ticket, on_draft_token=streamed.put)
                except Exception as exc:
                    state["error"] = exc
                finally:
                    streamed.put(None)

            threading.Thread(target=run_triage, daemon=True).start()

            def draft_tokens():
                while True:
                    token = streamed.get()
                    if token is None:
                        break
                    yield token

            st.markdown("#### Suggested response")
            st.write_stream(draft_tokens())
            if "error" in state:
                raise state["error"]
            _show_triage_result(state["result"], include_draft_response=False)
        except Exception:
            st.error("The ticket could not be triaged right now. Check the configured LLM provider and try again.")


def _account_brief_tab() -> None:
    st.header("Account Brief")
    accounts = _accounts()
    account_by_id = {account["account_id"]: account for account in accounts}
    selected_account_id = st.selectbox(
        "Account",
        options=list(account_by_id),
        format_func=lambda value: f"{value} — {account_by_id[value]['company']}",
    )
    manual_account_id = st.text_input("Or enter an account ID", placeholder="ACC-1234")
    account_id = manual_account_id.strip() or selected_account_id

    if st.button("Generate account brief", type="primary"):
        try:
            with st.spinner("Generating account brief..."):
                _show_account_brief(generate_brief(account_id))
        except AccountNotFoundError:
            st.info(f"No account exists for account ID {account_id!r}.")
        except Exception:
            st.error("The account brief could not be generated right now. Check the configured LLM provider and try again.")


def main() -> None:
    st.set_page_config(page_title="Support AI Suite", page_icon="🛟", layout="wide")
    st.title("Support AI Suite")
    triage_tab, account_brief_tab = st.tabs(("Triage a ticket", "Account Brief"))
    with triage_tab:
        _triage_tab()
    with account_brief_tab:
        _account_brief_tab()


if __name__ == "__main__":
    main()
