"""Exercise deterministic account-brief risk rules against real fixture data."""

from __future__ import annotations

import json

from src.account_brief.risk_rules import candidate_risk_signals
from src.account_brief.schema import AccountBrief
from src.data_loader import get_account_tickets, load_accounts, load_tickets


def test_candidate_risk_signals_for_real_accounts(capsys) -> None:
    accounts = {account["account_id"]: account for account in load_accounts()}
    tickets = load_tickets()

    # At Risk has one matched ticket; Healthy and Churning deliberately have none.
    selected_ids = ("ACC-3336", "ACC-3033", "ACC-2944")
    actual_output: dict[str, list[dict[str, str]]] = {}

    for account_id in selected_ids:
        account = accounts[account_id]
        filtered_tickets = get_account_tickets(account_id, tickets)
        signals = candidate_risk_signals(account, filtered_tickets)
        actual_output[account_id] = [signal.model_dump() for signal in signals]

        # The schema is instantiated for each real result shape without invoking an LLM.
        AccountBrief(
            executive_summary="Deterministic rule test only.",
            risk_flags=signals,
            talking_points=[],
        )

    print(json.dumps(actual_output, indent=2))
    captured = capsys.readouterr().out
    print(captured, end="")

    assert accounts["ACC-3336"]["health_status"] == "At Risk"
    assert len(get_account_tickets("ACC-3336", tickets)) == 1
    assert accounts["ACC-3033"]["health_status"] == "Healthy"
    assert get_account_tickets("ACC-3033", tickets) == []
    assert accounts["ACC-2944"]["health_status"] == "Churning"
    assert get_account_tickets("ACC-2944", tickets) == []
    assert any(signal["risk"] == "Account health is not Healthy" for signal in actual_output["ACC-2944"])
