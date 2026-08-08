"""FastAPI router: GET /account-brief/{account_id}."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.account_brief.agent import AccountNotFoundError, generate_brief
from src.account_brief.schema import AccountBrief

router = APIRouter()


@router.get("/account-brief/{account_id}", response_model=AccountBrief)
def account_brief(account_id: str) -> AccountBrief:
    """Generate the 3-section account-health brief for account_id."""
    try:
        return generate_brief(account_id)
    except AccountNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Account brief failed: {exc}") from exc
