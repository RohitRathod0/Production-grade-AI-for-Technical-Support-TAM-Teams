"""FastAPI router: POST /triage."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.triage.agent import classify_ticket
from src.triage.schema import TriageOutput

router = APIRouter()


class TriageRequest(BaseModel):
    subject: str = ""
    body: str


@router.post("/triage", response_model=TriageOutput)
def triage_ticket(request: TriageRequest) -> TriageOutput:
    """Classify a raw ticket ({subject, body}) and return the full TriageOutput."""
    try:
        return classify_ticket(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Triage failed: {exc}") from exc
