# support-ai-suite

Support-ticket triage, account-health briefs, and an evaluation harness over the supplied synthetic support dataset.

## Setup

Requires Python 3.11 and Git.

```bash
git clone <repository-url>
cd support-ai-suite

python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies and create your local environment file:

```bash
pip install -r requirements.txt
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# macOS/Linux
cp .env.example .env
```

Fill in `.env`. It is git-ignored; never commit provider keys.

| Variable | Needed for | Notes |
|---|---|---|
| `GROQ_API_KEY` | Tasks 1 and 2 | Primary provider: Groq, using `llama-3.3-70b-versatile`. |
| `MISTRAL_API_KEY` | Task 3 evaluation | Independent LLM-as-judge: Mistral, using `mistral-small-latest`. |
| `OPENROUTER_API_KEY` | Optional resilience | Used only after Groq exhausts all retries on a non-streaming 429 rate-limit failure. |
| `LLM_PROVIDER` | Optional | Defaults to `groq`. |
| `JUDGE_PROVIDER` | Optional | Defaults to `mistral`. |
| `OPENROUTER_MODEL` | Optional | Defaults to `inclusionai/ling-3.0-tiny:free`; verify availability before relying on a free-tier model. |

Provider routing follows PRD §8: Groq is primary, OpenRouter is the narrow 429 fallback, and Mistral judges outputs rather than grading Groq with Groq. Streaming calls do not fail over mid-response.

## Sample runs

The live model output contains free text, so wording can vary. The JSON structures, retrieved evidence, and quote-verification rules are the stable contract.

### Task 1 — Ticket triage

This session ran the Task 1 pipeline against the real `TKT-10000` record:

```powershell
.\.venv\Scripts\python.exe -c "import json; from pathlib import Path; from src.triage.agent import classify_ticket; ticket=next(x for x in json.loads(Path('data/tickets.json').read_text(encoding='utf-8')) if x['ticket_id']=='TKT-10000'); print(classify_ticket(ticket).model_dump_json(indent=2))"
```

Actual output:

```json
{
  "product": "DataBridge Pro",
  "product_area": "Data Ingestion",
  "category": "Feature Request",
  "urgency": "P2",
  "reasoning": "The ticket is classified as a Feature Request because the customer is asking for a new functionality (bulk archive entries) that is not currently available in DataBridge Pro. The urgency is P2 (major impact) because the customer's current workaround is manual and not scalable, which is impacting their operations as their usage has scaled to 116 users.",
  "draft_response": "Thank you for reaching out to us about your need for bulk archive entries in the Data Ingestion module of DataBridge Pro. We understand that the current one-by-one process is not scalable for your growing team. Our team is currently reviewing your request and we will be in touch soon to discuss possible solutions. In the meantime, please let us know if there's anything else we can assist you with.",
  "kb_match": {
    "doc_path": "troubleshooting/authentication-sso.md",
    "heading": "Scope Errors (`403 Forbidden: insufficient_scope`)",
    "snippet": "## Scope Errors (`403 Forbidden: insufficient_scope`)\n\nTokens are scoped at creation time. Common missing scopes:\n\n| Product | Operation | Required Scope |\n|---------|-----------|---------------|\n| DataBridge Pro | Write pipeline config | `pipelines:write` |\n| DataBridge Pro | Read audit logs | `audit:read` |\n| CloudSync | Manage permissions | `permissions:admin` |\n| AnalyticsHub | Export data |…"
  },
  "routing": {
    "team": "Product Engineering",
    "escalation": false,
    "reasoning": "category='Feature Request' routes to Product Engineering."
  }
}
```

### Task 2 — Account-health brief

This session ran the Task 2 pipeline against the real `ACC-7397` account:

```powershell
.\.venv\Scripts\python.exe -c "from src.account_brief.agent import generate_brief; print(generate_brief('ACC-7397').model_dump_json(indent=2))"
```

Actual output:

```json
{
  "executive_summary": "The account health of Gavin Belson Co is currently At Risk. There are 9 open tickets, with a stable usage trend. The NPS score is unavailable. Recent support tickets have shown negative sentiment, and there have been consecutive P1 incidents, indicating potential issues that need to be addressed.",
  "risk_flags": [
    {
      "risk": "Consecutive P1 tickets",
      "quote": "3 consecutive P1 tickets in the last 30 days"
    },
    {
      "risk": "Negative sentiment in recent support tickets",
      "quote": "Negative sentiment detected in recent support tickets"
    }
  ],
  "talking_points": [
    "Can you tell me more about the recent integration issues you've been experiencing with AnalyticsHub and Azure AD?",
    "How can we assist in resolving the consecutive P1 incidents that have occurred in the last 30 days?",
    "What are your current priorities, and how can we help address them to improve your overall experience with our service?"
  ]
}
```

Each `risk_flags[].quote` is retained only if it is an exact substring of an escalation note or a ticket body in the account's reproducible 90-day window.

### Task 3 — Evaluation

```bash
python run.py eval
```

The current evaluation run passed all hard gates:

```text
Eval: 10/10 cases passed hard gates, 0 failed, 0 skipped.
Average quality_score (judged cases only, n=9): 0.911
wrote: eval_report.json
wrote: eval_report.md
```

The complete current results are in [`eval_report.md`](eval_report.md) and [`eval_report.json`](eval_report.json). Rate-limit exhaustion was observed in an earlier run and is documented, with the resulting fallback correction, in the design note's failure-modes section.

CI (`.github/workflows/eval.yml`) runs with no provider keys at all — no secrets are stored in GitHub Actions. Without `GROQ_API_KEY`/`MISTRAL_API_KEY`, every case that needs a live LLM call is marked `SKIPPED`, not `FAIL`; only the fully rule-based case (`brief_05_missing_account`, which never reaches the LLM) runs for real. CI exits 0 as long as nothing that actually ran, failed:

```text
Eval: 1/10 cases passed hard gates, 0 failed, 9 skipped.
```

## Interfaces

```bash
python run.py serve --port 8000   # FastAPI: POST /triage, GET /account-brief/{id}
python run.py ui                  # Streamlit UI
```

## Design note

[Read the design note](DESIGN_NOTE.md).

## Loom

[Loom recording — add link](https://www.loom.com/share/REPLACE_ME)
