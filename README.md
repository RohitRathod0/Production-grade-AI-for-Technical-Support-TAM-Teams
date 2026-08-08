# support-ai-suite

## Setup

Requires Python 3.11.

```bash
git clone <this repo>
cd support-ai-suite
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

cp .env.example .env          # then fill in real values below
```

Fill in `.env` (git-ignored, never commit real keys):

| Var | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | yes | Primary LLM (Task 1 + 2 agents) — [console.groq.com](https://console.groq.com) |
| `MISTRAL_API_KEY` | yes | LLM-as-judge (Task 3 eval only) — [console.mistral.ai](https://console.mistral.ai) |
| `LLM_PROVIDER` | no | Defaults to `groq` |
| `JUDGE_PROVIDER` | no | Defaults to `mistral` |
| `OPENROUTER_API_KEY` | no | Fallback provider — only used if Groq is rate-limited after every retry ([openrouter.ai](https://openrouter.ai)); without it, that specific rate-limit case fails instead of failing over |
| `OPENROUTER_MODEL` | no | Defaults to a free-tier model id; verify it's still live at openrouter.ai/models before relying on it, free-tier catalogs change |

**Windows note:** clone to a short path (e.g. `C:\repos\...`, not a deeply nested folder). One dependency (`mistralai`) ships very long nested filenames that can exceed Windows' 260-character path limit inside a long install path.

Then run any of:

```bash
python run.py triage --ticket-file sample_ticket.json   # or --ticket-json '{"subject": "...", "body": "..."}'
python run.py brief --account-id ACC-7397
python run.py eval                                       # writes eval_report.json / eval_report.md
python run.py serve --port 8000                          # POST /triage, GET /account-brief/{id}
python run.py ui                                          # Streamlit UI
```

## Sample Runs

### Task 1

_Placeholder_

### Task 2

_Placeholder_

### Task 3

_Placeholder_

### Task 4

_Placeholder_

## Design Note

[Design Note](DESIGN_NOTE.md)

## Loom

_Placeholder for Loom recording link_
