# Prompt Changelog

| Version | Date | File | What changed | Why |
|---|---|---|---|---|
| triage_v1 | 2026-08-08 | `src/triage/prompts.py` | Initial structured triage prompt for product, area, category, urgency, reasoning, and customer draft response; includes retrieved KB context. | Establishes grounded classification and response drafting with a stable JSON contract. |
| brief_v1 | 2026-08-08 | `src/account_brief/prompts.py` | Initial three-section account brief prompt with exact-substring quote rules and null-NPS protection. | Ensures risk statements are grounded in supplied account evidence before downstream verification. |
| judge_v1 | 2026-08-08 | `src/eval/prompts.py` | Initial LLM-as-judge rubric (relevance/faithfulness/completeness, 0-1 each) for Task 3 quality scoring, run on Mistral. | A different model family than the primary Groq agent reduces same-model self-grading bias; only scores cases that already passed the rule-based hard gates. |
