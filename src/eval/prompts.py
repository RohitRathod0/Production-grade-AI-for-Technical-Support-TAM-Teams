"""Versioned rubric prompt for LLM-as-judge scoring (Task 3, PRD §4). Runs on Mistral
(config.JUDGE_PROVIDER) — a different model family than the primary Groq agent, to
reduce same-model self-grading bias.
"""

from __future__ import annotations

PROMPT_VERSION = "judge_v1"

_SYSTEM_TEMPLATE = """You are a strict but fair QA judge. You are given the real source data an \
AI support-ops system was given, and the JSON output it produced from that data. Score three \
dimensions, each a number from 0.0 to 1.0:

- relevance: does the output actually address the specific ticket/account given, rather than \
being generic boilerplate?
- faithfulness: does the output avoid stating any fact (a number, a name, a claim) that is not \
present in the source data below? Penalize fabrication heavily even if the output otherwise reads well.
- completeness: does the output cover what was asked of it (classification + KB grounding + \
draft response, or executive summary + risk flags + talking points), without missing an obvious signal present in the source data?

Return a single JSON object with exactly these keys: relevance, faithfulness, completeness, notes. \
"notes" is one short sentence explaining the scores. Do not include any text outside the JSON object."""


def build_judge_prompt(task: str, source_context: str, output_json: str) -> list[dict]:
    """Chat messages for a single judge call. `source_context` is the real ticket/
    account/ticket data the output was generated from; `output_json` is the actual
    TriageOutput/AccountBrief produced. Only called for cases that already passed
    every hard gate — there's no point judging the quality of a structural failure.
    """
    user = f"""Task: {task}

Source data the output should be grounded in:
{source_context}

Output to score:
{output_json}

Return the JSON object described above."""
    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE},
        {"role": "user", "content": user},
    ]
