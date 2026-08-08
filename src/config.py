"""Env loading and shared LLM defaults. Every agent module reads model config through
here — never call a provider SDK or os.getenv() directly (PRD §8).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0

# LLM-as-judge (Task 3 eval only) — a different model family than the primary agent,
# to reduce same-model self-grading bias (PRD §8).
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "mistral")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = "mistral-small-latest"
