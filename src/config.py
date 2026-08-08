"""Env loading and shared LLM defaults. Every agent module reads model config through
here — never call a provider SDK or os.getenv() directly (PRD §8).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# NOTE: `os.getenv(key, default)` only falls back to `default` when the key is
# ABSENT from .env — a key present but set to an empty string (e.g. "FOO=" with
# nothing after the "=", which .env.example's placeholders look exactly like) returns
# "" instead. Found live: OPENROUTER_MODEL="" from .env made an actual OpenRouter
# call fail with "No models provided" instead of using the intended default. Every
# var below that has a real (non-empty-string) default uses `... or default` instead
# to guard against that.

LLM_PROVIDER = os.getenv("LLM_PROVIDER") or "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0

# LLM-as-judge (Task 3 eval only) — a different model family than the primary agent,
# to reduce same-model self-grading bias (PRD §8).
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER") or "mistral"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = "mistral-small-latest"

# Fallback provider for the primary (groq) call, used only when every retry attempt
# is exhausted on a 429 rate limit (PRD §8, §11.4). OpenRouter's free-model catalog
# turns over fast: meta-llama/llama-3.1-8b-instruct:free and llama-3.3-70b-instruct:free
# (both once standard "known-good" free picks) returned 404 "unavailable for free"
# when actually tested live against OpenRouter's API on the date this was written.
# inclusionai/ling-3.0-tiny:free was verified live (real 200 response) as of the same
# test — but re-verify at openrouter.ai/models?q=free before relying on it later, and
# don't assume any hardcoded free-tier id stays valid.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL") or "inclusionai/ling-3.0-tiny:free"
