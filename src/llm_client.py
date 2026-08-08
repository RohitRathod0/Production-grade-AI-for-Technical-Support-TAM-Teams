"""Thin Groq/Mistral wrapper shared by the application agents and the eval judge."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from src import config

_groq_client: Groq | None = None
_mistral_client: Any | None = None

if TYPE_CHECKING:
    from mistralai.client import Mistral


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_mistral_client() -> Any:
    global _mistral_client
    if _mistral_client is None:
        try:
            # The installed mistralai 2.x distribution doesn't re-export Mistral from
            # its top-level namespace package (verified against the actual installed
            # package layout) — the real class lives one level down.
            from mistralai.client import Mistral
        except ImportError as exc:
            raise RuntimeError(
                "The Mistral SDK is unavailable. Install a current 'mistralai' package "
                "before using provider='mistral'."
            ) from exc
        _mistral_client = Mistral(api_key=config.MISTRAL_API_KEY)
    return _mistral_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def chat(
    messages: list[dict],
    json_mode: bool = False,
    temperature: float = config.DEFAULT_TEMPERATURE,
    seed: int | None = None,
    stream: bool = False,
    provider: str | None = None,
) -> str | Iterator[str]:
    """Send a completion and return its text or an iterator of content deltas.

    ``provider`` overrides config.LLM_PROVIDER for this one call — this is how the
    eval harness (src/eval/scorer.py) routes LLM-as-judge calls to
    config.JUDGE_PROVIDER (Mistral, a different model family than the primary Groq
    agent, per PRD §8) without agent code ever picking a provider itself.

    ``stream=True`` uses Groq's native streaming API and is not supported for the
    mistral provider (the judge never streams). Structured-output callers must buffer
    a complete stream and validate it before treating it as a result.
    """
    provider = provider or config.LLM_PROVIDER

    if provider == "groq":
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        if seed is not None:
            kwargs["seed"] = seed
        response = _get_groq_client().chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )
        if stream:
            return (
                delta.content
                for chunk in response
                if chunk.choices and (delta := chunk.choices[0].delta).content
            )
        return response.choices[0].message.content

    if provider == "mistral":
        if stream:
            raise ValueError("stream=True is not supported for provider='mistral'")
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = _get_mistral_client().chat.complete(
            model=config.MISTRAL_MODEL,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content

    raise ValueError(f"Unsupported provider: {provider!r}")
