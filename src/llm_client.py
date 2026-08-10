"""Thin Groq/Mistral/OpenRouter wrapper shared by the application agents and the eval
judge. The primary (groq) call retries with exponential backoff; if every retry is
exhausted on a 429 rate limit, chat() fails over to OpenRouter's free tier (PRD
§8/§11.4) transparently — callers never see a different return shape or need to know
a fallback happened. Every fallback event is logged at WARNING/ERROR so it's visible
in the CLI/Loom demo and citable as real evidence in DESIGN_NOTE.md's failure modes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx
from groq import Groq
from groq import RateLimitError as GroqRateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from src import config

logger = logging.getLogger(__name__)

_groq_client: Groq | None = None
_mistral_client: Any | None = None

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def _groq_call(
    messages: list[dict], json_mode: bool, temperature: float, seed: int | None, stream: bool
) -> str | Iterator[str]:
    # response_format={"type": "json_object"} makes Groq buffer and validate the
    # *entire* completion server-side before emitting anything, then send it as one
    # SSE chunk — confirmed live (single-chunk responses, byte-identical whether
    # measured against the real triage prompt or a minimal JSON-shaped prompt, 100%
    # reproducible across trials). stream=True still "succeeds" in that the client
    # gets an iterator, but there's genuinely only one item in it, which is why
    # on_draft_token callers saw output arrive all at once despite correct
    # token-by-token plumbing on the Python side. Plain (non-JSON-mode) streaming
    # measured 162 separate chunks over ~300ms for a comparable prompt, so the SDK/
    # network path is not the bottleneck — this is specific to json_mode+stream.
    # Omit response_format only on the streaming path; callers already rely on
    # prompt-enforced JSON (the system prompt says "respond with a single JSON
    # object") plus retry-on-parse-failure (see triage/agent.py._classify), the same
    # pattern already used for the OpenRouter fallback in _openrouter_call.
    kwargs = {"response_format": {"type": "json_object"}} if json_mode and not stream else {}
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def _mistral_call(messages: list[dict], json_mode: bool, temperature: float) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _get_mistral_client().chat.complete(
        model=config.MISTRAL_MODEL,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content


def _openrouter_call(messages: list[dict], temperature: float) -> str:
    """Secondary/fallback provider (PRD §8, §11.4). Called over plain HTTP (OpenAI-
    compatible REST shape) rather than pulling in another SDK, since this path only
    runs when Groq is rate-limited after every retry — deliberately rare.

    No `response_format` is sent (unlike the primary providers): tested live, the
    default free-tier model (config.OPENROUTER_MODEL) 400s on `response_format:
    json_object` with "does not support feature". Every prompt already instructs
    "return a single JSON object" in plain text (verified live: the model complies
    without needing enforced JSON mode), and callers already retry once on a parse
    failure — so this degrades to prompt-enforced JSON rather than provider-enforced,
    an acceptable trade-off for a rarely-used fallback path.
    """
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set — cannot fail over to OpenRouter.")
    payload: dict[str, Any] = {
        "model": config.OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    response = httpx.post(
        OPENROUTER_CHAT_URL,
        headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


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

    Fallback (provider="groq" only): each retry attempt is Groq; if all 3 are
    exhausted and the final failure is specifically a 429 rate limit, this fails over
    once to OpenRouter's free tier and returns its result instead of raising. A
    non-rate-limit failure (auth, bad request, ...) is never retried onto a different
    provider — retrying a config problem on another provider would just mask it.
    Streaming responses are never eligible for fallback (there's nothing sane to
    hand back mid-stream); a rate limit on a streaming call still raises.

    ``stream=True`` uses Groq's native streaming API. Mistral has no native streaming
    call implemented, so ``stream=True`` there degrades to a single-chunk iterator over
    the full (already-buffered) response — a caller can request streaming regardless of
    provider and never hard-fail; it just won't see real token-by-token output on
    Mistral. Structured-output callers must buffer a complete stream and validate it
    before treating it as a result.
    """
    provider = provider or config.LLM_PROVIDER

    if provider == "groq":
        try:
            return _groq_call(messages, json_mode, temperature, seed, stream)
        except GroqRateLimitError as exc:
            if stream:
                raise
            logger.warning(
                "Groq rate-limited after exhausting all retries (%s) — failing over to "
                "OpenRouter (%s).",
                exc,
                config.OPENROUTER_MODEL,
            )
            try:
                result = _openrouter_call(messages, temperature)
            except Exception:
                logger.error("OpenRouter fallback also failed; no further fallback configured.", exc_info=True)
                raise
            logger.warning("OpenRouter fallback succeeded for this call.")
            return result

    if provider == "mistral":
        text = _mistral_call(messages, json_mode, temperature)
        # No native streaming call is implemented for Mistral, so a caller that asked
        # for streaming gets the same single-chunk-iterator degradation already used
        # for Groq's json_mode+stream combination above: callers already buffer the
        # full stream before treating it as a result (see _classify's
        # _DraftResponseExtractor), so one chunk is a safe, transparent substitute for
        # real token-by-token streaming — never a hard failure.
        return iter([text]) if stream else text

    raise ValueError(f"Unsupported provider: {provider!r}")
