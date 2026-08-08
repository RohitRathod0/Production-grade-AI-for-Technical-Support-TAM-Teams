"""Verifies the Groq -> OpenRouter fallback path in llm_client.py (PRD §8/§11.4).

A real 429 isn't something we can reliably trigger on demand against the live Groq
API, so the retry-exhaustion condition is simulated by mocking the Groq client to
always raise a real groq.RateLimitError (constructed the same way the SDK itself
would) — everything downstream of that (the retry loop actually running 3 times, the
fallback branch firing, the OpenRouter HTTP call, the logged WARNING) is real code,
not mocked.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import RateLimitError

from src import llm_client


def _make_rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request, json={"error": {"message": "rate limited"}})
    return RateLimitError("rate limited", response=response, body=None)


def test_groq_rate_limit_exhausted_falls_over_to_openrouter(monkeypatch, caplog):
    monkeypatch.setattr(llm_client.config, "OPENROUTER_API_KEY", "test-key")

    always_rate_limited = MagicMock(side_effect=_make_rate_limit_error())
    fake_groq_client = MagicMock()
    fake_groq_client.chat.completions.create = always_rate_limited
    monkeypatch.setattr(llm_client, "_get_groq_client", lambda: fake_groq_client)

    fake_openrouter_response = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        request=httpx.Request("POST", llm_client.OPENROUTER_CHAT_URL),
    )
    with patch("httpx.post", return_value=fake_openrouter_response) as mock_post:
        with caplog.at_level(logging.WARNING):
            result = llm_client.chat([{"role": "user", "content": "hi"}], json_mode=True, provider="groq")

    # The retry loop actually ran 3 times against the mocked Groq client before giving up.
    assert always_rate_limited.call_count == 3
    # The fallback actually called OpenRouter's real endpoint shape, not a stub return.
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert call_kwargs["json"]["model"] == llm_client.config.OPENROUTER_MODEL
    # The fallback's response is what chat() actually returned.
    assert result == '{"ok": true}'
    # The fallback event was logged, not silent.
    assert any("failing over to OpenRouter" in r.message for r in caplog.records)
    assert any("fallback succeeded" in r.message for r in caplog.records)


def test_non_rate_limit_error_never_falls_over():
    """A non-429 failure must not silently retry onto a different provider — that
    would mask a real config/auth problem instead of surfacing it.
    """
    from groq import AuthenticationError

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=401, request=request, json={"error": {"message": "bad key"}})
    auth_error = AuthenticationError("bad key", response=response, body=None)

    fake_groq_client = MagicMock()
    fake_groq_client.chat.completions.create = MagicMock(side_effect=auth_error)

    with patch.object(llm_client, "_get_groq_client", lambda: fake_groq_client):
        with patch("httpx.post") as mock_post:
            with pytest.raises(AuthenticationError):
                llm_client.chat([{"role": "user", "content": "hi"}], provider="groq")
            mock_post.assert_not_called()


def test_streaming_rate_limit_does_not_fall_over():
    """Fallback is explicitly not attempted for streaming calls (nothing sane to
    return mid-stream from a different provider's response shape).
    """
    fake_groq_client = MagicMock()
    fake_groq_client.chat.completions.create = MagicMock(side_effect=_make_rate_limit_error())

    with patch.object(llm_client, "_get_groq_client", lambda: fake_groq_client):
        with patch("httpx.post") as mock_post:
            with pytest.raises(RateLimitError):
                llm_client.chat([{"role": "user", "content": "hi"}], provider="groq", stream=True)
            mock_post.assert_not_called()
