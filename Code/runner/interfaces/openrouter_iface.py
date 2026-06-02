"""OpenRouter interface — access 200+ LLMs through a single OpenAI-compatible endpoint.

OpenRouter (https://openrouter.ai) aggregates models from OpenAI, Anthropic, Google,
Meta, Mistral, and many others, routing requests through a unified API.

Authentication
--------------
In decreasing priority:
  1. ``access_key`` on the model config (YAML)
  2. ``openrouter`` entry in the provider key file
  3. ``OPENROUTER_API_KEY`` environment variable

YAML example::

    models:
      - name: claude-via-openrouter
        interface: openrouter
        parameters:
          model: anthropic/claude-3-5-sonnet
          temperature: 0.7
          max_tokens: 512
        roles: [teacher, student, judge]

      - name: llama3-via-openrouter
        interface: openrouter
        parameters:
          model: meta-llama/llama-3-70b-instruct
          temperature: 0.8
          max_tokens: 1024
        roles: [teacher, student]

Model IDs
---------
OpenRouter model IDs follow the pattern ``{provider}/{model-slug}``, e.g.:

* ``openai/gpt-4o``
* ``anthropic/claude-3-5-sonnet``
* ``google/gemini-flash-1.5``
* ``meta-llama/llama-3.1-70b-instruct``
* ``mistralai/mistral-large``

Browse the full list at https://openrouter.ai/models

Batching
--------
OpenRouter does not expose a native batch API; requests are issued individually
(real-time inference, no discount).
"""
from __future__ import annotations

import os
import time

from .base import ModelInterface

_BASE_URL = "https://openrouter.ai/api/v1"
_TRANSIENT_SIGNALS = ('rate limit', 'timeout', 'connection', '502', '503', '504', '529')
_FATAL_SIGNALS = ('invalid api key', 'authentication', 'model not found', 'does not exist')


class OpenRouterInterface(ModelInterface):
    """Chat completions via the OpenRouter API (OpenAI-compatible endpoint).

    Uses the ``openai`` SDK pointed at ``https://openrouter.ai/api/v1``.
    """

    def __init__(
        self,
        access_key: str | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required: pip install openai")

        key = access_key or os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise ValueError(
                "OpenRouter API key not found.  Set OPENROUTER_API_KEY or add "
                "'openrouter' to your provider key file."
            )

        # Optional attribution headers (see https://openrouter.ai/docs#headers)
        extra_headers: dict = {}
        if site_url:
            extra_headers['HTTP-Referer'] = site_url
        if site_name:
            extra_headers['X-Title'] = site_name

        self._client = OpenAI(
            api_key=key,
            base_url=_BASE_URL,
            default_headers=extra_headers or None,
        )

    def generate(self, prompt: str, parameters: dict) -> str:
        params = dict(parameters)
        model = params.pop('model')
        system_prompt = params.pop('system_prompt', None)
        temperature = params.pop('temperature', 0.7)
        max_tokens = params.pop('max_tokens', None)
        # Strip OpenRouter-specific non-inference keys
        params.pop('site_url', None)
        params.pop('site_name', None)

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        kwargs: dict = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
        }
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        kwargs.update(params)

        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                # Reasoning models (and some refusals) return content=None and put
                # text in a non-standard "reasoning" field; fall back to it before
                # failing, so one odd response does not crash the generation loop.
                content = msg.content if msg.content is not None else getattr(msg, "reasoning", None)
                if content is None:
                    raise ValueError(
                        "OpenRouter returned empty content (choices[0].message.content "
                        f"is None, finish_reason={response.choices[0].finish_reason!r}); "
                        "the model may have spent its token budget on reasoning -- "
                        "raise max_tokens or use a non-reasoning model")
                return content.strip()
            except Exception as exc:
                err_lower = str(exc).lower()
                if any(sig in err_lower for sig in _FATAL_SIGNALS):
                    raise
                last_err = exc
                if attempt < 2:
                    time.sleep(delay)
                    delay *= 2
        raise last_err  # type: ignore[misc]
