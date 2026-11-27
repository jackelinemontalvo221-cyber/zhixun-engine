"""Local-only PyTRIO Qwen gateway for realistic offline harness runs.

Production must continue using the contest host gateway (``rnd.model``).
"""

from __future__ import annotations

import os
import tomllib
import uuid
from pathlib import Path

from zhixun_harness.errors import DecisionError
from zhixun_harness.pipeline import ModelReceipt


DEFAULT_BASE_URL = "https://pytrio.com/api/openai/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"


def load_api_key(path: str | Path | None = None) -> str:
    """Read a key from the environment or the existing local PyTRIO profile."""
    environment = os.environ.get("PYTRIO_API_KEY", "").strip()
    if environment:
        return environment
    profile = Path(path).expanduser() if path else Path.home() / ".pytrio" / "config.toml"
    if not profile.is_file():
        raise DecisionError("PyTRIO is not logged in; run `trio login` first")
    try:
        value = tomllib.loads(profile.read_text(encoding="utf-8")).get("api_key")
    except (OSError, ValueError) as exc:
        raise DecisionError("cannot read the local PyTRIO profile") from exc
    if not isinstance(value, str) or not value.strip():
        raise DecisionError("PyTRIO API key is missing; run `trio login` again")
    return value.strip()


class PyTRIOQwenGateway:
    """Expose the real OpenAI-compatible Qwen API through ``ModelGateway``."""

    max_output_tokens = 512

    def __init__(
        self,
        round_object,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_output_tokens: int = 512,
        client_factory=None,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise DecisionError("PyTRIO API key is required")
        self.round = round_object
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 512:
            raise DecisionError("invalid local Qwen output limit")
        self.max_output_tokens = max_output_tokens
        self.client_factory = client_factory

    def call(self, messages, **options):
        max_tokens = options.get("max_tokens", self.max_output_tokens)
        if type(max_tokens) is not int or not 1 <= max_tokens <= self.max_output_tokens:
            raise DecisionError("invalid max_tokens for local Qwen")
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": options.get("temperature", 0),
            "enable_thinking": False,
        }
        if self.client_factory is None:
            try:
                import httpx
            except ImportError as exc:
                raise DecisionError("install the optional live dependencies") from exc
            factory = httpx.Client
        else:
            factory = self.client_factory
        with factory(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
        if response.status_code >= 400:
            body = ""
            try:
                body = response.text
            except Exception:
                pass
            body = " ".join(str(body or "").split())[:1000]
            retry_after = ""
            try:
                retry_after = response.headers.get("retry-after", "")
            except Exception:
                pass
            detail = body or "no response body"
            if retry_after:
                detail += f"; retry-after={retry_after}"
            raise DecisionError(f"Qwen HTTP {response.status_code}: {detail}")
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DecisionError("Qwen response has no text content") from exc
        if not isinstance(content, str) or not content.strip():
            raise DecisionError("Qwen returned empty content")
        call_id = f"pytrio-qwen-{uuid.uuid4()}"
        self.round.call_ids.append(call_id)
        return ModelReceipt(call_id, content, data)
