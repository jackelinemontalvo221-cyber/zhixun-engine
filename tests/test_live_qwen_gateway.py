"""Unit tests for the opt-in real-Qwen local gateway."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from member3.live_gateway import PyTRIOQwenGateway, load_api_key
from zhixun_harness.errors import DecisionError


class _Response:
    status_code = 200

    def json(self):
        return {
            "choices": [{"message": {"content": '{"action":"no_update","params":{},"rationale":"ok"}'}}],
            "usage": {"total_tokens": 12},
        }


class _ConflictResponse:
    status_code = 409
    text = '{"detail":"a sampling session is already active"}'
    headers = {"retry-after": "5"}

    def json(self):
        return {}


class _Client:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.request = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return _Response()


class _ConflictClient(_Client):
    def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return _ConflictResponse()


def test_api_key_can_come_from_the_local_pytrio_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTRIO_API_KEY", raising=False)
    profile = tmp_path / "config.toml"
    profile.write_text('api_key = "local-key"\n', encoding="utf-8")
    assert load_api_key(profile) == "local-key"


def test_gateway_returns_receipt_and_registers_round_call_id():
    round_object = SimpleNamespace(call_ids=[])
    gateway = PyTRIOQwenGateway(
        round_object,
        "secret",
        client_factory=_Client,
    )
    receipt = gateway.call([{"role": "user", "content": "decide"}], max_tokens=64, temperature=0)
    assert receipt.text.startswith('{"action":"no_update"')
    assert receipt.call_id.startswith("pytrio-qwen-")
    assert round_object.call_ids == [receipt.call_id]


def test_gateway_rejects_missing_key():
    with pytest.raises(DecisionError, match="API key"):
        PyTRIOQwenGateway(SimpleNamespace(call_ids=[]), "")


def test_gateway_surfaces_http_conflict_body_and_retry_hint():
    gateway = PyTRIOQwenGateway(
        SimpleNamespace(call_ids=[]),
        "secret",
        client_factory=_ConflictClient,
    )
    with pytest.raises(DecisionError) as caught:
        gateway.call([{"role": "user", "content": "decide"}], max_tokens=64)
    message = str(caught.value)
    assert "HTTP 409" in message
    assert "sampling session is already active" in message
    assert "retry-after=5" in message
