from __future__ import annotations

import os

import pytest

from zhixun_harness.fsutil import replace_with_retry


def _windows_error(code: int) -> OSError:
    error = OSError(13, "path is temporarily busy")
    error.winerror = code
    return error


def test_replace_retries_transient_windows_sharing_error(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"payload")
    original = os.replace
    calls = 0

    def flaky_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _windows_error(5)
        return original(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    replace_with_retry(source, destination, attempts=4, base_delay=0)
    assert calls == 3
    assert destination.read_bytes() == b"payload"


def test_replace_propagates_persistent_windows_error(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"payload")
    calls = 0

    def denied_replace(_src, _dst):
        nonlocal calls
        calls += 1
        raise _windows_error(5)

    monkeypatch.setattr(os, "replace", denied_replace)
    with pytest.raises(OSError) as caught:
        replace_with_retry(source, destination, attempts=3, base_delay=0)
    assert getattr(caught.value, "winerror", None) == 5
    assert calls == 3


def test_replace_does_not_retry_other_filesystem_errors(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    calls = 0

    def invalid_replace(_src, _dst):
        nonlocal calls
        calls += 1
        raise _windows_error(2)

    monkeypatch.setattr(os, "replace", invalid_replace)
    with pytest.raises(OSError):
        replace_with_retry(source, destination, attempts=4, base_delay=0)
    assert calls == 1
