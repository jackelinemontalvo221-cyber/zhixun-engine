"""Acceptance checks against the built Docker image (Linux, offline, read-only, non-root).

Skipped unless ZH_IMAGE names an image, e.g.:
    ZH_IMAGE=zhixun-harness:0.1.0 python -m pytest -q tests/test_image.py
``scripts/verify_image.sh`` runs these after its shell checks when the SDK is installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from tests.test_round_protocol import PACKAGES, Host, build_start

IMAGE = os.environ.get("ZH_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="set ZH_IMAGE to an image to run acceptance checks")
LOCKED = ["--network", "none", "--read-only", "--user", "65534:65534",
          "--tmpfs", "/tmp:mode=1777", "--tmpfs", "/workspace:mode=1777"]
TASKS = {"A": "classification", "B": "regression", "C": "regression"}


def docker(*args, **kwargs):
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kwargs)


def test_image_size_is_under_2gb():
    size = int(docker("image", "inspect", "--format", "{{.Size}}", IMAGE, check=True).stdout.strip())
    assert size < 2 * 1024 ** 3, f"image is {size / 1024 ** 3:.2f} GiB"


def test_image_records_a_real_commit():
    env = docker("image", "inspect", "--format", "{{json .Config.Env}}", IMAGE, check=True).stdout
    commit = next((item.split("=", 1)[1] for item in json.loads(env) if item.startswith("ZHIXUN_GIT_COMMIT=")), "")
    assert commit and commit != "unknown", "build with scripts/build_image.sh (--build-arg GIT_COMMIT=...)"


def test_describe_offline_read_only_non_root():
    out = docker("run", "--rm", *LOCKED, IMAGE, "describe", check=True).stdout
    data = json.loads(out)
    assert data["protocol"] == "contest.harness.workspace.v2" and data["sdk_version"] == "1.0.5"


def test_image_has_no_online_experiment_client():
    """Formal evaluation is offline; online logging from the container is not allowed."""
    result = docker("run", "--rm", "--network", "none", "--read-only", "--entrypoint", "python3", IMAGE,
                    "-c", "import importlib.util as u; print(u.find_spec('swanlab') is None)")
    assert result.stdout.strip() == "True"


@pytest.mark.parametrize("key", ["A", "B", "C"])
def test_predict_with_each_official_checkpoint(key):
    package = PACKAGES[key]
    rows = [json.loads(line) for line in (package / "public" / "validation.jsonl").read_text(
        encoding="utf-8").splitlines()[:20]]
    stdin = "".join(json.dumps({"task": TASKS[key], "inputs": row["inputs"]}) + "\n" for row in rows)
    started = time.monotonic()
    result = docker("run", "--rm", "-i", *LOCKED, "-v", f"{package / 'checkpoint.npz'}:/model/checkpoint.npz:ro",
                    "-e", "CONTEST_ARTIFACT_PATH=/model/checkpoint.npz", IMAGE, "predict", input=stdin)
    assert result.returncode == 0, result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == len(rows)
    assert all("outputs" in line for line in lines)
    assert time.monotonic() - started < 60


@pytest.mark.parametrize("key", ["A", "C"])
def test_full_round_offline_read_only_non_root(key):
    package = PACKAGES[key]
    start = build_start(1, key)
    start["task_contract"]["public_projection"] = {"train_path": "/data/train.jsonl",
                                                    "validation_path": "/data/validation.jsonl"}
    process = subprocess.Popen(
        ["docker", "run", "--rm", "-i", *LOCKED,
         "-v", f"{package / 'public'}:/data:ro",
         "-v", f"{package / 'checkpoint.npz'}:/previous/checkpoint.npz:ro",
         IMAGE, "train"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    host = Host(start)
    host.serve(process)
    process.wait(timeout=900)
    if host.failure:
        raise host.failure
    assert process.returncode == 0, process.stderr.read()
    record = host.record
    assert record is not None and record["model_call_ids"]
    assert record["selected_artifact_id"] in {"parent-1", "model-1"}
    assert "evidence" in record["configuration"]
    assert len(record["hypothesis"].encode("utf-8")) <= 4000
    states = host.uploaded("state")
    assert states and any(path.endswith("round-log.json") for path in states[-1])
    assert len(states[-1]) <= 64


def test_sigterm_stops_the_container_promptly():
    name = f"zh-term-{uuid.uuid4().hex[:8]}"
    process = subprocess.Popen(["docker", "run", "--rm", "-i", "--name", name, *LOCKED, IMAGE, "train"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 30
    while docker("inspect", "-f", "{{.State.Running}}", name).stdout.strip() != "true":
        assert time.monotonic() < deadline, "container did not start"
        time.sleep(0.2)
    time.sleep(1.0)                          # let python reach its stdin read
    began = time.monotonic()
    docker("kill", "--signal", "TERM", name, check=True)
    code = process.wait(timeout=20)
    assert time.monotonic() - began < 5, "SIGTERM was ignored (PID 1 without a handler)"
    assert code == 143
