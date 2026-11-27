import json

import pytest

from zhixun_harness.config import HarnessConfig, load_config
from zhixun_harness.errors import ConfigError


def test_defaults_are_valid():
    config = HarnessConfig()
    assert config.gate["direction"] == "maximize"
    assert config.pipeline == "fakes"


def test_unknown_config_field_is_rejected(tmp_path):
    path = tmp_path / "harness.json"
    path.write_text(json.dumps({"schema_version": "harness-config.v1", "unknown": 1}))
    with pytest.raises(ConfigError):
        load_config(path)


def test_invalid_gate_direction_is_rejected():
    with pytest.raises(ConfigError):
        HarnessConfig(gate={"direction": "sideways"})


def test_trial_options_are_validated(tmp_path):
    import json as _json

    from zhixun_harness.config import load_config as _load
    from zhixun_harness.errors import ConfigError as _ConfigError

    good = tmp_path / "good.json"
    good.write_text(_json.dumps({"schema_version": "harness-config.v1",
                                 "trial": {"force_rollback_round": 3, "bypass_rollback_gate": True}}))
    assert _load(good).trial == {"force_rollback_round": 3, "bypass_rollback_gate": True}
    for trial in ({"force_rollback_round": 1}, {"shell": "x"}, {"bypass_rollback_gate": "yes"}):
        bad = tmp_path / "bad.json"
        bad.write_text(_json.dumps({"schema_version": "harness-config.v1", "trial": trial}))
        with pytest.raises(_ConfigError):
            _load(bad)
