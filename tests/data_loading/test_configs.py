import json
from pathlib import Path

import pytest

from config.configs import Configs


def test_load_valid_json(tmp_path: Path):
    config_data = {"alpha": 1, "nested": {"beta": True}}

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    configs = Configs(path=config_file).load().as_dict()

    assert configs == config_data
    assert isinstance(configs, dict)


def test_load_missing_file(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError) as excinfo:
        Configs(path=missing_file).load()

    assert "Config file not found" in str(excinfo.value)


def test_load_invalid_json(tmp_path: Path):
    invalid_json = "{ this is not valid JSON"

    config_file = tmp_path / "config.json"
    config_file.write_text(invalid_json, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        Configs(path=config_file).load()

    assert "Invalid JSON in config file" in str(excinfo.value)


def test_load_accepts_string_path(tmp_path: Path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"x": 42}', encoding="utf-8")

    result = Configs(path=str(config_file)).load().as_dict()

    assert result == {"x": 42}


def test_load_without_source_raises():
    with pytest.raises(ValueError, match="No config source is set"):
        Configs().load()


def test_init_accepts_dict_source():
    result = Configs(source={"x": 1, "nested": {"y": 2}}).load().as_dict()
    assert result == {"x": 1, "nested": {"y": 2}}


def test_load_accepts_dict_source():
    cfg = Configs(path="/tmp/does-not-matter.json")
    result = cfg.load({"a": True}).as_dict()
    assert result == {"a": True}


def test_set_data_overrides_existing_data(tmp_path: Path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"x": 42}', encoding="utf-8")

    cfg = Configs(path=config_file).load()
    out = cfg.set_data({"x": 99}).as_dict()

    assert out == {"x": 99}
