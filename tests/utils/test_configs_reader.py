import json
import pytest
from pathlib import Path

from utils.configs_reader import read_json_configs


def test_read_json_configs_valid_json(tmp_path: Path):
    config_data = {"alpha": 1, "nested": {"beta": True}}

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    result = read_json_configs(config_file)

    assert result == config_data
    assert isinstance(result, dict)


def test_read_json_configs_missing_file(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError) as excinfo:
        read_json_configs(missing_file)

    assert "Config file not found" in str(excinfo.value)


def test_read_json_configs_invalid_json(tmp_path: Path):
    invalid_json = "{ this is not valid JSON"

    config_file = tmp_path / "config.json"
    config_file.write_text(invalid_json, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        read_json_configs(config_file)

    assert "Invalid JSON in config file" in str(excinfo.value)


def test_read_json_configs_accepts_string_path(tmp_path: Path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"x": 42}', encoding="utf-8")

    result = read_json_configs(str(config_file))

    assert result == {"x": 42}
