import json
from pathlib import Path
from typing import Union, Dict, Any

def read_json_configs(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a JSON configuration file from disk and return its contents.

    Args:
        path: Path to the JSON config file. May be a string or Path object.

    Returns:
        Dictionary containing the parsed JSON configuration. Nested JSON
        objects are returned as nested dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {path}") from e
