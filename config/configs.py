"""Stateful configuration loader utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PathLike = str | Path
ConfigInput = PathLike | dict[str, Any]


class Configs:
    """Stateful JSON configuration container.

    The class stores either a configuration file path or an in-memory
    configuration dictionary and exposes a unified loading interface.
    """

    def __init__(
        self,
        source: ConfigInput | None = None,
        *,
        path: PathLike | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the config container.

        Parameters
        ----------
        source
            Optional config source. May be either a file path or a dictionary.
        path
            Optional explicit path to a JSON configuration file.
        data
            Optional preloaded configuration dictionary.

        Notes
        -----
        Precedence is: explicit `data`, then `source` when it is a dict,
        then explicit `path`, then `source` when it is path-like.
        """
        self.path: Path | None = None
        self.data: dict[str, Any] = {}

        if isinstance(source, dict):
            self.data = dict(source)
        elif source is not None:
            self.path = Path(source).expanduser()

        if path is not None:
            self.path = Path(path).expanduser()

        if data is not None:
            self.data = dict(data)

    def set_path(self, path: PathLike) -> Configs:
        """Set the JSON configuration file path and return `self` for chaining."""
        self.path = Path(path).expanduser()
        return self

    def set_data(self, data: dict[str, Any]) -> Configs:
        """Set the in-memory configuration dictionary and return `self`."""
        self.data = dict(data)
        return self

    def load(self, source: ConfigInput | None = None) -> Configs:
        """Load configuration from dictionary or JSON file into `self.data`.

        Parameters
        ----------
        source
            Optional override source to load from. May be a dictionary or path.

        Returns
        -------
        Configs
            Current instance for fluent chaining.

        Raises
        ------
        ValueError
            If no source/path/data is available for loading.
        FileNotFoundError
            If the target config file does not exist.
        ValueError
            If file contents are not valid JSON.
        """
        if isinstance(source, dict):
            self.set_data(source)
            return self

        if source is not None:
            self.set_path(source)

        if self.data:
            return self

        if self.path is None:
            raise ValueError(
                "No config source is set. Provide a dict, call set_path(), or pass source to load()."
            )

        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        try:
            with self.path.open("r", encoding="utf-8") as file:
                self.data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file: {self.path}") from exc

        return self

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the loaded configuration dictionary."""
        return dict(self.data)


default_configs = Configs()

__all__ = ["Configs", "default_configs", "PathLike", "ConfigInput"]
