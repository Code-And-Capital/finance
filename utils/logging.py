import logging
from pathlib import Path
from typing import Final

_LEVEL_MAP: Final[dict[str, int]] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def log(
    text: str,
    name: str = "my_logger",
    type: str = "info",
    *,
    level: str | None = None,
) -> None:
    """Log to stdout with stable formatting and level mapping.

    The `type` parameter is kept for backward compatibility.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    selected = (level or type).lower()
    log_level = _LEVEL_MAP.get(selected, logging.INFO)
    logger.log(log_level, text)


def configure_file_logging(
    *,
    log_file: str,
    name: str = "my_logger",
) -> None:
    """Attach a file handler to the shared logger if not already attached."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = Path(handler.baseFilename).resolve()
            if existing == path:
                return

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
