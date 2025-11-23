import logging


def log(text: str, name: str = "my_logger", type: str = "info") -> None:
    """
    Log a message to the console with a specified log level and logger name.

    This function creates or retrieves a logger, configures it if necessary,
    and outputs the message at the specified log level. If the logger already
    has handlers, it reuses them to prevent duplicate logs.

    Parameters
    ----------
    text : str
        The message to log.
    name : str, default "my_logger"
        The name of the logger. Allows separation of loggers for different
        modules or contexts.
    type : str, default "info"
        The log level to use. Supported levels are:
            - "info"
            - "warning"
            - "error"
            - "debug"
            - "critical"
        Any unrecognized type defaults to "info".

    Returns
    -------
    None
        This function logs the message to the console and does not return anything.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Add a handler only once
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    type = type.lower()

    if type == "info":
        logger.info(text)
    elif type == "warning":
        logger.warning(text)
    elif type == "error":
        logger.error(text)
    elif type == "debug":
        logger.debug(text)
    elif type == "critical":
        logger.critical(text)
    else:
        logger.info(text)
