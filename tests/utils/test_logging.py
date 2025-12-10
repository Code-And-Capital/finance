from utils.logging import log
import logging


def test_info_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("hello", type="info")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.INFO
    assert record.message == "hello"


def test_warning_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("warn msg", type="warning")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert caplog.records[0].message == "warn msg"


def test_error_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("err msg", type="error")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR


def test_debug_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("dbg msg", type="debug")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG


def test_critical_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("crit msg", type="critical")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.CRITICAL


def test_invalid_type_defaults_to_info(caplog):
    with caplog.at_level(logging.DEBUG):
        log("fallback", type="notalevel")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert caplog.records[0].message == "fallback"


def test_custom_logger_name(caplog):
    with caplog.at_level(logging.DEBUG):
        log("msg", name="custom_logger")

    assert len(caplog.records) == 1
    assert caplog.records[0].name == "custom_logger"


def test_handler_added_only_once(caplog):
    # get the logger explicitly so we can inspect handlers
    logger = logging.getLogger("handler_test_logger")
    logger.handlers.clear()

    with caplog.at_level(logging.DEBUG):
        log("first", name="handler_test_logger")
        log("second", name="handler_test_logger")

    # Only one handler should still exist
    assert len(logger.handlers) == 1

    # Two log records should be emitted
    assert len(caplog.records) == 2
    assert caplog.records[0].message == "first"
    assert caplog.records[1].message == "second"


def test_logger_level_is_debug(caplog):
    # ensures logger is always set to DEBUG regardless of input type
    logger = logging.getLogger("levelcheck_logger")
    logger.handlers.clear()

    with caplog.at_level(logging.DEBUG):
        log("test", name="levelcheck_logger")

    assert logger.level == logging.DEBUG
