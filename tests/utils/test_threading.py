from utils.threading import ThreadWorkerPool
from unittest.mock import patch
import pytest


def test_run_successful_tasks():
    pool = ThreadWorkerPool(max_workers=4)

    def task1():
        return 1

    def task2():
        return 2

    def task3():
        return 3

    tasks = [task1, task2, task3]
    results = pool.run(tasks)

    assert results == [1, 2, 3]


def test_run_tasks_out_of_order_completion():
    pool = ThreadWorkerPool(max_workers=3)

    def task_fast():
        return "fast"

    def task_slow():
        import time

        time.sleep(0.1)
        return "slow"

    tasks = [task_slow, task_fast, task_slow]
    results = pool.run(tasks)

    assert results[0] == "slow"
    assert results[1] == "fast"
    assert results[2] == "slow"


def test_run_tasks_with_exceptions():
    pool = ThreadWorkerPool(max_workers=2)

    def task_ok():
        return 42

    def task_fail():
        raise ValueError("boom")

    tasks = [task_ok, task_fail, task_ok]

    with patch("utils.logging.log") as mock_log:
        results = pool.run(tasks)

    assert results == [42, None, 42]

    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert "Task 1 failed" in args[0]
    assert kwargs["type"] == "error"


def test_run_tasks_with_rate_limit_exception_logs_info():
    pool = ThreadWorkerPool(max_workers=2)

    def task_ok():
        return 42

    def task_fail():
        raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")

    tasks = [task_ok, task_fail, task_ok]

    with patch("utils.logging.log") as mock_log:
        results = pool.run(tasks)

    assert results == [42, None, 42]

    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert "Task 1 failed" in args[0]
    assert kwargs["type"] == "info"


def test_run_tasks_with_missing_fundamentals_404_logs_warning():
    pool = ThreadWorkerPool(max_workers=2)

    def task_ok():
        return 42

    def task_fail():
        raise RuntimeError(
            'HTTP Error 404: {"quoteSummary":{"result":null,"error":{"code":"Not Found","description":"No fundamentals data found for symbol: ERIE"}}}'
        )

    tasks = [task_ok, task_fail, task_ok]

    with patch("utils.logging.log") as mock_log:
        results = pool.run(tasks)

    assert results == [42, None, 42]

    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert "Task 1 failed" in args[0]
    assert kwargs["type"] == "warning"


def test_run_tasks_with_exceptions_returned_when_enabled():
    pool = ThreadWorkerPool(max_workers=2)

    def task_ok():
        return 42

    def task_fail():
        raise ValueError("boom")

    tasks = [task_ok, task_fail, task_ok]
    results = pool.run(tasks, return_exceptions=True)

    assert results[0] == 42
    assert isinstance(results[1], ValueError)
    assert str(results[1]) == "boom"
    assert results[2] == 42


def test_run_stops_on_first_exception_when_enabled():
    pool = ThreadWorkerPool(max_workers=2)

    def task_ok():
        return 1

    def task_fail():
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        pool.run([task_ok, task_fail, task_ok], stop_on_exception=True)


def test_run_empty_task_list():
    pool = ThreadWorkerPool(max_workers=2)
    results = pool.run([])
    assert results == []


def test_run_with_max_workers_parameter():
    pool = ThreadWorkerPool(max_workers=1)

    def task():
        return "ok"

    results = pool.run([task for _ in range(3)])
    assert results == ["ok", "ok", "ok"]


def test_run_task_with_sleep_to_simulate_delay():
    pool = ThreadWorkerPool(max_workers=3)

    def task1():
        return 1

    def task2():
        import time

        time.sleep(0.05)
        return 2

    def task3():
        return 3

    tasks = [task1, task2, task3]
    results = pool.run(tasks)
    assert results == [1, 2, 3]


def test_max_workers_must_be_positive():
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        ThreadWorkerPool(max_workers=0)
