from utils.threading import ThreadWorkerPool
from unittest.mock import patch


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

    # Results returned in order
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

    # Order must match input, not completion order
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

    # The failed task should return None
    assert results == [42, None, 42]

    # Logging called once with an error
    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert "Task 1 failed" in args[0]
    assert kwargs["type"] == "error"


def test_run_empty_task_list():
    pool = ThreadWorkerPool(max_workers=2)
    results = pool.run([])
    assert results == []


def test_run_with_max_workers_parameter():
    # Basic sanity check that setting max_workers doesn't break functionality
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
