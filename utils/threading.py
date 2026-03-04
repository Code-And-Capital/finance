from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import utils.logging as logging


class TaskExecutionError(RuntimeError):
    """Represents a task failure with partial ordered results."""

    def __init__(
        self,
        *,
        index: int,
        original_exception: Exception,
        partial_results: list[Any],
    ) -> None:
        super().__init__(f"Task {index} failed: {original_exception}")
        self.index = index
        self.original_exception = original_exception
        self.partial_results = partial_results


class ThreadWorkerPool:
    """Small wrapper around ThreadPoolExecutor with ordered outputs."""

    def __init__(self, max_workers: int = 8) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers

    def run(
        self,
        tasks: list[Callable[[], Any]],
        return_exceptions: bool = False,
        stop_on_exception: bool = False,
    ) -> list[Any]:
        """Execute callables concurrently and return results in input order.

        Parameters
        ----------
        tasks
            List of zero-argument callables to execute.
        return_exceptions
            If True, task exceptions are returned in the corresponding result
            positions. If False, failed tasks return ``None``.
        stop_on_exception
            If True, abort execution on the first task exception by cancelling
            all pending futures and re-raising that exception.
        """
        if not tasks:
            return []

        results: list[Any] = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(tasks[index]): index for index in range(len(tasks))
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    if self._is_rate_limit_error(exc):
                        log_type = "info"
                    elif self._is_expected_data_absence_error(exc):
                        log_type = "warning"
                    else:
                        log_type = "error"
                    logging.log(f"Task {index} failed: {exc}", type=log_type)
                    if stop_on_exception:
                        for pending in futures:
                            if pending is not future:
                                pending.cancel()
                        raise TaskExecutionError(
                            index=index,
                            original_exception=exc,
                            partial_results=results.copy(),
                        ) from exc
                    results[index] = exc if return_exceptions else None

        return results

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True when an exception message indicates a retriable Yahoo rate limit."""
        message = str(exc).lower()
        return (
            "too many requests" in message
            or "rate limited" in message
            or "earnings date" in message
        )

    @staticmethod
    def _is_expected_data_absence_error(exc: Exception) -> bool:
        """Return True for known non-critical Yahoo missing-data responses."""
        message = str(exc).lower()
        return "no fundamentals data found for symbol" in message or (
            "http error 404" in message and "quotesummary" in message
        )
