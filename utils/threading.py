from concurrent.futures import ThreadPoolExecutor, as_completed
import utils.logging as logging


class ThreadWorkerPool:
    """
    Generic thread-based execution pool for parallelizing tasks.

    This class provides a simple interface to execute a list of callables
    concurrently using `ThreadPoolExecutor`. It is particularly useful for
    parallelizing per-ticker operations such as retrieving company info,
    historical prices, corporate actions, or financial statements from Yahoo Finance.

    Attributes
    ----------
    max_workers : int
        Maximum number of threads to use for parallel execution.
    """

    def __init__(self, max_workers=8) -> None:
        """
        Initialize the ThreadWorkerPool.

        Parameters
        ----------
        max_workers : int, default 8
            Maximum number of threads to use.
        logger : optional
            Custom logger to use for error reporting (currently unused).
        """
        self.max_workers = max_workers

    def run(self, tasks):
        """
        Execute a list of callables concurrently and return results in order.

        Parameters
        ----------
        tasks : list of callable
            A list of functions (without arguments) to execute in parallel.

        Returns
        -------
        list
            A list of results from each task. If a task raises an exception,
            its corresponding result will be `None`.

        Notes
        -----
        - Results are returned in the same order as the input `tasks` list.
        - Exceptions from tasks are logged using the `log` function.
        """
        results = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(tasks[i]): i for i in range(len(tasks))}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logging.log(f"Task {idx} failed: {e}", type="error")
                    results[idx] = None

        return results
