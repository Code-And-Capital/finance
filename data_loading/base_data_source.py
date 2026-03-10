"""Base interface for reusable data-loading datasource classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence

import pandas as pd

from utils.logging import log


class BaseDataSource(ABC):
    """Template method base class for data ingestion components.

    Subclasses implement:
    - ``load``: fetch raw data from one or more sources.
    - ``transform``: apply standard cleaning and normalization.
    - ``format``: store project-ready dataframe outputs in ``self.formatted_data``.
    """

    def __init__(self) -> None:
        self.raw_data: pd.DataFrame | None = None
        self.transformed_data: pd.DataFrame | None = None
        self.formatted_data: Dict[str, pd.DataFrame] = {}

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load raw data from source systems."""
        raise NotImplementedError(
            "BaseDataSource.load() should be overwritten by subclasses."
        )

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform raw input into normalized tabular output."""
        raise NotImplementedError(
            "BaseDataSource.transform() should be overwritten by subclasses."
        )

    @abstractmethod
    def format(self, dates: Sequence[pd.Timestamp] | pd.Index | None = None) -> None:
        """Format ``self.transformed_data`` into ``self.formatted_data``."""
        raise NotImplementedError(
            "BaseDataSource.format() should be overwritten by subclasses."
        )

    def _requested_figis(self) -> list[str]:
        """Return normalized requested FIGI list."""
        raw = [self.figis] if isinstance(self.figis, str) else list(self.figis)
        normalized = [str(f).strip().upper() for f in raw if str(f).strip()]
        return [f for f in normalized if f not in {"NAN", "NONE"}]

    def run(self) -> pd.DataFrame:
        """Run load+transform workflow and return transformed dataframe.

        Formatting is intentionally kept external so callers can decide when and
        how to materialize project-specific output shapes.
        """
        class_name = self.__class__.__name__
        log(f"{class_name}: starting run()", type="info")
        self.raw_data = self.load()
        log(f"{class_name}: loaded {len(self.raw_data)} raw rows", type="info")
        self.transformed_data = self.transform(self.raw_data)
        log(
            f"{class_name}: transformed to {len(self.transformed_data)} rows",
            type="info",
        )
        self.formatted_data = {}
        log(f"{class_name}: run() completed", type="info")
        return self.transformed_data
