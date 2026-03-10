from __future__ import annotations

"""OpenFIGI API data source client."""

import time
from typing import Any

import pandas as pd
import requests

from utils.logging import log


class OpenFigiDataSource:
    """Stateful OpenFIGI client for security master enrichment."""

    OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
    LOCATION_MARKET_HINTS: dict[str, list[dict[str, str]]] = {
        "united states": [
            {"micCode": "XNYS"},
            {"micCode": "XNAS"},
            {"exchCode": "US"},
        ],
        "united states of america": [
            {"micCode": "XNYS"},
            {"micCode": "XNAS"},
            {"exchCode": "US"},
        ],
        "usa": [
            {"micCode": "XNYS"},
            {"micCode": "XNAS"},
            {"exchCode": "US"},
        ],
        "us": [
            {"micCode": "XNYS"},
            {"micCode": "XNAS"},
            {"exchCode": "US"},
        ],
        "canada": [
            {"micCode": "XTSE"},
            {"exchCode": "CN"},
            {"exchCode": "CA"},
        ],
        "netherlands": [
            {"micCode": "XAMS"},
            {"exchCode": "NA"},
        ],
        "united kingdom": [
            {"micCode": "XLON"},
            {"exchCode": "LN"},
        ],
        "france": [
            {"micCode": "XPAR"},
            {"exchCode": "FP"},
        ],
        "germany": [
            {"micCode": "XETR"},
            {"exchCode": "GY"},
        ],
        "switzerland": [
            {"micCode": "XSWX"},
            {"exchCode": "SE"},
        ],
        "japan": [
            {"micCode": "XTKS"},
            {"exchCode": "JP"},
        ],
        "hong kong": [
            {"micCode": "XHKG"},
            {"exchCode": "HK"},
        ],
        "australia": [
            {"micCode": "XASX"},
            {"exchCode": "AU"},
        ],
    }
    REQUEST_PAUSE_SECONDS = 0.25
    RATE_LIMIT_WAIT_SECONDS = 60

    def __init__(
        self,
        *,
        universe_df: pd.DataFrame,
        api_key: str | None = None,
    ) -> None:
        """Initialize OpenFIGI data source state."""
        self.universe_df = universe_df.copy()
        self.api_key = api_key
        self.last_response: dict[str, Any] | None = None

    def _candidate_market_hints(
        self,
        *,
        location: Any,
    ) -> list[dict[str, str]]:
        """Return ordered OpenFIGI market hints for ticker disambiguation."""
        hints: list[dict[str, str]] = []
        normalized_location = (
            str(location).strip().lower()
            if location is not None and not pd.isna(location)
            else ""
        )

        if normalized_location in self.LOCATION_MARKET_HINTS:
            hints.extend(self.LOCATION_MARKET_HINTS[normalized_location])

        # Final fallback with no location/exchange hint.
        hints.append({})

        deduped: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for hint in hints:
            key = tuple(sorted(hint.items()))
            if key not in seen:
                deduped.append(hint)
                seen.add(key)
        return deduped

    @staticmethod
    def _is_us_location(location: Any) -> bool:
        """Return True when location maps to the United States."""
        if location is None or pd.isna(location):
            return False
        text = str(location).strip().lower()
        return text in {"united states", "united states of america", "usa", "us"}

    @staticmethod
    def _candidate_id_values(ticker: str, *, is_us: bool) -> list[str]:
        """Return ordered ticker variants for OpenFIGI symbol matching.

        OpenFIGI often expects share-class tickers with ``/`` or ``.`` where
        upstream sources may use ``-`` (for example, ``BRK-B``).
        """
        base = str(ticker).strip().upper()
        ticker_root = base.split(".")[0]

        variants = [base] if is_us else [ticker_root, base]
        if "-" in base:
            variants.extend(
                [
                    base.replace("-", "/"),
                    base.replace("-", "."),
                ]
            )
        if "-" in ticker_root:
            variants.extend(
                [
                    ticker_root.replace("-", "/"),
                    ticker_root.replace("-", "."),
                ]
            )
        deduped: list[str] = []
        seen: set[str] = set()
        for value in variants:
            if value and value not in seen:
                deduped.append(value)
                seen.add(value)
        return deduped

    def _build_universe(self) -> pd.DataFrame:
        """Validate and normalize input universe."""
        universe = self.universe_df.loc[
            :, ~self.universe_df.columns.duplicated()
        ].copy()
        required = {"TICKER", "NAME", "LOCATION"}
        missing = sorted(required.difference(universe.columns))
        if missing:
            raise ValueError(
                "OpenFigiDataSource requires universe_df columns: "
                f"TICKER, NAME, LOCATION. Missing: {missing}"
            )

        universe["TICKER"] = universe["TICKER"].astype(str).str.strip().str.upper()
        universe = universe[universe["TICKER"] != ""].drop_duplicates(
            subset=["TICKER", "NAME", "LOCATION"], keep="first"
        )
        return universe[["TICKER", "NAME", "LOCATION"]].reset_index(drop=True)

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True when an exception indicates OpenFIGI rate limiting."""
        message = str(exc).lower()
        markers = (
            "openfigi rate limit hit",
            "too many requests",
            "429",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _empty_mapping() -> dict[str, Any]:
        """Return an empty OpenFIGI mapping payload."""
        return {
            "FIGI": None,
            "COMPOSITE_FIGI": None,
            "SHARE_CLASS_FIGI": None,
            "OPENFIGI_TICKER": None,
            "OPENFIGI_NAME": None,
            "OPENFIGI_EXCH_CODE": None,
            "OPENFIGI_SECURITY_TYPE": None,
            "OPENFIGI_SECURITY_TYPE2": None,
            "OPENFIGI_MARKET_SECTOR": None,
        }

    def _parse_mapping(self, response_item: dict[str, Any]) -> dict[str, Any]:
        """Parse first valid mapping from one OpenFIGI response item."""
        data = response_item.get("data", []) if isinstance(response_item, dict) else []
        first = data[0] if data else {}
        if not first:
            return self._empty_mapping()
        return {
            "FIGI": first.get("figi"),
            "COMPOSITE_FIGI": first.get("compositeFIGI"),
            "SHARE_CLASS_FIGI": first.get("shareClassFIGI"),
            "OPENFIGI_TICKER": first.get("ticker"),
            "OPENFIGI_NAME": first.get("name"),
            "OPENFIGI_EXCH_CODE": first.get("exchCode"),
            "OPENFIGI_SECURITY_TYPE": first.get("securityType"),
            "OPENFIGI_SECURITY_TYPE2": first.get("securityType2"),
            "OPENFIGI_MARKET_SECTOR": first.get("marketSector"),
        }

    def _post_mapping_job(self, headers: dict[str, str], job: dict[str, Any]) -> dict:
        """Submit one OpenFIGI mapping job and return first response item."""
        response = requests.post(
            self.OPENFIGI_URL,
            headers=headers,
            json=[job],
            timeout=30,
        )
        if response.status_code == 429:
            raise RuntimeError("OpenFIGI rate limit hit")
        response.raise_for_status()
        payload = response.json()
        self.last_response = payload[0] if payload else {}
        time.sleep(self.REQUEST_PAUSE_SECONDS)
        return self.last_response

    def get_security_master(self) -> pd.DataFrame:
        """Pull OpenFIGI mappings for all tickers in the universe."""
        universe = self._build_universe()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-OPENFIGI-APIKEY"] = self.api_key

        total = len(universe)
        output_rows: list[dict[str, Any]] = []
        for index, row in universe.iterrows():
            ticker = str(row["TICKER"])
            location = row.get("LOCATION", None)
            is_us = self._is_us_location(location)
            candidate_id_values = self._candidate_id_values(ticker, is_us=is_us)
            market_hints = self._candidate_market_hints(
                location=location,
            )
            parsed = None

            while True:
                rate_limited = False
                for id_value in candidate_id_values:
                    for market_hint in market_hints:
                        job: dict[str, Any] = {
                            "idType": "TICKER",
                            "idValue": id_value,
                            "marketSecDes": "Equity",
                        }
                        job.update(market_hint)

                        try:
                            response_item = self._post_mapping_job(headers, job)
                            parsed = self._parse_mapping(response_item)
                        except Exception as exc:  # noqa: BLE001
                            if self._is_rate_limit_error(exc):
                                rate_limited = True
                                break
                            continue

                        if parsed.get("FIGI"):
                            break
                    if rate_limited or (parsed and parsed.get("FIGI")):
                        break

                if not rate_limited:
                    break

                time.sleep(self.RATE_LIMIT_WAIT_SECONDS)
                completed = index
                remaining = total - completed
                log(
                    f"{completed}/{total} tickers completed for OpenFIGI. "
                    f"Rate limit detected; retrying after "
                    f"{self.RATE_LIMIT_WAIT_SECONDS} seconds. "
                    f"{remaining} tickers remaining.",
                    type="warning",
                )
                log("OpenFIGI wait complete. Retrying current ticker now.", type="info")

            final_parsed = parsed if parsed is not None else self._empty_mapping()
            if not final_parsed.get("FIGI"):
                log(
                    f"OpenFIGI missing mapping data for ticker: {ticker}",
                    type="warning",
                )
                continue
            output_rows.append(
                {
                    "TICKER": row["TICKER"],
                    "NAME": (
                        final_parsed.get("OPENFIGI_NAME")
                        if (not is_us and final_parsed.get("OPENFIGI_NAME"))
                        else row["NAME"]
                    ),
                    "LOCATION": row["LOCATION"],
                    **final_parsed,
                }
            )

        out = pd.DataFrame(output_rows)
        if out.empty:
            out = pd.DataFrame(
                columns=[
                    "TICKER",
                    "NAME",
                    "LOCATION",
                    "FIGI",
                    "COMPOSITE_FIGI",
                    "SHARE_CLASS_FIGI",
                    "OPENFIGI_TICKER",
                    "OPENFIGI_NAME",
                    "OPENFIGI_EXCH_CODE",
                    "OPENFIGI_SECURITY_TYPE",
                    "OPENFIGI_SECURITY_TYPE2",
                    "OPENFIGI_MARKET_SECTOR",
                ]
            )
        out["DATE"] = pd.Timestamp.today().date()
        cols = ["DATE"] + [column for column in out.columns if column != "DATE"]
        return out[cols]


default_open_figi_data_source = OpenFigiDataSource(
    universe_df=pd.DataFrame(columns=["TICKER", "NAME", "LOCATION"])
)
