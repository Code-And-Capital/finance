from unittest.mock import patch

import pandas as pd

from connectors.open_figi_data_source import OpenFigiDataSource


def test_build_universe_normalizes_and_deduplicates():
    source = OpenFigiDataSource(
        universe_df=pd.DataFrame(
            {
                "TICKER": [" aapl ", "AAPL", "msft"],
                "NAME": ["Apple", "Apple", "Microsoft"],
                "LOCATION": ["United States", "United States", "United States"],
            }
        )
    )

    universe = source._build_universe()

    assert list(universe.columns) == ["TICKER", "NAME", "LOCATION"]
    assert set(universe["TICKER"]) == {"AAPL", "MSFT"}
    assert len(universe) == 2


def test_get_security_master_skips_missing_and_retries_rate_limit():
    source = OpenFigiDataSource(
        universe_df=pd.DataFrame(
            {
                "TICKER": ["AAPL", "MSFT"],
                "NAME": ["Apple", "Microsoft"],
                "LOCATION": ["United States", "United States"],
            }
        )
    )
    post_calls = {"count": 0}

    def fake_post(_headers, job):
        post_calls["count"] += 1
        if post_calls["count"] == 1:
            raise RuntimeError("OpenFIGI rate limit hit")
        if job["idValue"] == "AAPL":
            return {"data": [{"figi": "FIGI_AAPL", "name": "Apple Inc"}]}
        return {"data": []}

    source._post_mapping_job = fake_post  # type: ignore[method-assign]

    with patch("connectors.open_figi_data_source.time.sleep") as mock_sleep:
        out = source.get_security_master()

    assert len(out) == 1
    assert out["TICKER"].iloc[0] == "AAPL"
    assert out["FIGI"].iloc[0] == "FIGI_AAPL"
    assert list(out.columns)[0] == "DATE"
    mock_sleep.assert_any_call(source.RATE_LIMIT_WAIT_SECONDS)


def test_get_security_master_overwrites_name_for_international():
    source = OpenFigiDataSource(
        universe_df=pd.DataFrame(
            {
                "TICKER": ["CSU.TO"],
                "NAME": ["Constellation Software"],
                "LOCATION": ["Canada"],
            }
        )
    )
    source._post_mapping_job = lambda _headers, _job: {  # type: ignore[method-assign]
        "data": [{"figi": "FIGI_CSU", "name": "CONSTELLATION SOFTWARE INC"}]
    }

    out = source.get_security_master()

    assert len(out) == 1
    assert out["NAME"].iloc[0] == "CONSTELLATION SOFTWARE INC"
    assert out["FIGI"].iloc[0] == "FIGI_CSU"
