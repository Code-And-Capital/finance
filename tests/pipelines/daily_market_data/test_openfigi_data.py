from unittest.mock import MagicMock

import pandas as pd

from pipelines.daily_market_data.openfigi_data import OpenFigiData


def test_filter_existing_universe_us_by_ticker_name_and_intl_by_ticker_country():
    pipeline = OpenFigiData(
        universe_df=pd.DataFrame(
            {
                "TICKER": ["AAPL", "MSFT", "CSU.TO", "HEIA.AS"],
                "NAME": ["Apple Inc", "Microsoft Corp", "Constellation", "Heineken"],
                "LOCATION": ["United States", "United States", "Canada", "Netherlands"],
            }
        )
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        return_value=pd.DataFrame(
            {
                "TICKER": ["AAPL", "CSU.TO"],
                "NAME": ["Apple Inc", "Old Name"],
                "COUNTRY": ["United States", "Canada"],
            }
        )
    )

    out = pipeline._filter_existing_universe(configs_path=None)

    assert set(out["TICKER"]) == {"MSFT", "HEIA.AS"}


def test_run_skips_when_no_rows_remaining_after_filter():
    pipeline = OpenFigiData(
        universe_df=pd.DataFrame(
            {"TICKER": ["AAPL"], "NAME": ["Apple"], "LOCATION": ["United States"]}
        )
    )
    pipeline._filter_existing_universe = MagicMock(return_value=pd.DataFrame())
    pipeline._pull_generic = MagicMock()

    out = pipeline.run()

    assert out.empty
    pipeline._pull_generic.assert_not_called()


def test_run_write_to_azure_writes_security_master():
    pipeline = OpenFigiData(
        universe_df=pd.DataFrame(
            {"TICKER": ["AAPL"], "NAME": ["Apple"], "LOCATION": ["United States"]}
        )
    )
    pipeline._filter_existing_universe = MagicMock(return_value=pipeline.universe_df)
    pipeline._pull_generic = MagicMock(
        return_value=pd.DataFrame(
            {
                "DATE": [pd.Timestamp("2026-03-08").date()],
                "TICKER": ["AAPL"],
                "NAME": ["Apple"],
                "LOCATION": ["United States"],
                "FIGI": ["FIGI_AAPL"],
            }
        )
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    out = pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    assert len(out) == 1
    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "security_master"
    assert kwargs["overwrite"] is False
