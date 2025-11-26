import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from datetime import date

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath("__file__")), ".."))
)

import data_loader.yahoo_finance as yahoo_finance
import utils.downloading_utils as downloading_utils
import utils.sql_utils as sql_utils
import utils.mapping as mapping
import utils.dataframe_utils as dataframe_utils

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)

########### Dowload Holdings Files ###########

etf_urls = mapping.etf_urls
data_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath("__file__")), "..")),
    "Data",
)

all_indices = pd.DataFrame()
for fund_name, url in etf_urls.items():
    print(fund_name)
    df_temp = downloading_utils.download_holdings(
        fund_name, url, download_folder=data_path
    )
    all_indices = pd.concat([all_indices, df_temp])

sql_utils.write_sql_table(
    database_name="CODE_CAPITAL", table_name="holdings", df=all_indices, overwrite=True
)

########### Store Company Data ###########


def find_missing_tickers(df, ticker_list):
    """
    Ensures all tickers in ticker_list appear in df.
    Adds missing tickers with today's date as start_date.
    """

    # Find tickers that are missing from the dataframe
    existing = set(df["TICKER"])
    missing = [t for t in ticker_list if t not in existing]

    return missing


query = """
SELECT DISTINCT TICKER
FROM company_info
WHERE DATE = (
    SELECT MAX(DATE)
    FROM company_info
)
"""
t = sql_utils.read_sql_table(query=query, database_name="CODE_CAPITAL")
missing = find_missing_tickers(t, all_indices["TICKER"].unique())


today = date.today()
if today.weekday() == 0:
    client = yahoo_finance.YahooDataClient(
        all_indices["TICKER"].unique(), max_workers=10
    )
    run = True
else:
    client = yahoo_finance.YahooDataClient(missing, max_workers=10)
    if missing:
        run = True
    else:
        run = False


if run:
    all_info = client.get_company_info()
    all_officers = client.get_officer_info()

    all_info = all_info.applymap(str)
    sql_utils.write_sql_table(
        table_name="company_info",
        database_name="CODE_CAPITAL",
        df=all_info,
        overwrite=False,
    )
    sql_utils.write_sql_table(
        table_name="officers",
        database_name="CODE_CAPITAL",
        df=all_officers,
        overwrite=False,
    )


def add_missing_tickers(df, ticker_list):
    """
    Ensures all tickers in ticker_list appear in df.
    Adds missing tickers with today's date as start_date.
    """

    # Find tickers that are missing from the dataframe
    existing = set(df["TICKER"])
    missing = [t for t in ticker_list if t not in existing]

    # Create rows for missing tickers
    if missing:
        new_rows = pd.DataFrame({"TICKER": missing, "START_DATE": "2000-01-01"})
        # Append to the original dataframe
        df = pd.concat([df, new_rows], ignore_index=True)

    return df


query = """
SELECT TICKER, MAX(DATE) AS START_DATE
FROM prices
GROUP BY TICKER
"""

max_dates = sql_utils.read_sql_table(query=query, database_name="CODE_CAPITAL")
max_dates["START_DATE"] = pd.to_datetime(max_dates["START_DATE"])
max_dates["START_DATE"] = max_dates["START_DATE"] + BDay(1)
# max_dates = dataframe_utils.df_to_dict(max_dates, "TICKER", "START_DATE")

start_date_mapping = dataframe_utils.df_to_dict(
    add_missing_tickers(max_dates, all_indices["TICKER"].unique()),
    "TICKER",
    "START_DATE",
)


client_prices = yahoo_finance.YahooDataClient(
    all_indices["TICKER"].unique(), max_workers=10
)

all_prices = client_prices.get_prices(start_date=start_date_mapping)

sql_utils.write_sql_table(
    table_name="prices", database_name="CODE_CAPITAL", df=all_prices, overwrite=False
)

if run:
    all_financial_annual = client.get_financials(
        annual=True, statement_type="financial"
    )
    all_financial_quarterly = client.get_financials(
        annual=False, statement_type="financial"
    )
    all_balancesheet_annual = client.get_financials(
        annual=True, statement_type="balance_sheet"
    )
    all_balancesheet_quarterly = client.get_financials(
        annual=False, statement_type="balance_sheet"
    )
    all_income_annual = client.get_financials(
        annual=True, statement_type="income_statement"
    )
    all_income_quarterly = client.get_financials(
        annual=False, statement_type="income_statement"
    )
    all_cashflow_annual = client.get_financials(annual=True, statement_type="cashflow")
    all_cashflow_quarterly = client.get_financials(
        annual=False, statement_type="cashflow"
    )

    sql_utils.write_sql_table(
        table_name="financial_annual",
        database_name="CODE_CAPITAL",
        df=all_financial_annual,
        overwrite=False,
    )
    sql_utils.write_sql_table(
        table_name="financial_quarterly",
        database_name="CODE_CAPITAL",
        df=all_financial_quarterly,
        overwrite=False,
    )
    sql_utils.write_sql_table(
        table_name="balancesheet_annual",
        database_name="CODE_CAPITAL",
        df=all_balancesheet_annual,
        overwrite=False,
    )
    sql_utils.write_sql_table(
        table_name="balancesheet_quarterly",
        database_name="CODE_CAPITAL",
        df=all_balancesheet_quarterly,
        overwrite=False,
    )
    sql_utils.write_sql_table(
        table_name="incomestatement_annual",
        database_name="CODE_CAPITAL",
        df=all_income_annual,
        overwrite=False,
    )
    sql_utils.write_sql_table(
        table_name="incomestatement_quarterly",
        database_name="CODE_CAPITAL",
        df=all_income_quarterly,
        overwrite=False,
    )
    sql_utils.write_sql_table(
        table_name="cashflow_annual",
        database_name="CODE_CAPITAL",
        df=all_cashflow_annual,
        overwrite=False,
    )
    sql_utils.write_sql_table(
        table_name="cashflow_quarterly",
        database_name="CODE_CAPITAL",
        df=all_cashflow_quarterly,
        overwrite=False,
    )
