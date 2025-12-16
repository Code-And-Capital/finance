import pandas as pd
import pytest

from utils.geo import (
    normalize_zip,
    build_full_address,
    build_fallback_address,
    build_city_country_address,
    lookup_cache,
    geocode_dataframe,
)

# ───────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────


@pytest.fixture
def sample_row():
    return pd.Series(
        {
            "ADDRESS1": "1 Apple Park Way",
            "CITY": "Cupertino",
            "STATE": "CA",
            "ZIP": "95014-2083",
            "COUNTRY": "United States",
        }
    )


@pytest.fixture
def cache_df():
    return pd.DataFrame(
        [
            {
                "ADDRESS1": "1 Apple Park Way",
                "CITY": "Cupertino",
                "COUNTRY": "United States",
                "LAT": 37.3349,
                "LON": -122.0090,
            }
        ]
    )


# ───────────────────────────────────────────────────────────────
# Helper tests
# ───────────────────────────────────────────────────────────────


def test_normalize_zip_strips_suffix():
    assert normalize_zip("12345-6789") == "12345"
    assert normalize_zip("12345") == "12345"
    assert normalize_zip(None) is None


# ───────────────────────────────────────────────────────────────
# Address builder tests
# ───────────────────────────────────────────────────────────────


def test_build_full_address_us(sample_row):
    address = build_full_address(sample_row)
    assert "Cupertino" in address
    assert "95014" in address
    assert "United States" in address


def test_build_fallback_address(sample_row):
    address = build_fallback_address(sample_row)
    assert address.startswith("Cupertino")
    assert "United States" in address


def test_build_city_country_address(sample_row):
    address = build_city_country_address(sample_row)
    assert address == "Cupertino, United States"


def test_build_city_country_missing_city():
    row = pd.Series({"COUNTRY": "United States"})
    assert build_city_country_address(row) is None


# ───────────────────────────────────────────────────────────────
# Cache lookup tests (IMPORTANT CHANGES)
# ───────────────────────────────────────────────────────────────


def test_lookup_cache_hit(sample_row, cache_df):
    lat, lon = lookup_cache(sample_row, cache_df)
    assert lat == 37.3349
    assert lon == -122.0090


def test_lookup_cache_miss(sample_row):
    empty_cache = pd.DataFrame(columns=["ADDRESS1", "CITY", "COUNTRY", "LAT", "LON"])
    lat, lon = lookup_cache(sample_row, empty_cache)
    assert lat is None
    assert lon is None


def test_lookup_cache_missing_columns(sample_row):
    bad_cache = pd.DataFrame([{"foo": "bar"}])
    lat, lon = lookup_cache(sample_row, bad_cache)
    assert lat is None
    assert lon is None


# ───────────────────────────────────────────────────────────────
# Geocode dataframe tests (NO REAL API CALLS)
# ───────────────────────────────────────────────────────────────


class DummyLocation:
    latitude = 10.0
    longitude = 20.0


class DummyGeocoder:
    def geocode(self, address):
        return DummyLocation()


@pytest.fixture
def monkeypatched_geocoder(monkeypatch):
    monkeypatch.setattr(
        "utils.geo.Nominatim",
        lambda user_agent=None: DummyGeocoder(),
    )


def test_geocode_dataframe_uses_cache_first(
    monkeypatched_geocoder, sample_row, cache_df
):
    df = pd.DataFrame([sample_row])

    result = geocode_dataframe(
        df,
        cache_df=cache_df,
        delay=0,
    )

    assert result.loc[0, "LAT"] == 37.3349
    assert result.loc[0, "LON"] == -122.0090


def test_geocode_dataframe_falls_back_to_geocoder(monkeypatched_geocoder, sample_row):
    df = pd.DataFrame([sample_row])

    result = geocode_dataframe(
        df,
        cache_df=None,
        delay=0,
    )

    assert result.loc[0, "LAT"] == 10.0
    assert result.loc[0, "LON"] == 20.0
