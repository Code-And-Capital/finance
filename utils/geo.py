import time
import pandas as pd
from tqdm import tqdm
from geopy.geocoders import Nominatim

# ───────────────────────────────────────────────────────────────
# Address Patterns
# ───────────────────────────────────────────────────────────────

ADDRESS_PATTERNS = {
    "UNITED STATES": "{street}, {city}, {state}, {zip}, {country}",
    "SINGAPORE": "{street}, {city}, {zip}, {country}",
    "BERMUDA": "{street}, {city}, {zip}, {country}",
    "CANADA": "{street}, {city}, {state} {zip}, {country}",
    "UNITED KINGDOM": "{street}, {city}, {zip}, {country}",
    "IRELAND": "{street}, {city}, {zip}, {country}",
    "NETHERLANDS": "{street}, {zip} {city}, {country}",
    "SWITZERLAND": "{street}, {zip} {city}, {country}",
    "LUXEMBOURG": "{street}, {zip} {city}, {country}",
    "FINLAND": "{street}, {zip} {city}, {country}",
    "CAYMAN ISLANDS": "{street}, {city}, {zip}, {country}",
}


ADDRESS_FALLBACK_PATTERNS = {
    "UNITED STATES": "{city}, {state}, {zip}, {country}",
    "SINGAPORE": "{city}, {zip}, {country}",
    "BERMUDA": "{city}, {zip}, {country}",
    "CANADA": "{city}, {state}, {zip}, {country}",
    "UNITED KINGDOM": "{city}, {zip}, {country}",
    "IRELAND": "{city}, {zip}, {country}",
    "NETHERLANDS": "{zip} {city}, {country}",
    "SWITZERLAND": "{zip} {city}, {country}",
    "LUXEMBOURG": "{zip} {city}, {country}",
    "FINLAND": "{zip} {city}, {country}",
    "CAYMAN ISLANDS": "{city}, {zip}, {country}",
}

DEFAULT_ADDRESS_COLS = {
    "street": "ADDRESS1",
    "city": "CITY",
    "state": "STATE",
    "zip": "ZIP",
    "country": "COUNTRY",
}

# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────


def normalize_zip(zip_code):
    """Strip ZIP+4 suffixes if present."""
    if isinstance(zip_code, str) and "-" in zip_code:
        return zip_code.split("-")[0].strip()
    return zip_code


def try_geocode(geolocator, address, delay):
    """Attempt geocoding; return (lat, lon) or (None, None)."""
    if not address:
        return None, None

    try:
        location = geolocator.geocode(address)
        time.sleep(delay)
    except Exception:
        return None, None

    if location:
        return location.latitude, location.longitude

    return None, None


def lookup_cache(row, cache_df, address_cols=None):
    """
    Lookup cached coordinates using the same column mapping
    logic as the address builders.

    Parameters
    ----------
    row : pandas.Series
        Row containing address fields.

    cache_df : pandas.DataFrame
        Cache DataFrame containing address fields and coordinates.

    address_cols : dict, optional
        Column mapping for address components. Defaults to DEFAULT_ADDRESS_COLS.

    Returns
    -------
    (lat, lon) : tuple
        Cached latitude and longitude if found, otherwise (None, None).
    """
    if cache_df is None or cache_df.empty:
        return None, None

    address_cols = address_cols or DEFAULT_ADDRESS_COLS

    street = row.get(address_cols["street"], "")
    city = row.get(address_cols["city"], "")
    country = row.get(address_cols["country"], "")

    # Defensive: required cache columns
    required_cols = {
        address_cols["street"],
        address_cols["city"],
        address_cols["country"],
        "LAT",
        "LON",
    }
    if not required_cols.issubset(cache_df.columns):
        return None, None

    match = cache_df[
        (cache_df[address_cols["street"]] == street)
        & (cache_df[address_cols["city"]] == city)
        & (cache_df[address_cols["country"]] == country)
    ]

    if not match.empty:
        cached = match.iloc[0]
        return cached["LAT"], cached["LON"]

    return None, None


def append_to_cache(cache_df, address, city, country, lat, lon):
    """Append a new geocoded row to the cache DataFrame."""
    new_row = {
        "ADDRESS": address,
        "CITY": city,
        "COUNTRY": country,
        "LAT": lat,
        "LON": lon,
    }
    return pd.concat([cache_df, pd.DataFrame([new_row])], ignore_index=True)


# ───────────────────────────────────────────────────────────────
# Address Builders
# ───────────────────────────────────────────────────────────────


def build_full_address(row, address_cols=None):
    """Build a full, country-aware address string."""
    address_cols = address_cols or DEFAULT_ADDRESS_COLS

    street = row.get(address_cols["street"], "")
    city = row.get(address_cols["city"], "")
    state = row.get(address_cols["state"], "")
    zip_code = normalize_zip(row.get(address_cols["zip"], ""))
    country = row.get(address_cols["country"], "")

    pattern = ADDRESS_PATTERNS.get(
        str(country).upper().strip(),
        "{street}, {city}, {state}, {zip}, {country}",
    )

    return pattern.format(
        street=street,
        city=city,
        state=state,
        zip=zip_code,
        country=country,
    )


def build_fallback_address(row, address_cols=None):
    """Build a reduced-precision fallback address."""
    address_cols = address_cols or DEFAULT_ADDRESS_COLS

    city = row.get(address_cols["city"], "")
    state = row.get(address_cols["state"], "")
    zip_code = normalize_zip(row.get(address_cols["zip"], ""))
    country = row.get(address_cols["country"], "")

    pattern = ADDRESS_FALLBACK_PATTERNS.get(
        str(country).upper().strip(),
        "{city}, {zip}, {country}",
    )

    raw = pattern.format(
        city=city,
        state=state,
        zip=zip_code,
        country=country,
    )

    return ", ".join(part.strip() for part in raw.split(",") if part.strip())


def build_city_country_address(row, address_cols=None):
    """Build the lowest-precision address (city + country)."""
    address_cols = address_cols or DEFAULT_ADDRESS_COLS

    city = row.get(address_cols["city"])
    country = row.get(address_cols["country"])

    if not city or not country:
        return None

    return f"{city}, {country}"


# ───────────────────────────────────────────────────────────────
# Main Geocoding Function
# ───────────────────────────────────────────────────────────────


def geocode_dataframe(
    df: pd.DataFrame,
    address_cols: dict = None,
    cache_df: pd.DataFrame = None,
    user_agent: str = "geo_app",
    delay: float = 1.0,
):
    """
    Geocode a DataFrame using a DataFrame-backed cache and
    progressive fallback addresses.
    """
    address_cols = address_cols or DEFAULT_ADDRESS_COLS
    geolocator = Nominatim(user_agent=user_agent)

    if cache_df is None:
        cache_df = pd.DataFrame(columns=["ADDRESS", "CITY", "COUNTRY", "LAT", "LON"])

    builders = [
        build_full_address,
        build_fallback_address,
        build_city_country_address,
    ]

    lats, lons = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding"):
        lat = lon = None

        for builder in builders:
            full_address = builder(row, address_cols)

            lat, lon = lookup_cache(
                row=row, cache_df=cache_df, address_cols=address_cols
            )
            if lat is not None:
                break

            lat, lon = try_geocode(geolocator, full_address, delay)
            if lat is not None:
                break

        lats.append(lat)
        lons.append(lon)

    result = df.copy()
    result["LAT"] = lats
    result["LON"] = lons

    return result
