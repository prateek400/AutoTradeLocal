import time
import datetime
from datetime import date
from typing import Optional, Dict, Tuple

import pandas as pd
from nsepython import *
from const import OptionType

# ---- Constants ---- #
_CACHE_TTL_SEC = 2

# ---- Internal cache ---- #
_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}

# ---- Helpers ---- #
def __to_nse_date_str(d: date) -> str:
    return d.strftime("%d-%b-%Y")  # Example: '13-Jun-2024'

def __from_nse_date_str(s: str) -> date:
    return datetime.datetime.strptime(s, "%d-%b-%Y").date()

def __parse_option_chain_to_df(resp: dict) -> pd.DataFrame:
    records = resp.get("records", {}).get("data", [])
    rows = []

    for rec in records:
        strike = rec["strikePrice"]
        expiry = rec["expiryDate"]

        if "CE" in rec:
            ce = rec["CE"]
            rows.append({
                "strikePrice": strike,
                "expiryDate": expiry,
                "optionType": "CE",
                "impliedVolatility": ce.get("impliedVolatility"),
                "lastPrice": ce.get("lastPrice"),
                "openInterest": ce.get("openInterest")
            })

        if "PE" in rec:
            pe = rec["PE"]
            rows.append({
                "strikePrice": strike,
                "expiryDate": expiry,
                "optionType": "PE",
                "impliedVolatility": pe.get("impliedVolatility"),
                "lastPrice": pe.get("lastPrice"),
                "openInterest": pe.get("openInterest")
            })

    return pd.DataFrame(rows)

def _get_option_chain_df_with_cache(symbol: str, expiry_date: date) -> pd.DataFrame:
    symbol = symbol.upper()
    expiry_str = __to_nse_date_str(expiry_date)
    cache_key = (symbol, expiry_str)
    now = time.time()

    if cache_key in _cache:
        cached_time, cached_df = _cache[cache_key]
        if now - cached_time < _CACHE_TTL_SEC:
            return cached_df

    raw_resp = nse_optionchain_scrapper(symbol)
    full_df = __parse_option_chain_to_df(raw_resp)

    # Cache per expiry
    for exp in full_df["expiryDate"].unique():
        exp_df = full_df[full_df["expiryDate"] == exp].copy()
        _cache[(symbol, exp)] = (now, exp_df)

    if cache_key not in _cache:
        raise ValueError(f"No data for {symbol} expiry {expiry_str}")

    return _cache[cache_key][1]

# ---- Public API ---- #
def get_option_iv(
    symbol: str,
    strike_price: int,
    option_type: OptionType,
    expiry_date: date
) -> float:
    """
    Fetch the Implied Volatility (IV) for a specific option.

    Args:
        symbol (str): e.g., 'NIFTY', 'BANKNIFTY'
        strike_price (int): Desired strike
        option_type (OptionType): OptionType.CE or OptionType.PE
        expiry_date (date): Python date object (e.g., datetime.date(2024, 6, 13))

    Returns:
        float: IV in %

    Raises:
        ValueError: If no matching IV found
    """
    expiry_str = __to_nse_date_str(expiry_date)
    df = _get_option_chain_df_with_cache(symbol, expiry_date)

    filtered = df[
        (df["strikePrice"] == strike_price) &
        (df["optionType"] == option_type.value)
    ]

    if not filtered.empty:
        iv = filtered.iloc[0]["impliedVolatility"]
        if iv is not None:
            return iv

    raise ValueError(f"IV not found for {symbol} {strike_price}{option_type.value} {expiry_str}")


def get_expiry_dates(symbol: str) -> list[date]:
    """
    Fetch and return sorted list of expiry dates as date objects for a given symbol.
    Uses cached option chain data if available.
    """
    symbol = symbol.upper()
    now = time.time()
    expiry_strs = set()

    # Try using any cached expiry
    for (cached_symbol, cached_expiry_str), (cached_time, _) in _cache.items():
        if cached_symbol == symbol and now - cached_time < _CACHE_TTL_SEC:
            expiry_strs.add(cached_expiry_str)

    if expiry_strs:
        return sorted(__from_nse_date_str(s) for s in expiry_strs)
    # No fresh cache found, fetch new data
    raw_resp = nse_optionchain_scrapper(symbol)
    full_df = __parse_option_chain_to_df(raw_resp)

    now = time.time()
    for exp in full_df["expiryDate"].unique():
        exp_df = full_df[full_df["expiryDate"] == exp].copy()
        _cache[(symbol, exp)] = (now, exp_df)

    expiry_strs = full_df["expiryDate"].unique().tolist()
    return sorted({__from_nse_date_str(s) for s in expiry_strs})
