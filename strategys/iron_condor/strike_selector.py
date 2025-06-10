import math
import datetime
from nse_lib.nse_python_iv_cal import *
from market_data import get_ltp_of_symbol
from const import OptionInstrumentDetails, OptionType
import logging
logger = logging.getLogger(__name__)

# ---- Helpers ---- #
def __round_to_nearest(x: float, base: int) -> int:
    return int(round(x / base) * base)

def __get_expected_move(iv_percent: float, days_to_expiry: int, spot_price: float) -> float:
    """
    Calculate 1 standard deviation move using Black-Scholes approximation.
    """
    annual_iv = iv_percent / 100
    return spot_price * annual_iv * math.sqrt(days_to_expiry / 365)

def __is_iv_acceptable(iv: float, min_iv: float, max_iv: float) -> bool:
    return min_iv <= iv <= max_iv


# ---- Core Logic ---- #
def get_best_iron_condor_expiry(instrument: OptionInstrumentDetails) -> Optional[date]:
    """
    Return the best expiry date for Iron Condor strategy.
    Prefers expiry with 7–10 DTE, otherwise closest future expiry.
    """
    today = datetime.date.today()
    expiries = get_expiry_dates(instrument.nse_lib_symbol)
    future_expiries = [d for d in expiries if d > today]

    if not future_expiries:
        logger.error(f"[ERROR] No future expiries available for {instrument.nse_lib_symbol}")
        return None

    ideal_expiries = [d for d in future_expiries if 7 <= (d - today).days <= 10]
    if ideal_expiries:
        return ideal_expiries[0]

    # fallback
    return future_expiries[0]


def get_strikes_for_iron_condor(instrument: OptionInstrumentDetails) -> Optional[dict]:
    """
    Generates Iron Condor strike levels for given instrument.
    Returns None if IV filter blocks trade or data is missing.
    """
    instrument_name = f"{instrument.exchange}:{instrument.kite_trading_symbol}"
    spot_price = get_ltp_of_symbol(instrument_name)
    expiry_date = get_best_iron_condor_expiry(instrument)

    if expiry_date is None:
        logger.error(f"[ERROR] Could not determine expiry date for {instrument.kite_trading_symbol}")
        return None

    spot_rounded = __round_to_nearest(spot_price, instrument.strike_step)
    days_to_expiry = (expiry_date - datetime.date.today()).days

    try:
        iv = get_option_iv(
            instrument.nse_lib_symbol,
            spot_rounded,
            OptionType.CE,
            expiry_date
        )
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch IV: {e}")
        return None

    if not __is_iv_acceptable(iv, instrument.iron_condor_min_iv, instrument.iron_condor_max_iv):
        logger.warning(f"for instument {instrument_name} with expiry {expiry_date} [IV Filter] IV={iv:.2f}% not in range {instrument.iron_condor_min_iv}-{instrument.iron_condor_max_iv}")
        return None

    expected_move = __get_expected_move(iv, days_to_expiry, spot_price)
    expected_move_rounded = __round_to_nearest(expected_move, instrument.strike_step)

    short_put = spot_rounded - expected_move_rounded
    short_call = spot_rounded + expected_move_rounded
    long_put = short_put - instrument.spread_width
    long_call = short_call + instrument.spread_width

    return {
        "underlying": instrument.kite_trading_symbol,
        "expiry_date": expiry_date,
        "spot_price": spot_price,
        "IV": iv,
        "expected_move": expected_move_rounded,
        "long_put": long_put,
        "short_put": short_put,
        "short_call": short_call,
        "long_call": long_call
    }
