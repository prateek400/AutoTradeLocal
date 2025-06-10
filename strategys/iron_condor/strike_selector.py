import math
from market_data import get_ltp_of_symbol
from const import *

def __round_to_nearest(x, base):
    return int(round(x / base) * base)

def __get_expected_move(iv, days_to_expiry, spot_price):
    """
    Calculate 1 standard deviation expected move
    """
    annual_iv = iv / 100  # Convert % to decimal
    move = spot_price * annual_iv * math.sqrt(days_to_expiry / 365)
    return move

def __is_iv_acceptable(iv, min_iv=10, max_iv=20):
    return min_iv <= iv <= max_iv

def get_strikes_for_instrument(iv: float, days_to_expiry: int, instrument: OptionInstrumentDetails):
    instrument_name = instrument.exchange + ":" + instrument.kite_trading_symbol
    min_iv = instrument.iron_condor_min_iv
    max_iv = instrument.iron_condor_max_iv
    strike_step = instrument.strike_step
    spread_width = instrument.spread_width
    return _get_strike_prices(get_ltp_of_symbol(instrument_name), iv, days_to_expiry, strike_step, spread_width, min_iv, max_iv)

def _get_strike_prices(spot_price, iv, days_to_expiry, strike_step=50, spread_width=100, min_iv=10, max_iv=20):
    """
    Calculate iron condor strikes, instrument independent
    
    Parameters:
    - spot_price (float): current underlying price
    - iv (float): implied volatility in %
    - days_to_expiry (int): calendar days to expiry
    - strike_step (int): tick size / strike price increment (e.g. 50, 5, 1)
    - spread_width (int): width between short and long strikes
    - min_iv (float): minimum IV threshold to trade
    - max_iv (float): maximum IV threshold to trade

    Returns:
    - dict or None if IV filter blocks trade
    """
    if not __is_iv_acceptable(iv, min_iv, max_iv):
        print(f"[IV Filter] Skipping trade: IV={iv} outside range {min_iv}-{max_iv}")
        return None

    expected_move = __get_expected_move(iv, days_to_expiry, spot_price)
    expected_move_rounded = __round_to_nearest(expected_move, strike_step)
    spot_rounded = __round_to_nearest(spot_price, strike_step)

    # Short strikes at ± expected move
    short_put = spot_rounded - expected_move_rounded
    short_call = spot_rounded + expected_move_rounded

    # Long strikes spaced by spread_width
    long_put = short_put - spread_width
    long_call = short_call + spread_width

    return {
        "spot_price": spot_price,
        "long_put": long_put,
        "short_put": short_put,
        "short_call": short_call,
        "long_call": long_call
    }
