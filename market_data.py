# market_data.py
from kite_client import get_kite_client
from datetime import datetime, timedelta
from const import *
import pandas as pd
from typing import Tuple, Dict

kite = get_kite_client()

def get_ltp(symbols):
    data = kite.ltp(symbols)
    return {s: data[s]['last_price'] for s in symbols if s in data}

def get_ltp2(symbols):
    return kite.ltp(symbols)

def get_ltp_of_symbol(underlying_symbol):
    ltp_dict = get_ltp([underlying_symbol])
    return ltp_dict.get(underlying_symbol)

# Cache structure: key = (instrument_token, interval), value = list of (start_time, end_time, DataFrame)
ohlc_cache: Dict[Tuple[int, str], list] = {}


INTERVAL_LIMITS = {
    CandleInterval.MIN_1: 30,
    CandleInterval.MIN_3: 60,
    CandleInterval.MIN_5: 60,
    CandleInterval.MIN_10: 60,
    CandleInterval.MIN_15: 60,
    CandleInterval.MIN_30: 60,
    CandleInterval.MIN_60: 60,
    CandleInterval.DAY: 2000,
}

def fetch_max_allowed_ohlc(instrument, interval: CandleInterval) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=INTERVAL_LIMITS[interval])
    return fetch_ohlc_between_given_time(instrument, start_date, end_date, interval)
    

def fetch_ohlc_between_given_time(instrument: BasicInstrumentDetails,
                                  start_time: datetime, end_time: datetime,
                                  interval: CandleInterval) -> pd.DataFrame:
    key = (instrument.instrument_token, interval.value)
        # Check cache
    # if key in ohlc_cache:
    #     for cached_start, cached_end, cached_df in ohlc_cache[key]:
    #         if cached_start <= start_time and cached_end >= end_time:
    #             # Slice and return only the required interval
    #             return cached_df[(cached_df['date'] >= start_time) & (cached_df['date'] <= end_time)].reset_index(drop=True)

    data = kite.historical_data(
        instrument.instrument_token,
        from_date=start_time,
        to_date=end_time,
        interval=interval.value,
        continuous=False
    )
    df = pd.DataFrame(data)
    # Save to cache
    ohlc_cache.setdefault(key, []).append((start_time, end_time, df.copy()))

    return df


def fetch_ohlc(instrument: BasicInstrumentDetails, end_time: datetime, interval: CandleInterval, lookback_candles: int = 1000) -> pd.DataFrame:
    interval_map = {
        CandleInterval.MIN_1: timedelta(minutes=1),
        CandleInterval.MIN_3: timedelta(minutes=3),
        CandleInterval.MIN_5: timedelta(minutes=5),
        CandleInterval.MIN_10: timedelta(minutes=10),
        CandleInterval.MIN_15: timedelta(minutes=15),
        CandleInterval.MIN_30: timedelta(minutes=30),
        CandleInterval.MIN_60: timedelta(minutes=60),
        CandleInterval.DAY: timedelta(days=1),
    }

    delta = interval_map[interval] * lookback_candles
    from_date = (end_time - delta)

    return fetch_ohlc_between_given_time(instrument, from_date, end_time, interval)

    # data = kite.historical_data(
    #     instrument.instrument_token,
    #     from_date=from_date,
    #     to_date=end_time,
    #     interval=interval.value,
    #     continuous=False
    # )

    # return pd.DataFrame(data)

