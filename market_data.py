# market_data.py
from kite_client import get_kite_client
from datetime import datetime, timedelta
from const import *
import pandas as pd

kite = get_kite_client()

def get_ltp(symbols):
    data = kite.ltp(symbols)
    return {s: data[s]['last_price'] for s in symbols if s in data}

def get_ltp2(symbols):
    return kite.ltp(symbols)

def get_ltp_of_symbol(underlying_symbol):
    ltp_dict = get_ltp([underlying_symbol])
    return ltp_dict.get(underlying_symbol)

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

    data = kite.historical_data(
        instrument.instrument_token,
        from_date=from_date,
        to_date=end_time,
        interval=interval.value,
        continuous=False
    )

    return pd.DataFrame(data)

