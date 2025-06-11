# market_data.py
from kite_client import get_kite_client

def get_ltp(symbols):
    kite = get_kite_client()
    data = kite.ltp(symbols)
    return {s: data[s]['last_price'] for s in symbols if s in data}

def get_ltp_of_symbol(underlying_symbol):
    ltp_dict = get_ltp([underlying_symbol])
    return ltp_dict.get(underlying_symbol)