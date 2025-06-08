# market_data.py
from kite_client import get_kite_client

def get_ltp(symbols):
    kite = get_kite_client()
    data = kite.ltp(symbols)
    return {s: data[s]['last_price'] for s in symbols}
