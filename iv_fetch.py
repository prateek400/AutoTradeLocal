from kite_client import get_kite_client, get_kite_ticker
from instrument_helper import get_instruments
import logging

import datetime

kite = get_kite_client()


def find_instrument_token(underlying,strike_price, expiry, option_type):
    df = get_instruments()
    df = df[df["segment"] == 'NFO-OPT']
    df = df[df["tradingsymbol"] == construct_option_tradingsymbol(underlying, expiry, strike_price, option_type)]
    return df.iloc[0]["instrument_token"]

def construct_option_tradingsymbol(underlying, expiry_date, strike, option_type):
    """
    underlying: e.g. 'NIFTY'
    expiry_date: datetime.date object
    strike: int (e.g. 25000)
    option_type: 'CE' or 'PE'
    """
    year_short = str(expiry_date.year)[-2:]
    month = f"{expiry_date.month}"
    day = f"{expiry_date.day:02d}"
    return f"{underlying}{year_short}{month}{day}{strike}{option_type}"


token = find_instrument_token("NIFTY",24700, datetime.date(2025, 6, 12), "PE")

def on_ticks(ws, ticks):
    # Callback to receive ticks.
    print(f"Ticks: {ticks}")
    # logging.debug("Ticks: {}".format(ticks))

def on_connect(ws, response):
    # Callback on successful connect.
    # Subscribe to a list of instrument_tokens (RELIANCE and ACC here).
    t = int(token) #11482370
    ws.subscribe([t])

    ws.set_mode(ws.MODE_FULL, [t])

def on_close(ws, code, reason):
    # On connection close stop the event loop.
    # Reconnection will not happen after executing `ws.stop()`
    ws.stop()

kws = get_kite_ticker()
# Assign the callbacks.
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

kws.connect()