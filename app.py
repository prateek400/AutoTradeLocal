# docker build -t autotrade-local . && docker run --rm -v "$PWD/output:/app/output" autotrade-local

from kite_client import get_kite_client, generate_access_token
from config import API_KEY, API_SECRET
from strategys.iron_condor.strike_selector import get_strike_prices, get_strikes_for_instrument
from market_data import get_ltp_of_symbol
from iv_fetch import find_instrument_token, construct_option_tradingsymbol
import datetime

instrument = "NIFTY"

# print(f"for instument {instrument} value is {get_ltp_of_symbol(instrument)}")

kite = get_kite_client()

# instruments = kite.instruments("NFO")

# print(instruments)
# strikes = get_strikes_for_instrument(
#     iv=14,
#     days_to_expiry=5,
#     instrument=instrument
# )

# print(strikes)

print(find_instrument_token("NIFTY",24700, datetime.date(2025, 6, 12), "PE"))


# print(generate_access_token())


# kite = get_kite_client()

# print(kite.profile())


# df = get_instruments()

# df = df[(df["segment"] == "NSE") & (df["instrument_type"] == "EQ")]

# df.to_csv("output/inst_nse_eq.csv")