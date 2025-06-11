# docker build -t autotrade-local . && docker run --rm -v "$PWD/output:/app/output" autotrade-local

from kite_client import get_kite_client, generate_access_token
from config import API_KEY, API_SECRET
from strategys.iron_condor.strike_selector import *
from market_data import *
from nsepython import *
from nse_lib.nse_python_iv_cal import *
from const import *
from instrument_helper import *
from trend_detector.trend_detector import *
import datetime
import logging


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# print(f"for instument {instrument} value is {get_ltp_of_symbol(instrument)}")

# kite = get_kite_client()

start = time.time()
# print(get_option_iv("nifty", 25000, OptionType.PE, '12-Jun-2025'))
end = time.time()


# print(indices)

# instruments = kite.instruments("NFO")
# print(instruments)

# logger.info(fetch_ohlc(Instrument.NIFTY.value, datetime.datetime.now(), CandleInterval.DAY))

for inst in Instrument:
    logger.info(f"for instrument {inst.name} trend is {detect_final_trend(inst.value)}")

# logger.info(fetch_ohlc(Instrument.NIFTY.value, datetime.datetime.now(), CandleInterval.DAY))
# logger.info(f'current trend is: {detect_final_trend(Instrument.BANKNIFTY.value, datetime.datetime.now(), CandleInterval.DAY)}')

# trends = detect_trend_at(Instrument.NIFTY.value, datetime.datetime.now(), CandleInterval.DAY)

# for strategy_name, trend in trends.items():
#     logger.info(f'  {strategy_name} ->  {trend}')

# logger.info(detect_trend_at(Instrument.NIFTY.value, datetime.datetime.now(), CandleInterval.DAY))

# for inst in Instrument:
#     strikes = get_strikes_for_iron_condor(inst.value)
#     logger.info(strikes)

# print(find_instrument_token("NIFTY",24700, datetime.date(2025, 6, 12), "PE"))


# print(generate_access_token())


# kite = get_kite_client()

# print(kite.profile())


# df = get_instruments()

# df = df[(df["segment"] == "NSE") & (df["instrument_type"] == "EQ")]

# df.to_csv("output/inst_nse_eq.csv")