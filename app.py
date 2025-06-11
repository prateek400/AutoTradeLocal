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
from datetime import datetime, timedelta

import logging
from pprint import pprint
from trend_detector.trend_backtest import *
import json


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

# start  = datetime.datetime.now()
# for inst in Instrument:
#     logger.info(f"for instrument {inst.name} trend is {json.dumps(get_trend_breakdown(inst.value), indent=4)}")

# logger.info(f'time req -> {datetime.datetime.now() - start}')


# Simple usage
# backtester = TrendDetectorBacktester(confidence_threshold=0.1)
# results = backtester.run_backtest(
#     instrument=Instrument.NIFTY.value,
#     start_date=datetime(2024, 1, 1),
#     end_date=datetime(2025, 6, 1),
#     interval=CandleInterval.DAY
# )

# # Get summary for 5-day forward period
# summary = backtester.calculate_metrics(results[5])
# backtester.print_summary(summary, 5)

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 6, 1)


backtest_results = run_comprehensive_backtest(
        instrument=Instrument.NIFTY.value,
        start_date=start_date,
        end_date=end_date,
        intervals=[CandleInterval.DAY]
    )
    
# print("\n" + "="*80)
# print("BACKTEST COMPLETED")
# print("="*80)
# print("Results stored in backtest_results dictionary")
# print("Keys:", list(backtest_results.keys()))


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