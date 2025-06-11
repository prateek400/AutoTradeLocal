from enum import Enum
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np
import talib
from sklearn.linear_model import LinearRegression
import datetime
from const import *
from market_data import fetch_ohlc

# ----------------------------------------
# Enum for Trend
# ----------------------------------------
class Trend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDE_WAY = "SIDE_WAY"
    NA = "NA"

# ----------------------------------------
# Abstract Base Strategy
# ----------------------------------------
class TrendStrategy(ABC):
    @abstractmethod
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> Trend:
        pass

# ----------------------------------------
# Strategy 1: HH/HL Price Action
# ----------------------------------------
class PriceActionStrategy(TrendStrategy):
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> Trend:
        if len(df) < 5:
            return Trend.NA

        highs = df['high'].tolist()
        lows = df['low'].tolist()

        hh = all(x < y for x, y in zip(highs, highs[1:]))
        hl = all(x < y for x, y in zip(lows, lows[1:]))
        lh = all(x > y for x, y in zip(highs, highs[1:]))
        ll = all(x > y for x, y in zip(lows, lows[1:]))

        if hh and hl:
            return Trend.BULLISH
        elif lh and ll:
            return Trend.BEARISH
        else:
            return Trend.SIDE_WAY

# ----------------------------------------
# Strategy 2: Moving Average Crossover
# ----------------------------------------
class MovingAverageStrategy(TrendStrategy):
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> Trend:
        if len(df) < 50:
            return Trend.NA

        ema20 = talib.EMA(df['close'], timeperiod=20)
        ema50 = talib.EMA(df['close'], timeperiod=50)

        if ema20.iloc[-1] > ema50.iloc[-1]:
            return Trend.BULLISH
        elif ema20.iloc[-1] < ema50.iloc[-1]:
            return Trend.BEARISH
        else:
            return Trend.SIDE_WAY

# ----------------------------------------
# Strategy 3: ADX Based
# ----------------------------------------
class ADXStrategy(TrendStrategy):
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> Trend:
        if len(df) < 15:
            return Trend.NA

        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

        if adx.iloc[-1] < 20:
            return Trend.SIDE_WAY
        elif plus_di.iloc[-1] > minus_di.iloc[-1]:
            return Trend.BULLISH
        else:
            return Trend.BEARISH

# ----------------------------------------
# Strategy 4: Linear Regression Slope
# ----------------------------------------
class SlopeRegressionStrategy(TrendStrategy):
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> Trend:
        period = 20
        if len(df) < period:
            return Trend.NA

        y = df['close'].tail(period).values.reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        slope = model.coef_[0][0]

        if abs(slope) < 0.01:
            return Trend.SIDE_WAY
        elif slope > 0:
            return Trend.BULLISH
        else:
            return Trend.BEARISH

# ----------------------------------------
# Strategy 5: Volatility Compression (Range Bound)
# ----------------------------------------
class RangeBoundStrategy(TrendStrategy):
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> Trend:
        if len(df) < 20:
            return Trend.NA

        high = df['high'].tail(20).max()
        low = df['low'].tail(20).min()
        range_pct = (high - low) / low * 100

        return Trend.SIDE_WAY if range_pct < 2.0 else Trend.BULLISH if df['close'].iloc[-1] > df['close'].iloc[0] else Trend.BEARISH

# ----------------------------------------
# Strategy 6: RSI
# ----------------------------------------
class RSIStrategy(TrendStrategy):
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> Trend:
        if len(df) < 20:
            return Trend.NA

        rsi = talib.RSI(df['close'], timeperiod=14)
        rsi_recent = rsi.dropna().iloc[-5:]

        if len(rsi_recent) < 5:
            return Trend.SIDE_WAY

        slope = (rsi_recent.iloc[-1] - rsi_recent.iloc[0]) / 5

        if rsi.iloc[-1] > 60 and slope > 0:
            return Trend.BULLISH
        elif rsi.iloc[-1] < 40 and slope < 0:
            return Trend.BEARISH
        else:
            return Trend.SIDE_WAY

# ----------------------------------------
# Strategy 7: Supertrend
# ----------------------------------------
class SupertrendStrategy(TrendStrategy):
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> Trend:
        if len(df) < 15:
            return Trend.NA

        period = 10
        multiplier = 3.0

        high = df['high']
        low = df['low']
        close = df['close']

        atr = talib.ATR(high, low, close, timeperiod=period)
        hl2 = (high + low) / 2
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        supertrend = pd.Series(index=close.index)
        supertrend.iloc[0] = upperband.iloc[0]
        trend = [True]

        for i in range(1, len(close)):
            curr_close = close.iloc[i]
            prev_supertrend = supertrend.iloc[i - 1]
            if curr_close > prev_supertrend:
                supertrend.iloc[i] = lowerband.iloc[i]
                trend.append(True)
            else:
                supertrend.iloc[i] = upperband.iloc[i]
                trend.append(False)

        if lowerband.iloc[-1] < close.iloc[-1] < upperband.iloc[-1]:
            return Trend.NA

        atr_pct = atr.iloc[-1] / close.iloc[-1]
        if atr_pct < 0.005:
            return Trend.NA

        trend_bools = pd.Series(trend).tail(10)
        flip_count = (trend_bools != trend_bools.shift(1)).sum()
        if flip_count >= 3:
            return Trend.NA

        return Trend.BULLISH if trend[-1] else Trend.BEARISH

# ----------------------------------------
# Strategy 8: MACDHistogramSlopeStrategy
# ----------------------------------------
class MACDHistogramSlopeStrategy(TrendStrategy):
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> Trend:
        if len(df) < 35:
            return Trend.NA

        macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        rsi = talib.RSI(df['close'], timeperiod=14)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        hist = hist.dropna()
        if len(hist) < 6:
            return Trend.NA

        hist_slope = (hist.iloc[-1] - hist.iloc[-5]) / 5
        hist_std = hist.std()

        if abs(hist_slope) < 0.5 * hist_std:
            return Trend.NA

        atr_pct = atr.iloc[-1] / df['close'].iloc[-1]
        if atr_pct < 0.005:
            return Trend.NA

        recent_hist = hist.tail(10)
        if ((recent_hist < 0) & (recent_hist.shift(1) > 0)).any() or \
           ((recent_hist > 0) & (recent_hist.shift(1) < 0)).any():
            return Trend.NA

        if macd.iloc[-1] > signal.iloc[-1] and rsi.iloc[-1] > 55:
            return Trend.BULLISH
        elif macd.iloc[-1] < signal.iloc[-1] and rsi.iloc[-1] < 45:
            return Trend.BEARISH

        return Trend.NA

# ----------------------------------------
# TrendDetector
# ----------------------------------------
class TrendDetector:
    def __init__(self, strategies: List[TrendStrategy]):
        self.strategies = strategies

    def detect(self, instrument: BasicInstrumentDetails, df: pd.DataFrame) -> dict:
        result_map = {
            strategy.__class__.__name__: strategy.detect_trend(instrument, df)
            for strategy in self.strategies
        }
        return result_map

interval_strategy_weights = {
    CandleInterval.MIN_1: {
        "ADXStrategy": 2,
        "RangeBoundStrategy": 2,
        "SlopeRegressionStrategy": 1,
        "RSIStrategy": 1,
        "SupertrendStrategy": 2,
        "MACDHistogramSlopeStrategy": 1,
    },
    CandleInterval.MIN_5: {
        "ADXStrategy": 2,
        "RangeBoundStrategy": 1,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 1,
        "MovingAverageStrategy": 1,
        "SupertrendStrategy": 3,
        "MACDHistogramSlopeStrategy": 2,
    },
    CandleInterval.MIN_15: {
        "ADXStrategy": 2,
        "MovingAverageStrategy": 2,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 1,
        "PriceActionStrategy": 1,
        "SupertrendStrategy": 3,
        "MACDHistogramSlopeStrategy": 2,
    },
    CandleInterval.MIN_30: {
        "ADXStrategy": 2,
        "MovingAverageStrategy": 2,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 1,
        "PriceActionStrategy": 1,
        "SupertrendStrategy": 3,
        "MACDHistogramSlopeStrategy": 3,
    },
    CandleInterval.MIN_60: {
        "ADXStrategy": 2,
        "MovingAverageStrategy": 2,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 2,
        "PriceActionStrategy": 1,
        "SupertrendStrategy": 2,
        "MACDHistogramSlopeStrategy": 3,
    },
    CandleInterval.DAY: {
        "ADXStrategy": 2,
        "MovingAverageStrategy": 3,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 2,
        "PriceActionStrategy": 1,
        "SupertrendStrategy": 2,
        "MACDHistogramSlopeStrategy": 3,
    }
}

# ----------------------------------------
# Trend Detection Entrypoint
# ----------------------------------------
def detect_final_trend(instrument: BasicInstrumentDetails, time_to_check: datetime.datetime = datetime.datetime.now(),
                        interval: CandleInterval = CandleInterval.DAY) -> Trend:
    df = fetch_ohlc(instrument, time_to_check, interval).dropna()
    if df.empty or len(df) < 20:
        raise Exception("Not enough data to detect trend")

    all_strategies = [
        PriceActionStrategy(), MovingAverageStrategy(), ADXStrategy(),
        SlopeRegressionStrategy(), RangeBoundStrategy(), RSIStrategy(),
        SupertrendStrategy(), MACDHistogramSlopeStrategy()
    ]

    weights = interval_strategy_weights.get(interval, {})
    active_strategies = [s for s in all_strategies if weights.get(s.__class__.__name__, 0) > 0]
    detector = TrendDetector(active_strategies)
    result_map = detector.detect(instrument, df)

    vote_count = {
        Trend.BULLISH: 0,
        Trend.BEARISH: 0,
        Trend.SIDE_WAY: 0,
    }

    for strategy_name, trend in result_map.items():
        weight = weights.get(strategy_name, 0)
        if trend != Trend.NA:
            vote_count[trend] += weight

    sorted_votes = sorted(vote_count.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_votes) > 1 and sorted_votes[0][1] == sorted_votes[1][1]:
        return Trend.SIDE_WAY
    return sorted_votes[0][0]
