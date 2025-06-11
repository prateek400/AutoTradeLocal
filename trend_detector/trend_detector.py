from enum import Enum
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np
import talib
from sklearn.linear_model import LinearRegression
import datetime
from const import *
# ----------------------------------------
# Enum for Trend
# ----------------------------------------
class Trend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDE_WAY = "SIDE_WAY"

# ----------------------------------------
# Abstract Base Strategy
# ----------------------------------------
class TrendStrategy(ABC):

    @abstractmethod
    def detect_trend(self, instrument: str, df: pd.DataFrame) -> Trend:
        pass

# ----------------------------------------
# Strategy 1: HH/HL Price Action
# ----------------------------------------
class PriceActionStrategy(TrendStrategy):
    def detect_trend(self, instrument: str, df: pd.DataFrame) -> Trend:
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
    def detect_trend(self, instrument: str, df: pd.DataFrame) -> Trend:
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
    def detect_trend(self, instrument: str, df: pd.DataFrame) -> Trend:
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
    def detect_trend(self, instrument: str, df: pd.DataFrame, period: int = 20) -> Trend:
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
    def detect_trend(self, instrument: str, df: pd.DataFrame, percent: float = 2.0) -> Trend:
        high = df['high'].tail(20).max()
        low = df['low'].tail(20).min()
        range_pct = (high - low) / low * 100
        return Trend.SIDE_WAY if range_pct < percent else Trend.BULLISH if df['close'].iloc[-1] > df['close'].iloc[0] else Trend.BEARISH

# ----------------------------------------
# Trend Detection Manager
# ----------------------------------------
class TrendDetector:
    def __init__(self, strategies: List[TrendStrategy]):
        self.strategies = strategies

    def detect(self, instrument: str, df: pd.DataFrame) -> Trend:
        results = [strategy.detect_trend(instrument, df) for strategy in self.strategies]
        most_common = max(set(results), key=results.count)
        return most_common

# ----------------------------------------
# Usage Example (assuming df is already fetched OHLC DataFrame)
# ----------------------------------------
# from kiteconnect import KiteConnect
# df = fetch_ohlc_from_kite('RELIANCE', '15minute', lookback=20)

# strategies = [
#     PriceActionStrategy(),
#     MovingAverageStrategy(),
#     ADXStrategy(),
#     SlopeRegressionStrategy(),
#     RangeBoundStrategy(),
# ]
# detector = TrendDetector(strategies)
# print(detector.detect("RELIANCE", df))


# def detect_trend_at(instrument: str, time_to_check: datetime, interval: CandleInterval, strategies: List[TrendStrategy]) -> Trend:
#     df = fetch_ohlc(instrument, time_to_check, interval)
#     if df.empty or len(df) < 20:
#         raise Exception("Not enough data to detect trend")
#     detector = TrendDetector(strategies)
#     return detector.detect(instrument, df)
