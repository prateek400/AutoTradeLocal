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
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame, period: int = 20) -> Trend:
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
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame, percent: float = 2.0) -> Trend:
        high = df['high'].tail(20).max()
        low = df['low'].tail(20).min()
        range_pct = (high - low) / low * 100
        return Trend.SIDE_WAY if range_pct < percent else Trend.BULLISH if df['close'].iloc[-1] > df['close'].iloc[0] else Trend.BEARISH

# ----------------------------------------
# Strategy 6: RSI
# ----------------------------------------

class RSIStrategy(TrendStrategy):
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame, period: int = 14) -> Trend:
        rsi = talib.RSI(df['close'], timeperiod=period)
        rsi_recent = rsi.dropna().iloc[-5:]  # Take last 5 RSI values for slope
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
# Strategy Weights by Interval
# ----------------------------------------
interval_strategy_weights = {
    CandleInterval.MIN_1: {
        "ADXStrategy": 2,
        "RangeBoundStrategy": 2,
        "SlopeRegressionStrategy": 1,
        "RSIStrategy": 1,
    },
    CandleInterval.MIN_5: {
        "ADXStrategy": 2,
        "RangeBoundStrategy": 1,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 1,
        "MovingAverageStrategy": 1,
    },
    CandleInterval.MIN_15: {
        "ADXStrategy": 2,
        "MovingAverageStrategy": 2,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 1,
        "PriceActionStrategy": 1,
    },
    CandleInterval.MIN_30: {
        "ADXStrategy": 2,
        "MovingAverageStrategy": 2,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 1,
        "PriceActionStrategy": 1,
    },
    CandleInterval.MIN_60: {
        "ADXStrategy": 2,
        "MovingAverageStrategy": 2,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 2,
        "PriceActionStrategy": 1,
    },
    CandleInterval.DAY: {
        "ADXStrategy": 2,
        "MovingAverageStrategy": 3,
        "SlopeRegressionStrategy": 2,
        "RSIStrategy": 2,
        "PriceActionStrategy": 1,
    }
}


# ----------------------------------------
# Trend Detection Manager
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


def detect_trend_at(instrument: BasicInstrumentDetails, time_to_check: datetime, interval: CandleInterval, strategies: List[TrendStrategy] = None) -> dict:
    if strategies is None:
        strategies = [
            PriceActionStrategy(),
            MovingAverageStrategy(),
            ADXStrategy(),
            SlopeRegressionStrategy(),
            RangeBoundStrategy(),
            RSIStrategy()
        ]
    
    df = fetch_ohlc(instrument, time_to_check, interval)
    if df.empty or len(df) < 20:
        raise Exception("Not enough data to detect trend")
    detector = TrendDetector(strategies)
    return detector.detect(instrument, df)

def detect_final_trend(instrument: BasicInstrumentDetails, time_to_check: datetime = datetime.datetime.now(),
                        interval: CandleInterval = CandleInterval.DAY) -> Trend:
    df = fetch_ohlc(instrument, time_to_check, interval)
    if df.empty or len(df) < 20:
        raise Exception("Not enough data to detect trend")

    # Use all strategies by default
    all_strategies = [
        PriceActionStrategy(),
        MovingAverageStrategy(),
        ADXStrategy(),
        SlopeRegressionStrategy(),
        RangeBoundStrategy(),
        RSIStrategy()
    ]

    detector = TrendDetector(all_strategies)
    result_map = detector.detect(instrument, df)

    weights = interval_strategy_weights.get(interval, {})

    vote_count = {
        Trend.BULLISH: 0,
        Trend.BEARISH: 0,
        Trend.SIDE_WAY: 0
    }

    for strategy_name, trend in result_map.items():
        weight = weights.get(strategy_name, 0)
        vote_count[trend] += weight

    # Final Decision
    final_trend = max(vote_count.items(), key=lambda x: x[1])[0]
    return final_trend
