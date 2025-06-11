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

# ------------------------
# Enum for Trend
# ------------------------
class Trend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDE_WAY = "SIDE_WAY"
    NA = "NA"

# ------------------------
# Base Strategy Interface
# ------------------------
class TrendStrategy(ABC):
    @abstractmethod
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame, indicators: dict) -> Trend:
        pass

# ------------------------
# Indicator Cache
# ------------------------
def compute_indicators(df: pd.DataFrame) -> dict:
    close = df['close']
    high = df['high']
    low = df['low']

    indicators = {}
    indicators['ema_20'] = talib.EMA(close, timeperiod=20)
    indicators['ema_50'] = talib.EMA(close, timeperiod=50)
    indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
    indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    indicators['rsi_14'] = talib.RSI(close, timeperiod=14)
    indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close, 12, 26, 9)
    indicators['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    return indicators

# ------------------------
# Price Action Strategy
# ------------------------
class PriceActionStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        highs, lows = df['high'].tolist(), df['low'].tolist()
        if all(x < y for x, y in zip(highs, highs[1:])) and all(x < y for x, y in zip(lows, lows[1:])):
            return Trend.BULLISH
        elif all(x > y for x, y in zip(highs, highs[1:])) and all(x > y for x, y in zip(lows, lows[1:])):
            return Trend.BEARISH
        return Trend.SIDE_WAY

class MovingAverageStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        ema20, ema50 = indicators['ema_20'], indicators['ema_50']
        if ema20.iloc[-1] > ema50.iloc[-1]:
            return Trend.BULLISH
        elif ema20.iloc[-1] < ema50.iloc[-1]:
            return Trend.BEARISH
        return Trend.SIDE_WAY

class ADXStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        adx = indicators['adx']
        plus_di = indicators['plus_di']
        minus_di = indicators['minus_di']
        if adx.iloc[-1] < 20:
            return Trend.SIDE_WAY
        elif plus_di.iloc[-1] > minus_di.iloc[-1]:
            return Trend.BULLISH
        return Trend.BEARISH

class SlopeRegressionStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators, period=20):
        y = df['close'].tail(period).values.reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)
        slope = LinearRegression().fit(x, y).coef_[0][0]
        if abs(slope) < 0.01:
            return Trend.SIDE_WAY
        return Trend.BULLISH if slope > 0 else Trend.BEARISH

class RangeBoundStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators, percent=2.0):
        high, low = df['high'].tail(20).max(), df['low'].tail(20).min()
        range_pct = (high - low) / low * 100
        if range_pct < percent:
            return Trend.SIDE_WAY
        return Trend.BULLISH if df['close'].iloc[-1] > df['close'].iloc[0] else Trend.BEARISH

class RSIStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        rsi = indicators['rsi_14'].dropna()
        if len(rsi) < 5:
            return Trend.SIDE_WAY
        rsi_recent = rsi.iloc[-5:]
        slope = (rsi_recent.iloc[-1] - rsi_recent.iloc[0]) / 5
        if rsi.iloc[-1] > 60 and slope > 0:
            return Trend.BULLISH
        elif rsi.iloc[-1] < 40 and slope < 0:
            return Trend.BEARISH
        return Trend.SIDE_WAY

class SupertrendStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        period = 10
        multiplier = 3.0
        high, low, close = df['high'], df['low'], df['close']
        atr = indicators['atr_14']
        hl2 = (high + low) / 2
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        supertrend = pd.Series(index=close.index)
        supertrend.iloc[0] = upperband.iloc[0]
        trend = [True]  # bullish

        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lowerband.iloc[i]
                trend.append(True)
            else:
                supertrend.iloc[i] = upperband.iloc[i]
                trend.append(False)

        if lowerband.iloc[-1] < close.iloc[-1] < upperband.iloc[-1]:
            return Trend.NA
        if atr.iloc[-1] / close.iloc[-1] < 0.005:
            return Trend.NA
        if pd.Series(trend).tail(10).ne(pd.Series(trend).tail(10).shift()).sum() >= 3:
            return Trend.NA

        return Trend.BULLISH if trend[-1] else Trend.BEARISH

class MACDHistogramSlopeStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        macd, signal, hist = indicators['macd'], indicators['macd_signal'], indicators['macd_hist']
        rsi = indicators['rsi_14']
        atr = indicators['atr_14']

        if len(hist.dropna()) < 6:
            return Trend.NA

        hist_slope = (hist.iloc[-1] - hist.iloc[-5]) / 5
        if abs(hist_slope) < 0.5 * hist.std():
            return Trend.NA
        if atr.iloc[-1] / df['close'].iloc[-1] < 0.005:
            return Trend.NA
        recent_hist = hist.tail(10)
        if ((recent_hist < 0) & (recent_hist.shift(1) > 0)).any() or ((recent_hist > 0) & (recent_hist.shift(1) < 0)).any():
            return Trend.NA

        if macd.iloc[-1] > signal.iloc[-1] and rsi.iloc[-1] > 55:
            return Trend.BULLISH
        elif macd.iloc[-1] < signal.iloc[-1] and rsi.iloc[-1] < 45:
            return Trend.BEARISH
        return Trend.NA

# ------------------------
# Trend Detector Class
# ------------------------
class TrendDetector:
    def __init__(self, strategies: List[TrendStrategy]):
        self.strategies = strategies

    def detect(self, instrument, df) -> dict:
        indicators = compute_indicators(df)
        return {
            strategy.__class__.__name__: strategy.detect_trend(instrument, df, indicators)
            for strategy in self.strategies
        }

# ------------------------
# Strategy Weights (Inject Your Existing Mapping)
# ------------------------
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

# ------------------------
# Detection Entry Points
# ------------------------
def detect_trend_at(instrument: BasicInstrumentDetails, time_to_check: datetime, interval: CandleInterval, strategies: List[TrendStrategy] = None) -> dict:
    if strategies is None:
        strategies = [
            PriceActionStrategy(),
            MovingAverageStrategy(),
            ADXStrategy(),
            SlopeRegressionStrategy(),
            RangeBoundStrategy(),
            RSIStrategy(),
            SupertrendStrategy(),
            MACDHistogramSlopeStrategy()
        ]

    df = fetch_ohlc(instrument, time_to_check, interval)
    if df.empty or len(df) < 20:
        raise Exception("Not enough data to detect trend")
    
    detector = TrendDetector(strategies)
    return detector.detect(instrument, df)

def detect_final_trend(instrument: BasicInstrumentDetails, time_to_check: datetime = datetime.datetime.now(), interval: CandleInterval = CandleInterval.DAY) -> Trend:
    df = fetch_ohlc(instrument, time_to_check, interval)
    if df.empty or len(df) < 20:
        raise Exception("Not enough data to detect trend")

    all_strategies = [
        PriceActionStrategy(),
        MovingAverageStrategy(),
        ADXStrategy(),
        SlopeRegressionStrategy(),
        RangeBoundStrategy(),
        RSIStrategy(),
        SupertrendStrategy(),
        MACDHistogramSlopeStrategy()
    ]

    detector = TrendDetector(all_strategies)
    result_map = detector.detect(instrument, df)

    weights = interval_strategy_weights.get(interval, {})
    vote_count = {Trend.BULLISH: 0, Trend.BEARISH: 0, Trend.SIDE_WAY: 0}

    for strategy_name, trend in result_map.items():
        if trend != Trend.NA:
            vote_count[trend] += weights.get(strategy_name, 0)

    return max(vote_count.items(), key=lambda x: x[1])[0]
