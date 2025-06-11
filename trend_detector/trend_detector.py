from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Tuple, NamedTuple
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
# Trend Result with Confidence
# ------------------------
class TrendResult(NamedTuple):
    trend: Trend
    confidence: float  # 0.0 to 1.0, where 1.0 is highest confidence

# ------------------------
# Base Strategy Interface
# ------------------------
class TrendStrategy(ABC):
    @abstractmethod
    def detect_trend(self, instrument: BasicInstrumentDetails, df: pd.DataFrame, indicators: dict) -> TrendResult:
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
        
        # Check if we have enough data
        if len(highs) < 5:
            return TrendResult(Trend.NA, 0.0)
        
        # Count consecutive higher highs/lows or lower highs/lows
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        total_points = len(highs) - 1
        bullish_score = (higher_highs + higher_lows) / (2 * total_points)
        bearish_score = (lower_highs + lower_lows) / (2 * total_points)
        
        if bullish_score > 0.6:
            return TrendResult(Trend.BULLISH, bullish_score)
        elif bearish_score > 0.6:
            return TrendResult(Trend.BEARISH, bearish_score)
        else:
            return TrendResult(Trend.SIDE_WAY, max(0.3, 1 - max(bullish_score, bearish_score)))

class MovingAverageStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        ema20, ema50 = indicators['ema_20'], indicators['ema_50']
        
        if pd.isna(ema20.iloc[-1]) or pd.isna(ema50.iloc[-1]):
            return TrendResult(Trend.NA, 0.0)
        
        # Calculate the percentage difference between EMAs
        diff_pct = abs(ema20.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1] * 100
        
        # Normalize confidence based on the separation (more separation = higher confidence)
        confidence = min(1.0, diff_pct / 2.0)  # 2% separation = 100% confidence
        
        if ema20.iloc[-1] > ema50.iloc[-1]:
            return TrendResult(Trend.BULLISH, confidence)
        elif ema20.iloc[-1] < ema50.iloc[-1]:
            return TrendResult(Trend.BEARISH, confidence)
        else:
            return TrendResult(Trend.SIDE_WAY, 0.5)

class ADXStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        adx = indicators['adx']
        plus_di = indicators['plus_di']
        minus_di = indicators['minus_di']
        
        if pd.isna(adx.iloc[-1]) or pd.isna(plus_di.iloc[-1]) or pd.isna(minus_di.iloc[-1]):
            return TrendResult(Trend.NA, 0.0)
        
        adx_val = adx.iloc[-1]
        
        if adx_val < 20:
            # Weak trend, high confidence in sideways
            confidence = (20 - adx_val) / 20
            return TrendResult(Trend.SIDE_WAY, confidence)
        
        # Strong trend, calculate confidence based on ADX strength and DI separation
        di_diff = abs(plus_di.iloc[-1] - minus_di.iloc[-1])
        adx_confidence = min(1.0, (adx_val - 20) / 30)  # ADX 50+ = full confidence
        di_confidence = min(1.0, di_diff / 20)  # 20+ DI difference = full confidence
        
        confidence = (adx_confidence + di_confidence) / 2
        
        if plus_di.iloc[-1] > minus_di.iloc[-1]:
            return TrendResult(Trend.BULLISH, confidence)
        else:
            return TrendResult(Trend.BEARISH, confidence)

class SlopeRegressionStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators, period=20):
        if len(df) < period:
            return TrendResult(Trend.NA, 0.0)
        
        y = df['close'].tail(period).values.reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)
        
        model = LinearRegression().fit(x, y)
        slope = model.coef_[0][0]
        r_squared = model.score(x, y)
        
        # Normalize slope by price to get percentage change per period
        slope_pct = abs(slope) / df['close'].iloc[-1] * 100
        
        # Confidence based on R-squared and slope steepness
        slope_confidence = min(1.0, slope_pct / 0.5)  # 0.5% slope = full confidence
        confidence = r_squared * slope_confidence
        
        if abs(slope) < 0.01:
            return TrendResult(Trend.SIDE_WAY, max(0.3, 1 - slope_confidence))
        
        trend = Trend.BULLISH if slope > 0 else Trend.BEARISH
        return TrendResult(trend, confidence)

class RangeBoundStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators, percent=2.0):
        if len(df) < 20:
            return TrendResult(Trend.NA, 0.0)
        
        high, low = df['high'].tail(20).max(), df['low'].tail(20).min()
        range_pct = (high - low) / low * 100
        
        close_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100
        
        if range_pct < percent:
            # High confidence in sideways when range is tight
            confidence = (percent - range_pct) / percent
            return TrendResult(Trend.SIDE_WAY, confidence)
        
        # Trending market - confidence based on price movement vs range
        movement_confidence = min(1.0, abs(close_change) / range_pct)
        
        if close_change > 0:
            return TrendResult(Trend.BULLISH, movement_confidence)
        else:
            return TrendResult(Trend.BEARISH, movement_confidence)

class RSIStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        rsi = indicators['rsi_14'].dropna()
        if len(rsi) < 5:
            return TrendResult(Trend.SIDE_WAY, 0.0)
        
        rsi_recent = rsi.iloc[-5:]
        slope = (rsi_recent.iloc[-1] - rsi_recent.iloc[0]) / 5
        rsi_val = rsi.iloc[-1]
        
        # Calculate confidence based on RSI level and slope consistency
        if rsi_val > 60 and slope > 0:
            # Bullish: higher RSI = higher confidence, steeper slope = higher confidence
            level_confidence = min(1.0, (rsi_val - 60) / 20)  # RSI 80+ = full confidence
            slope_confidence = min(1.0, slope / 5)  # slope of 5+ = full confidence
            confidence = (level_confidence + slope_confidence) / 2
            return TrendResult(Trend.BULLISH, confidence)
        elif rsi_val < 40 and slope < 0:
            # Bearish: lower RSI = higher confidence, steeper negative slope = higher confidence
            level_confidence = min(1.0, (40 - rsi_val) / 20)  # RSI 20- = full confidence
            slope_confidence = min(1.0, abs(slope) / 5)
            confidence = (level_confidence + slope_confidence) / 2
            return TrendResult(Trend.BEARISH, confidence)
        else:
            # Sideways or neutral
            neutral_confidence = 1 - abs(rsi_val - 50) / 50  # RSI near 50 = high sideways confidence
            return TrendResult(Trend.SIDE_WAY, max(0.3, neutral_confidence))

class SupertrendStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        period = 10
        multiplier = 3.0
        high, low, close = df['high'], df['low'], df['close']
        atr = indicators['atr_14']
        
        if len(close) < period or pd.isna(atr.iloc[-1]):
            return TrendResult(Trend.NA, 0.0)
        
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

        # Calculate confidence based on price distance from supertrend and trend consistency
        price_distance = abs(close.iloc[-1] - supertrend.iloc[-1]) / close.iloc[-1] * 100
        distance_confidence = min(1.0, price_distance / 1.0)  # 1% distance = full confidence
        
        # Check trend consistency (fewer changes = higher confidence)
        recent_changes = pd.Series(trend).tail(10).ne(pd.Series(trend).tail(10).shift()).sum()
        consistency_confidence = max(0.1, 1 - recent_changes / 10)
        
        # Check if price is between bands (uncertain zone)
        if lowerband.iloc[-1] < close.iloc[-1] < upperband.iloc[-1]:
            return TrendResult(Trend.NA, 0.0)
        
        # Check for low volatility
        if atr.iloc[-1] / close.iloc[-1] < 0.005:
            return TrendResult(Trend.NA, 0.0)
        
        # Check for too many trend changes
        if recent_changes >= 3:
            return TrendResult(Trend.NA, 0.0)

        confidence = (distance_confidence + consistency_confidence) / 2
        trend_result = Trend.BULLISH if trend[-1] else Trend.BEARISH
        return TrendResult(trend_result, confidence)

class MACDHistogramSlopeStrategy(TrendStrategy):
    def detect_trend(self, instrument, df, indicators):
        macd, signal, hist = indicators['macd'], indicators['macd_signal'], indicators['macd_hist']
        rsi = indicators['rsi_14']
        atr = indicators['atr_14']

        if len(hist.dropna()) < 6:
            return TrendResult(Trend.NA, 0.0)

        hist_slope = (hist.iloc[-1] - hist.iloc[-5]) / 5
        hist_std = hist.std()
        
        # Check various conditions for trend validity
        if abs(hist_slope) < 0.5 * hist_std:
            return TrendResult(Trend.NA, 0.0)
        if atr.iloc[-1] / df['close'].iloc[-1] < 0.005:
            return TrendResult(Trend.NA, 0.0)
        
        recent_hist = hist.tail(10)
        if ((recent_hist < 0) & (recent_hist.shift(1) > 0)).any() or ((recent_hist > 0) & (recent_hist.shift(1) < 0)).any():
            return TrendResult(Trend.NA, 0.0)

        # Calculate confidence based on MACD-Signal separation and RSI confirmation
        macd_separation = abs(macd.iloc[-1] - signal.iloc[-1])
        separation_confidence = min(1.0, macd_separation / (2 * hist_std))
        
        # RSI confirmation confidence
        rsi_val = rsi.iloc[-1]
        
        if macd.iloc[-1] > signal.iloc[-1] and rsi_val > 55:
            rsi_confidence = min(1.0, (rsi_val - 55) / 25)  # RSI 80+ = full confidence
            confidence = (separation_confidence + rsi_confidence) / 2
            return TrendResult(Trend.BULLISH, confidence)
        elif macd.iloc[-1] < signal.iloc[-1] and rsi_val < 45:
            rsi_confidence = min(1.0, (45 - rsi_val) / 25)  # RSI 20- = full confidence
            confidence = (separation_confidence + rsi_confidence) / 2
            return TrendResult(Trend.BEARISH, confidence)
        
        return TrendResult(Trend.NA, 0.0)

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
# Strategy Weights - Optimized for Indian Market
# ------------------------
interval_strategy_weights = {
    # 1-minute: High volatility, noise-prone, focus on momentum and volatility-based indicators
    CandleInterval.MIN_1: {
        "ADXStrategy": 3,                    # High weight - excellent for trend strength in volatile conditions
        "RangeBoundStrategy": 3,             # High weight - Indian market often range-bound intraday
        "SupertrendStrategy": 2,             # Reduced - too much noise in 1-min
        "RSIStrategy": 2,                    # Good for overbought/oversold in short timeframes
        "SlopeRegressionStrategy": 1,        # Low - too noisy for regression
        "MACDHistogramSlopeStrategy": 1,     # Low - not reliable in 1-min noise
        "MovingAverageStrategy": 1,          # Low - EMAs lag too much
        "PriceActionStrategy": 1,            # Low - hard to identify clean patterns
    },
    
    # 5-minute: Still noisy but starting to show cleaner patterns
    CandleInterval.MIN_5: {
        "ADXStrategy": 3,                    # Excellent for trend identification
        "SupertrendStrategy": 3,             # Very effective in Indian intraday trading
        "RangeBoundStrategy": 2,             # Indian stocks often consolidate
        "RSIStrategy": 2,                    # Good momentum indicator
        "MACDHistogramSlopeStrategy": 2,     # Starting to be more reliable
        "MovingAverageStrategy": 2,          # EMAs become more useful
        "SlopeRegressionStrategy": 1,        # Still somewhat noisy
        "PriceActionStrategy": 1,            # Patterns not clear enough yet
    },
    
    # 15-minute: Sweet spot for intraday trading in Indian markets
    CandleInterval.MIN_15: {
        "SupertrendStrategy": 4,             # Highest weight - excellent for Indian intraday
        "ADXStrategy": 3,                    # Very reliable trend strength indicator
        "MovingAverageStrategy": 3,          # EMAs work well at this timeframe
        "MACDHistogramSlopeStrategy": 3,     # Very effective momentum indicator
        "SlopeRegressionStrategy": 2,        # Regression becomes more reliable
        "RSIStrategy": 2,                    # Good for entry/exit timing
        "PriceActionStrategy": 2,            # Patterns start becoming clearer
        "RangeBoundStrategy": 1,             # Less relevant as trends develop
    },
    
    # 30-minute: Excellent balance of signal clarity and responsiveness
    CandleInterval.MIN_30: {
        "SupertrendStrategy": 4,             # Excellent performance in Indian markets
        "MACDHistogramSlopeStrategy": 4,     # Very reliable at this timeframe
        "ADXStrategy": 3,                    # Strong trend identification
        "MovingAverageStrategy": 3,          # EMAs very effective
        "SlopeRegressionStrategy": 3,        # Regression analysis reliable
        "PriceActionStrategy": 2,            # Clear pattern recognition
        "RSIStrategy": 2,                    # Good momentum confirmation
        "RangeBoundStrategy": 1,             # Less common in 30-min trends
    },
    
    # 1-hour: Good for swing trading and position entries
    CandleInterval.MIN_60: {
        "MACDHistogramSlopeStrategy": 4,     # Excellent momentum detection
        "SupertrendStrategy": 3,             # Strong trend following
        "MovingAverageStrategy": 3,          # Very reliable trend identification
        "SlopeRegressionStrategy": 3,        # Excellent regression analysis
        "ADXStrategy": 3,                    # Reliable trend strength
        "PriceActionStrategy": 3,            # Clear pattern recognition
        "RSIStrategy": 2,                    # Good for timing entries
        "RangeBoundStrategy": 1,             # Less relevant for hourly trends
    },
    
    # Daily: Best for positional and swing trading
    CandleInterval.DAY: {
        "MovingAverageStrategy": 4,          # Most reliable for long-term trends
        "MACDHistogramSlopeStrategy": 4,     # Excellent for trend changes
        "SlopeRegressionStrategy": 4,        # Very reliable regression analysis
        "PriceActionStrategy": 3,            # Clear daily patterns
        "ADXStrategy": 3,                    # Good trend strength indicator
        "SupertrendStrategy": 3,             # Reliable for daily trends
        "RSIStrategy": 3,                    # Good for overbought/oversold levels
        "RangeBoundStrategy": 2,             # Some stocks do range-bound on daily
    }
}

# ------------------------
# Detection Entry Points
# ------------------------
def detect_trend_at(instrument: BasicInstrumentDetails, time_to_check: datetime, interval: CandleInterval, strategies: List[TrendStrategy] = None) -> dict:
    """
    Returns dict with strategy names as keys and TrendResult (trend, confidence) as values
    """
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

def detect_final_trend(instrument: BasicInstrumentDetails, time_to_check: datetime = datetime.datetime.now(), interval: CandleInterval = CandleInterval.DAY) -> Tuple[Trend, float]:
    """
    Returns a tuple of (final_trend, overall_confidence)
    """
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
    confidence_sum = {Trend.BULLISH: 0, Trend.BEARISH: 0, Trend.SIDE_WAY: 0}
    total_weight = 0

    for strategy_name, trend_result in result_map.items():
        if trend_result.trend != Trend.NA:
            weight = weights.get(strategy_name, 0)
            # Weight the vote by both strategy weight and confidence
            weighted_vote = weight * trend_result.confidence
            vote_count[trend_result.trend] += weighted_vote
            confidence_sum[trend_result.trend] += trend_result.confidence * weight
            total_weight += weight

    if total_weight == 0:
        return Trend.SIDE_WAY, 0.0

    # Find the trend with highest weighted vote
    final_trend = max(vote_count.items(), key=lambda x: x[1])[0]
    
    # Calculate overall confidence as weighted average of contributing strategies
    if vote_count[final_trend] > 0:
        overall_confidence = confidence_sum[final_trend] / sum(weights.get(name, 0) 
                                                              for name, result in result_map.items() 
                                                              if result.trend == final_trend)
    else:
        overall_confidence = 0.0

    return final_trend, overall_confidence

# ------------------------
# Helper function to get detailed breakdown
# ------------------------
def get_trend_breakdown(instrument: BasicInstrumentDetails, time_to_check: datetime = datetime.datetime.now(), interval: CandleInterval = CandleInterval.DAY) -> dict:
    """
    Returns detailed breakdown of all strategies with their trends, confidences, and weights
    """
    result_map = detect_trend_at(instrument, time_to_check, interval)
    weights = interval_strategy_weights.get(interval, {})
    
    breakdown = []
    for strategy_name, trend_result in result_map.items():
        breakdown.append({
            'strategy': strategy_name,
            'trend': trend_result.trend.value,
            'confidence': round(trend_result.confidence, 3),
            'weight': weights.get(strategy_name, 0),
            'weighted_score': round(weights.get(strategy_name, 0) * trend_result.confidence, 3)
        })
    
    final_trend, overall_confidence = detect_final_trend(instrument, time_to_check, interval)
    
    return {
        'strategies': breakdown,
        'final_trend': final_trend.value,
        'overall_confidence': round(overall_confidence, 3)
    }