import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict
from const import *
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules (assuming they're available)
from const import CandleInterval, BasicInstrumentDetails
from market_data import fetch_ohlc, fetch_ohlc_between_given_time
from trend_detector.trend_detector import detect_final_trend, Trend

@dataclass
class BacktestResult:
    """Container for individual backtest results"""
    date: datetime
    predicted_trend: Trend
    confidence: float
    actual_return: float
    forward_period_days: int
    interval: CandleInterval
    
@dataclass
class BacktestSummary:
    """Summary statistics for backtest results"""
    total_predictions: int
    accuracy: float
    precision_bullish: float
    precision_bearish: float
    precision_sideway: float
    recall_bullish: float
    recall_bearish: float
    recall_sideway: float
    avg_confidence: float
    confidence_correlation: float
    profitable_predictions: int
    avg_return_bullish: float
    avg_return_bearish: float
    avg_return_sideway: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

class TrendDetectorBacktester:
    """
    Comprehensive backtesting framework for trend detection strategies
    """
    
    def __init__(self, 
                 return_thresholds: Dict[str, float] = None,
                 confidence_threshold: float = 0.0):
        """
        Initialize backtester
        
        Args:
            return_thresholds: Dict mapping trend types to return thresholds
                              e.g., {'bullish': 0.02, 'bearish': -0.02, 'sideway': 0.01}
            confidence_threshold: Minimum confidence to consider predictions
        """
        self.return_thresholds = return_thresholds or {
            'bullish': 0.015,   # 1.5% gain for bullish
            'bearish': -0.015,  # 1.5% loss for bearish  
            'sideway': 0.01     # Within 1% for sideways
        }
        self.confidence_threshold = confidence_threshold
        self.results: List[BacktestResult] = []
        
    def classify_actual_trend(self, actual_return: float) -> Trend:
        """Classify actual market movement based on return thresholds"""
        if actual_return >= self.return_thresholds['bullish']:
            return Trend.BULLISH
        elif actual_return <= self.return_thresholds['bearish']:
            return Trend.BEARISH
        else:
            return Trend.SIDE_WAY
    
    def calculate_forward_returns(self, 
                                instrument: BasicInstrumentDetails,
                                start_date: datetime,
                                forward_periods: List[int]) -> Dict[int, float]:
        """
        Calculate forward returns for different periods
        
        Args:
            instrument: Trading instrument
            start_date: Starting date for calculation
            forward_periods: List of forward periods in days [1, 3, 5, 10, 20]
            
        Returns:
            Dict mapping period to return percentage
        """
        returns = {}
        
        try:
            # Fetch daily data for a wider range to ensure we have enough data
            end_date = start_date + timedelta(days=max(forward_periods) + 10)
            # df = fetch_ohlc(instrument, end_date, CandleInterval.DAY)
            df = fetch_ohlc_between_given_time(instrument, start_date,end_date, CandleInterval.DAY)
            if df.empty:
                return {period: 0.0 for period in forward_periods}
            
            # Find the starting price (closest date to start_date)
            df['date'] = pd.to_datetime(df.index)
            start_idx = df[df['date'] >= start_date].index
            
            if len(start_idx) == 0:
                return {period: 0.0 for period in forward_periods}
            
            start_price = df.loc[start_idx[0], 'close']
            
            for period in forward_periods:
                try:
                    end_idx = start_idx[0] + period
                    if end_idx < len(df):
                        end_price = df.iloc[end_idx]['close']
                        returns[period] = (end_price - start_price) / start_price
                    else:
                        returns[period] = 0.0
                except:
                    returns[period] = 0.0
                    
        except Exception as e:
            print(f"Error calculating forward returns: {e}")
            returns = {period: 0.0 for period in forward_periods}
            
        return returns
    
    def run_backtest(self,
                    instrument: BasicInstrumentDetails,
                    start_date: datetime,
                    end_date: datetime,
                    interval: CandleInterval = CandleInterval.DAY,
                    forward_periods: List[int] = [1, 3, 5, 10, 20],
                    step_days: int = 1) -> Dict[int, List[BacktestResult]]:
        """
        Run comprehensive backtest
        
        Args:
            instrument: Trading instrument to test
            start_date: Start date for backtesting
            end_date: End date for backtesting  
            interval: Timeframe for trend detection
            forward_periods: List of forward-looking periods to evaluate
            step_days: Step size in days between predictions
            
        Returns:
            Dict mapping forward periods to list of BacktestResult objects
        """
        results_by_period = {period: [] for period in forward_periods}
        current_date = start_date
        
        print(f"Running backtest from {start_date.date()} to {end_date.date()}")
        print(f"Instrument: {instrument.kite_trading_symbol}")
        print(f"Interval: {interval}")
        print(f"Forward periods: {forward_periods}")
        
        total_dates = (end_date - start_date).days // step_days
        processed = 0
        
        while current_date <= end_date:
            try:
                # Get trend prediction
                predicted_trend, confidence = detect_final_trend(
                    instrument, current_date, interval
                )
                
                # Skip if confidence is below threshold
                if confidence < self.confidence_threshold:
                    current_date += timedelta(days=step_days)
                    continue
                
                # Calculate forward returns for all periods
                forward_returns = self.calculate_forward_returns(
                    instrument, current_date, forward_periods
                )
                
                # Create results for each forward period
                for period in forward_periods:
                    actual_return = forward_returns.get(period, 0.0)
                    
                    result = BacktestResult(
                        date=current_date,
                        predicted_trend=predicted_trend,
                        confidence=confidence,
                        actual_return=actual_return,
                        forward_period_days=period,
                        interval=interval
                    )
                    
                    results_by_period[period].append(result)
                
                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed}/{total_dates} dates ({processed/total_dates*100:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing {current_date}: {e}")
                
            current_date += timedelta(days=step_days)
        
        # Store results for later analysis
        self.results = []
        for period_results in results_by_period.values():
            self.results.extend(period_results)
            
        print(f"Backtest completed. Total predictions: {sum(len(r) for r in results_by_period.values())}")
        return results_by_period
    
    def calculate_metrics(self, results: List[BacktestResult]) -> BacktestSummary:
        """Calculate comprehensive performance metrics"""
        if not results:
            return BacktestSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        df = pd.DataFrame([{
            'predicted': r.predicted_trend.value,
            'confidence': r.confidence,
            'return': r.actual_return,
            'date': r.date
        } for r in results])
        
        # Classify actual trends
        df['actual'] = df['return'].apply(self.classify_actual_trend).apply(lambda x: x.value)
        
        # Basic accuracy
        accuracy = (df['predicted'] == df['actual']).mean()
        
        # Precision and Recall for each class
        trends = ['BULLISH', 'BEARISH', 'SIDE_WAY']
        precision = {}
        recall = {}
        
        for trend in trends:
            predicted_trend = df['predicted'] == trend
            actual_trend = df['actual'] == trend
            
            if predicted_trend.sum() > 0:
                precision[trend] = (predicted_trend & actual_trend).sum() / predicted_trend.sum()
            else:
                precision[trend] = 0.0
                
            if actual_trend.sum() > 0:
                recall[trend] = (predicted_trend & actual_trend).sum() / actual_trend.sum()
            else:
                recall[trend] = 0.0
        
        # Confidence correlation with accuracy
        df['correct'] = (df['predicted'] == df['actual']).astype(int)
        confidence_correlation = df['confidence'].corr(df['correct'])
        
        # Return analysis
        avg_returns = df.groupby('predicted')['return'].mean()
        
        # Trading performance metrics
        df['profitable'] = df['return'] > 0
        profitable_predictions = df['profitable'].sum()
        win_rate = df['profitable'].mean()
        
        # Sharpe ratio (assuming daily returns)
        if df['return'].std() > 0:
            sharpe_ratio = df['return'].mean() / df['return'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative_returns = (1 + df['return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return BacktestSummary(
            total_predictions=len(results),
            accuracy=accuracy,
            precision_bullish=precision.get('BULLISH', 0),
            precision_bearish=precision.get('BEARISH', 0),
            precision_sideway=precision.get('SIDE_WAY', 0),
            recall_bullish=recall.get('BULLISH', 0),
            recall_bearish=recall.get('BEARISH', 0),
            recall_sideway=recall.get('SIDE_WAY', 0),
            avg_confidence=df['confidence'].mean(),
            confidence_correlation=confidence_correlation if not pd.isna(confidence_correlation) else 0,
            profitable_predictions=profitable_predictions,
            avg_return_bullish=avg_returns.get('BULLISH', 0),
            avg_return_bearish=avg_returns.get('BEARISH', 0),
            avg_return_sideway=avg_returns.get('SIDE_WAY', 0),
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate
        )
    
    def print_summary(self, summary: BacktestSummary, period: int = None):
        """Print formatted summary statistics"""
        period_str = f" ({period}-day forward)" if period else ""
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY{period_str}")
        print(f"{'='*60}")
        print(f"Total Predictions: {summary.total_predictions}")
        print(f"Overall Accuracy: {summary.accuracy:.3f}")
        print(f"Average Confidence: {summary.avg_confidence:.3f}")
        print(f"Confidence-Accuracy Correlation: {summary.confidence_correlation:.3f}")
        
        print(f"\nPRECISION BY TREND:")
        print(f"  Bullish: {summary.precision_bullish:.3f}")
        print(f"  Bearish: {summary.precision_bearish:.3f}")
        print(f"  Sideways: {summary.precision_sideway:.3f}")
        
        print(f"\nRECALL BY TREND:")
        print(f"  Bullish: {summary.recall_bullish:.3f}")
        print(f"  Bearish: {summary.recall_bearish:.3f}")
        print(f"  Sideways: {summary.recall_sideway:.3f}")
        
        print(f"\nRETURN ANALYSIS:")
        print(f"  Avg Return (Bullish Predictions): {summary.avg_return_bullish:.4f}")
        print(f"  Avg Return (Bearish Predictions): {summary.avg_return_bearish:.4f}")
        print(f"  Avg Return (Sideways Predictions): {summary.avg_return_sideway:.4f}")
        
        print(f"\nTRADING METRICS:")
        print(f"  Win Rate: {summary.win_rate:.3f}")
        print(f"  Sharpe Ratio: {summary.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {summary.max_drawdown:.3f}")
        print(f"  Profitable Predictions: {summary.profitable_predictions}")
    
    def plot_results(self, results_by_period: Dict[int, List[BacktestResult]]):
        """Create comprehensive visualization of backtest results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trend Detector Backtest Results', fontsize=16)
        
        # Prepare data for plotting
        all_results = []
        for period, results in results_by_period.items():
            for r in results:
                all_results.append({
                    'period': period,
                    'predicted': r.predicted_trend.value,
                    'confidence': r.confidence,
                    'return': r.actual_return,
                    'date': r.date,
                    'actual': self.classify_actual_trend(r.actual_return).value
                })
        
        df_plot = pd.DataFrame(all_results)
        
        # 1. Accuracy by forward period
        accuracy_by_period = df_plot.groupby('period').apply(
            lambda x: (x['predicted'] == x['actual']).mean()
        )
        axes[0,0].bar(accuracy_by_period.index, accuracy_by_period.values)
        axes[0,0].set_title('Accuracy by Forward Period')
        axes[0,0].set_xlabel('Days Forward')
        axes[0,0].set_ylabel('Accuracy')
        
        # 2. Confidence distribution
        axes[0,1].hist(df_plot['confidence'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Confidence Distribution')
        axes[0,1].set_xlabel('Confidence')
        axes[0,1].set_ylabel('Frequency')
        
        # 3. Return distribution by predicted trend
        for trend in ['BULLISH', 'BEARISH', 'SIDE_WAY']:
            trend_returns = df_plot[df_plot['predicted'] == trend]['return']
            if len(trend_returns) > 0:
                axes[0,2].hist(trend_returns, alpha=0.5, label=trend, bins=20)
        axes[0,2].set_title('Return Distribution by Predicted Trend')
        axes[0,2].set_xlabel('Return')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].legend()
        
        # 4. Confidence vs Accuracy scatter
        confidence_bins = pd.cut(df_plot['confidence'], bins=10)
        conf_acc = df_plot.groupby(confidence_bins).apply(
            lambda x: (x['predicted'] == x['actual']).mean()
        )
        conf_centers = [interval.mid for interval in conf_acc.index]
        axes[1,0].scatter(conf_centers, conf_acc.values)
        axes[1,0].plot(conf_centers, conf_acc.values, 'r--', alpha=0.5)
        axes[1,0].set_title('Confidence vs Accuracy')
        axes[1,0].set_xlabel('Confidence')
        axes[1,0].set_ylabel('Accuracy')
        
        # 5. Confusion Matrix (for 5-day period as example)
        if 5 in results_by_period:
            df_5day = df_plot[df_plot['period'] == 5]
            confusion_data = pd.crosstab(df_5day['actual'], df_5day['predicted'])
            sns.heatmap(confusion_data, annot=True, fmt='d', ax=axes[1,1], cmap='Blues')
            axes[1,1].set_title('Confusion Matrix (5-day forward)')
        
        # 6. Cumulative returns over time
        df_plot_sorted = df_plot.sort_values('date')
        cumulative_returns = (1 + df_plot_sorted['return']).cumprod()
        axes[1,2].plot(df_plot_sorted['date'], cumulative_returns)
        axes[1,2].set_title('Cumulative Returns Over Time')
        axes[1,2].set_xlabel('Date')
        axes[1,2].set_ylabel('Cumulative Return')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_by_confidence_buckets(self, results: List[BacktestResult], buckets: int = 5):
        """Analyze performance by confidence buckets"""
        df = pd.DataFrame([{
            'confidence': r.confidence,
            'return': r.actual_return,
            'predicted': r.predicted_trend.value,
            'actual': self.classify_actual_trend(r.actual_return).value
        } for r in results])
        
        df['conf_bucket'] = pd.cut(df['confidence'], bins=buckets, labels=[f'Q{i+1}' for i in range(buckets)])
        
        bucket_analysis = df.groupby('conf_bucket').agg({
            'confidence': ['mean', 'count'],
            'return': 'mean',
        }).round(4)
        
        bucket_accuracy = df.groupby('conf_bucket').apply(
            lambda x: (x['predicted'] == x['actual']).mean()
        ).round(4)
        
        print(f"\n{'='*50}")
        print("CONFIDENCE BUCKET ANALYSIS")
        print(f"{'='*50}")
        print(bucket_analysis)
        print(f"\nAccuracy by bucket:")
        print(bucket_accuracy)
        
        return bucket_analysis, bucket_accuracy

# Example usage and testing functions
def run_comprehensive_backtest(instrument: BasicInstrumentDetails,
                              start_date: datetime,
                              end_date: datetime,
                              intervals: List[CandleInterval] = None):
    """
    Run comprehensive backtest across multiple intervals
    """
    if intervals is None:
        intervals = [CandleInterval.DAY, CandleInterval.MIN_60, CandleInterval.MIN_30]
    
    results = {}
    
    for interval in intervals:
        print(f"\n{'#'*80}")
        print(f"TESTING INTERVAL: {interval}")
        print(f"{'#'*80}")
        
        backtester = TrendDetectorBacktester(
            confidence_threshold=0.1  # Only consider predictions with >10% confidence
        )
        
        # Run backtest
        results_by_period = backtester.run_backtest(
            instrument=instrument,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            forward_periods=[1, 3, 5, 10, 20],
            step_days=3  # Test every 3 days to reduce computation
        )
        
        # Analyze results for each forward period
        interval_results = {}
        for period, period_results in results_by_period.items():
            if period_results:  # Only analyze if we have results
                summary = backtester.calculate_metrics(period_results)
                backtester.print_summary(summary, period)
                interval_results[period] = summary
        
        results[interval] = interval_results
        
        # Confidence bucket analysis for 5-day forward period
        if 5 in results_by_period and results_by_period[5]:
            backtester.analyze_by_confidence_buckets(results_by_period[5])
        
        # Plot results
        if results_by_period:
            backtester.plot_results(results_by_period)
    
    return results

# Example usage
if __name__ == "__main__":
    # Example instrument (you'll need to replace with actual instrument)
    example_instrument = Instrument.NIFTY.value
    # BasicInstrumentDetails(
    #     symbol="RELIANCE",
    #     exchange="NSE",
    #     # Add other required fields based on your BasicInstrumentDetails class
    # )
    
    # Define backtest period
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 6, 1)
    
    # Run comprehensive backtest
    backtest_results = run_comprehensive_backtest(
        instrument=example_instrument,
        start_date=start_date,
        end_date=end_date,
        intervals=[CandleInterval.DAY, CandleInterval.MIN_60]
    )
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETED")
    print("="*80)
    print("Results stored in backtest_results dictionary")
    print("Keys:", list(backtest_results.keys()))