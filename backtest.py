"""
Backtesting Module
Simple historical backtest of the strategy
"""

import pandas as pd
import numpy as np
from features import FeatureEngine
from ml_ranker import MLRanker


class Backtester:
    """Simple backtesting engine"""
    
    def __init__(self, initial_capital=100000, risk_per_trade=0.5):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk per trade as % of capital
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.feature_engine = FeatureEngine()
        self.ml_ranker = MLRanker()
    
    def run_backtest(self, df, atr_multiplier=2.0, reward_ratio=2.0):
        """
        Run backtest on historical data
        
        Args:
            df: Historical OHLCV data
            atr_multiplier: Stop loss ATR multiplier
            reward_ratio: Risk:Reward ratio
            
        Returns:
            dict with backtest results
        """
        if df is None or len(df) < 100:
            return None
        
        # Add features
        df = self.feature_engine.add_technical_indicators(df)
        
        if df is None:
            return None
        
        # Train ML model
        train_size = int(len(df) * 0.7)
        train_df = df.iloc[:train_size].copy()
        
        metrics = self.ml_ranker.train(train_df)
        
        if metrics is None:
            return None
        
        # Test on remaining data
        test_df = df.iloc[train_size:].copy()
        test_df = test_df.dropna()
        
        trades = []
        capital = self.initial_capital
        
        for i in range(len(test_df) - 10):
            row = test_df.iloc[i]
            
            # Get features
            features = {}
            for col in self.ml_ranker.feature_cols:
                if col in row:
                    features[col] = row[col]
            
            # Get ML score
            score = self.ml_ranker.predict_proba(features)
            
            if score < 0.6:
                continue
            
            # Simulate trade
            entry = row['close']
            atr = row['atr']
            stop = entry - (atr_multiplier * atr)
            target = entry + ((entry - stop) * reward_ratio)
            
            risk_per_share = entry - stop
            risk_amount = capital * (self.risk_per_trade / 100)
            quantity = int(risk_amount / risk_per_share)
            
            if quantity <= 0:
                continue
            
            # Check outcome in next 5 days
            future_prices = test_df.iloc[i+1:i+6]['close'].values
            
            if len(future_prices) == 0:
                continue
            
            # Check if stop or target hit
            hit_stop = any(future_prices <= stop)
            hit_target = any(future_prices >= target)
            
            if hit_target:
                exit_price = target
                pnl = (exit_price - entry) * quantity
            elif hit_stop:
                exit_price = stop
                pnl = (exit_price - entry) * quantity
            else:
                exit_price = future_prices[-1]
                pnl = (exit_price - entry) * quantity
            
            capital += pnl
            
            trades.append({
                'entry': entry,
                'exit': exit_price,
                'pnl': pnl,
                'pnl_pct': ((exit_price - entry) / entry) * 100,
                'score': score
            })
        
        if len(trades) == 0:
            return None
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        
        winners = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winners) / len(trades_df) * 100
        
        total_pnl = trades_df['pnl'].sum()
        final_capital = self.initial_capital + total_pnl
        
        # Sharpe ratio
        returns = trades_df['pnl_pct'].values
        sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Max drawdown
        cum_pnl = trades_df['pnl'].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_trades': len(trades_df),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'final_capital': round(final_capital, 2),
            'return_pct': round((total_pnl / self.initial_capital) * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 2),
            'avg_win': round(winners['pnl'].mean(), 2) if len(winners) > 0 else 0,
            'avg_loss': round(trades_df[trades_df['pnl'] < 0]['pnl'].mean(), 2),
            'equity_curve': cum_pnl.values,
            'trades': trades_df
        }
