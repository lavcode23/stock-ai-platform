"""
Trading Journal Module
Manages paper trading records and performance tracking
"""

import pandas as pd
import os
from datetime import datetime


class TradingJournal:
    """Manages trading journal and paper trades"""
    
    def __init__(self, journal_file='trades_journal.csv'):
        """
        Initialize journal
        
        Args:
            journal_file: CSV file path for storing trades
        """
        self.journal_file = journal_file
        self.trades = self.load_trades()
    
    def load_trades(self):
        """
        Load trades from CSV
        
        Returns:
            DataFrame with trades
        """
        if os.path.exists(self.journal_file):
            df = pd.read_csv(self.journal_file)
            # Convert timestamp columns
            if 'entry_time' in df.columns:
                df['entry_time'] = pd.to_datetime(df['entry_time'])
            if 'exit_time' in df.columns:
                df['exit_time'] = pd.to_datetime(df['exit_time'])
            return df
        else:
            return pd.DataFrame(columns=[
                'trade_id', 'symbol', 'entry_time', 'entry_price', 
                'stop_loss', 'target', 'quantity', 'status',
                'exit_time', 'exit_price', 'pnl', 'pnl_pct',
                'ml_score', 'notes'
            ])
    
    def save_trades(self):
        """Save trades to CSV"""
        self.trades.to_csv(self.journal_file, index=False)
    
    def add_trade(self, signal, notes=''):
        """
        Add new trade entry
        
        Args:
            signal: signal dict with trade details
            notes: optional notes
            
        Returns:
            int: trade_id
        """
        trade_id = len(self.trades) + 1
        
        new_trade = {
            'trade_id': trade_id,
            'symbol': signal['symbol'],
            'entry_time': datetime.now(),
            'entry_price': signal['entry'],
            'stop_loss': signal['stop_loss'],
            'target': signal['target'],
            'quantity': signal['quantity'],
            'status': 'OPEN',
            'exit_time': None,
            'exit_price': None,
            'pnl': None,
            'pnl_pct': None,
            'ml_score': signal['ml_score'],
            'notes': notes
        }
        
        self.trades = pd.concat([self.trades, pd.DataFrame([new_trade])], ignore_index=True)
        self.save_trades()
        
        return trade_id
    
    def close_trade(self, trade_id, exit_price, exit_time=None):
        """
        Close a trade
        
        Args:
            trade_id: ID of trade to close
            exit_price: Exit price
            exit_time: Exit timestamp (default: now)
        """
        if exit_time is None:
            exit_time = datetime.now()
        
        idx = self.trades[self.trades['trade_id'] == trade_id].index
        
        if len(idx) == 0:
            return
        
        idx = idx[0]
        
        entry_price = self.trades.loc[idx, 'entry_price']
        quantity = self.trades.loc[idx, 'quantity']
        
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        self.trades.loc[idx, 'status'] = 'CLOSED'
        self.trades.loc[idx, 'exit_time'] = exit_time
        self.trades.loc[idx, 'exit_price'] = exit_price
        self.trades.loc[idx, 'pnl'] = round(pnl, 2)
        self.trades.loc[idx, 'pnl_pct'] = round(pnl_pct, 2)
        
        self.save_trades()
    
    def get_open_trades(self):
        """
        Get all open trades
        
        Returns:
            DataFrame with open trades
        """
        return self.trades[self.trades['status'] == 'OPEN'].copy()
    
    def get_closed_trades(self):
        """
        Get all closed trades
        
        Returns:
            DataFrame with closed trades
        """
        return self.trades[self.trades['status'] == 'CLOSED'].copy()
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
            dict with metrics
        """
        closed = self.get_closed_trades()
        
        if len(closed) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        winners = closed[closed['pnl'] > 0]
        losers = closed[closed['pnl'] <= 0]
        
        win_rate = len(winners) / len(closed) * 100
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
        total_pnl = closed['pnl'].sum()
        
        # Calculate Sharpe ratio (simplified)
        returns = closed['pnl_pct'].values
        sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        cum_returns = (1 + closed['pnl_pct'] / 100).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'total_trades': len(closed),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'total_pnl': round(total_pnl, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 2)
        }
    
    def get_equity_curve(self):
        """
        Generate equity curve
        
        Returns:
            DataFrame with cumulative PnL
        """
        closed = self.get_closed_trades()
        
        if len(closed) == 0:
            return pd.DataFrame()
        
        closed = closed.sort_values('exit_time')
        closed['cumulative_pnl'] = closed['pnl'].cumsum()
        
        return closed[['exit_time', 'cumulative_pnl', 'pnl']]
