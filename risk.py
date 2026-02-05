"""
Risk Management Module
Calculates position size and enforces risk rules
"""

import pandas as pd
from datetime import datetime, date


class RiskManager:
    """Manages position sizing and risk limits"""
    
    def __init__(self, account_size, daily_risk_pct=1.0, max_trades_per_day=3):
        """
        Initialize risk manager
        
        Args:
            account_size: Total account capital
            daily_risk_pct: Max daily risk as percentage
            max_trades_per_day: Maximum trades allowed per day
        """
        self.account_size = account_size
        self.daily_risk_pct = daily_risk_pct
        self.max_trades_per_day = max_trades_per_day
        self.daily_risk_amount = account_size * (daily_risk_pct / 100)
    
    def calculate_position_size(self, entry, stop_loss, risk_per_trade_pct=0.5):
        """
        Calculate position size based on risk
        
        Args:
            entry: Entry price
            stop_loss: Stop loss price
            risk_per_trade_pct: Risk per trade as % of account
            
        Returns:
            int: number of shares to buy
        """
        risk_per_share = entry - stop_loss
        
        if risk_per_share <= 0:
            return 0
        
        # Risk amount for this trade
        risk_amount = self.account_size * (risk_per_trade_pct / 100)
        
        # Calculate quantity
        quantity = int(risk_amount / risk_per_share)
        
        return max(1, quantity)
    
    def add_position_size_to_signal(self, signal):
        """
        Add position size calculation to signal
        
        Args:
            signal: signal dict
            
        Returns:
            signal dict with quantity added
        """
        quantity = self.calculate_position_size(
            signal['entry'],
            signal['stop_loss'],
            risk_per_trade_pct=0.5
        )
        
        signal['quantity'] = quantity
        signal['position_value'] = round(signal['entry'] * quantity, 2)
        signal['risk_amount'] = round(signal['risk_per_share'] * quantity, 2)
        signal['reward_amount'] = round(signal['reward_per_share'] * quantity, 2)
        
        return signal
    
    def check_daily_limits(self, today_trades):
        """
        Check if daily trading limits are reached
        
        Args:
            today_trades: list of trades executed today
            
        Returns:
            dict with limit status
        """
        today = date.today()
        
        # Filter trades for today
        trades_today = [
            t for t in today_trades 
            if isinstance(t.get('entry_time'), datetime) and t['entry_time'].date() == today
        ]
        
        num_trades = len(trades_today)
        
        # Calculate total risk taken
        total_risk = sum([t.get('risk_amount', 0) for t in trades_today])
        
        can_trade = (
            num_trades < self.max_trades_per_day and
            total_risk < self.daily_risk_amount
        )
        
        return {
            'can_trade': can_trade,
            'trades_taken': num_trades,
            'max_trades': self.max_trades_per_day,
            'risk_used': round(total_risk, 2),
            'max_risk': round(self.daily_risk_amount, 2),
            'risk_remaining': round(self.daily_risk_amount - total_risk, 2)
        }
    
    def get_portfolio_summary(self, trades):
        """
        Get portfolio risk summary
        
        Args:
            trades: list of all trades
            
        Returns:
            dict with portfolio metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'open_positions': 0,
                'total_risk': 0,
                'total_capital_used': 0
            }
        
        open_trades = [t for t in trades if t.get('status') == 'OPEN']
        
        total_risk = sum([t.get('risk_amount', 0) for t in open_trades])
        total_capital = sum([t.get('position_value', 0) for t in open_trades])
        
        return {
            'total_trades': len(trades),
            'open_positions': len(open_trades),
            'total_risk': round(total_risk, 2),
            'total_capital_used': round(total_capital, 2),
            'risk_pct_of_account': round((total_risk / self.account_size) * 100, 2)
        }
