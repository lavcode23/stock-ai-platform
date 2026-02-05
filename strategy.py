"""
Strategy Module
Generates trading signals with entry, stop, target
"""

import pandas as pd
import numpy as np
from datetime import datetime


class TradingStrategy:
    """Generates trading signals based on ML scores and technical setup"""
    
    def __init__(self, atr_multiplier=2.0, reward_ratio=2.0, min_score=0.6):
        """
        Initialize strategy
        
        Args:
            atr_multiplier: ATR multiplier for stop loss
            reward_ratio: Reward to risk ratio
            min_score: Minimum ML score to generate signal
        """
        self.atr_multiplier = atr_multiplier
        self.reward_ratio = reward_ratio
        self.min_score = min_score
    
    def generate_signal(self, symbol, price_data, features, ml_score):
        """
        Generate trading signal for a symbol
        
        Args:
            symbol: Stock symbol
            price_data: dict with current price info
            features: dict with technical features
            ml_score: ML probability score
            
        Returns:
            dict with signal details or None
        """
        if ml_score < self.min_score:
            return None
        
        if price_data is None or features is None:
            return None
        
        # Get current price and ATR
        ltp = price_data['ltp']
        atr = features.get('atr', 0)
        
        if atr < 5:  # Minimum ATR threshold
            return None
        
        # Calculate entry (can be current price or slightly above)
        entry = round(ltp, 2)
        
        # Calculate stop loss
        stop_loss = round(entry - (self.atr_multiplier * atr), 2)
        
        # Risk per share
        risk_per_share = entry - stop_loss
        
        if risk_per_share <= 0:
            return None
        
        # Calculate target
        target = round(entry + (risk_per_share * self.reward_ratio), 2)
        
        # Create signal
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'entry': entry,
            'stop_loss': stop_loss,
            'target': target,
            'atr': round(atr, 2),
            'risk_per_share': round(risk_per_share, 2),
            'reward_per_share': round(target - entry, 2),
            'ml_score': ml_score,
            'rsi': round(features.get('rsi', 50), 2),
            'adx': round(features.get('adx', 0), 2),
            'status': 'PENDING'
        }
        
        return signal
    
    def filter_signals(self, signals, max_signals=5):
        """
        Filter and prioritize signals
        
        Args:
            signals: list of signal dicts
            max_signals: maximum number of signals to return
            
        Returns:
            list of top signals
        """
        if not signals:
            return []
        
        # Sort by ML score
        sorted_signals = sorted(signals, key=lambda x: x['ml_score'], reverse=True)
        
        # Return top N
        return sorted_signals[:max_signals]
    
    def check_signal_status(self, signal, current_price):
        """
        Check if signal has been triggered
        
        Args:
            signal: signal dict
            current_price: current market price
            
        Returns:
            str: status (WAITING, TRIGGERED, STOPPED, TARGET)
        """
        entry = signal['entry']
        stop = signal['stop_loss']
        target = signal['target']
        
        if current_price >= entry:
            if current_price >= target:
                return 'TARGET'
            elif current_price <= stop:
                return 'STOPPED'
            else:
                return 'TRIGGERED'
        else:
            return 'WAITING'
    
    def get_distance_to_entry(self, signal, current_price):
        """
        Calculate distance to entry price
        
        Args:
            signal: signal dict
            current_price: current price
            
        Returns:
            float: percentage distance
        """
        entry = signal['entry']
        distance_pct = ((current_price - entry) / entry) * 100
        
        return round(distance_pct, 2)
    
    def get_action_message(self, signal, current_price):
        """
        Get user action message based on signal status
        
        Args:
            signal: signal dict
            current_price: current price
            
        Returns:
            str: action message
        """
        status = self.check_signal_status(signal, current_price)
        
        if status == 'WAITING':
            distance = self.get_distance_to_entry(signal, current_price)
            return f"Wait for entry at {signal['entry']} (currently {distance:+.2f}%)"
        elif status == 'TRIGGERED':
            return f"Position active - Hold until {signal['target']} or SL {signal['stop_loss']}"
        elif status == 'TARGET':
            return "EXIT - Target reached!"
        elif status == 'STOPPED':
            return "EXIT - Stop loss hit"
        else:
            return "Monitor"
