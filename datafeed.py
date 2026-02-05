"""
Data Feed Module
Fetches live and historical stock data using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st


class DataFeed:
    """Handles all data fetching operations"""
    
    def __init__(self, symbols, index_symbol="^NSEI"):
        """
        Initialize DataFeed
        
        Args:
            symbols: List of stock symbols
            index_symbol: Index symbol for market regime
        """
        self.symbols = symbols
        self.index_symbol = index_symbol
        
    @st.cache_data(ttl=600)
    def fetch_historical_data(_self, symbol, period="3mo", interval="1d"):
        """
        Fetch historical data for a symbol
        
        Args:
            symbol: Stock symbol
            period: Data period (1mo, 3mo, 6mo, 1y, etc.)
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
                
            df.reset_index(inplace=True)
            df.columns = [col.lower() for col in df.columns]
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching {symbol}: {str(e)}")
            return None
    
    def fetch_live_price(self, symbol):
        """
        Fetch current live price
        
        Args:
            symbol: Stock symbol
            
        Returns:
            dict with live price info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get latest price
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                return None
            
            ltp = hist['Close'].iloc[-1]
            open_price = info.get('open', hist['Open'].iloc[0])
            prev_close = info.get('previousClose', ltp)
            
            return {
                'symbol': symbol,
                'ltp': round(ltp, 2),
                'open': round(open_price, 2),
                'prev_close': round(prev_close, 2),
                'change': round(ltp - prev_close, 2),
                'change_pct': round(((ltp - prev_close) / prev_close) * 100, 2),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return None
    
    def fetch_batch_live_prices(self, symbols):
        """
        Fetch live prices for multiple symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            dict of symbol: price_info
        """
        prices = {}
        for symbol in symbols:
            price_data = self.fetch_live_price(symbol)
            if price_data:
                prices[symbol] = price_data
        
        return prices
    
    def get_market_regime(self):
        """
        Determine market regime using NIFTY moving averages
        
        Returns:
            str: 'BULL', 'BEAR', or 'SIDEWAYS'
        """
        try:
            df = self.fetch_historical_data(self.index_symbol, period="6mo", interval="1d")
            
            if df is None or len(df) < 50:
                return "UNKNOWN"
            
            # Calculate moving averages
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma50'] = df['close'].rolling(window=50).mean()
            
            current_price = df['close'].iloc[-1]
            ma20 = df['ma20'].iloc[-1]
            ma50 = df['ma50'].iloc[-1]
            
            # Determine regime
            if current_price > ma20 > ma50:
                return "BULL"
            elif current_price < ma20 < ma50:
                return "BEAR"
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            return "UNKNOWN"
    
    def get_index_data(self):
        """
        Get current NIFTY index data
        
        Returns:
            dict with index info
        """
        try:
            df = self.fetch_historical_data(self.index_symbol, period="5d", interval="1d")
            
            if df is None or len(df) < 2:
                return None
            
            current = df['close'].iloc[-1]
            previous = df['close'].iloc[-2]
            change = current - previous
            change_pct = (change / previous) * 100
            
            return {
                'value': round(current, 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2),
                'regime': self.get_market_regime()
            }
            
        except Exception as e:
            return None
