"""
Feature Engineering Module
Creates technical indicators and ML features
"""

import pandas as pd
import numpy as np
import ta


class FeatureEngine:
    """Generates technical features for ML model"""
    
    def __init__(self):
        """Initialize feature engine"""
        pass
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to dataframe
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            DataFrame with added features
        """
        if df is None or len(df) < 50:
            return None
        
        df = df.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], window=14, smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        
        # Price position relative to MAs
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # Momentum
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        return df
    
    def create_ml_features(self, df):
        """
        Create feature set for ML model
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with ML features
        """
        if df is None:
            return None
        
        df = self.add_technical_indicators(df)
        
        if df is None:
            return None
        
        # Select features for ML
        feature_cols = [
            'rsi', 'macd_diff', 'bb_width', 'atr',
            'volume_ratio', 'stoch_k', 'adx',
            'price_vs_sma20', 'price_vs_sma50',
            'momentum_10', 'momentum_20', 'volatility_20'
        ]
        
        # Drop NaN rows
        df = df.dropna(subset=feature_cols)
        
        return df
    
    def get_latest_features(self, df):
        """
        Get latest feature values for prediction
        
        Args:
            df: DataFrame with features
            
        Returns:
            dict of feature values
        """
        if df is None or len(df) == 0:
            return None
        
        latest = df.iloc[-1]
        
        feature_cols = [
            'rsi', 'macd_diff', 'bb_width', 'atr',
            'volume_ratio', 'stoch_k', 'adx',
            'price_vs_sma20', 'price_vs_sma50',
            'momentum_10', 'momentum_20', 'volatility_20'
        ]
        
        features = {}
        for col in feature_cols:
            if col in latest:
                features[col] = latest[col]
        
        return features
