"""
ML Ranking Module
Trains and uses RandomForest to rank stocks
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MLRanker:
    """Machine Learning model for ranking stocks"""
    
    def __init__(self):
        """Initialize ML ranker"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = [
            'rsi', 'macd_diff', 'bb_width', 'atr',
            'volume_ratio', 'stoch_k', 'adx',
            'price_vs_sma20', 'price_vs_sma50',
            'momentum_10', 'momentum_20', 'volatility_20'
        ]
    
    def create_labels(self, df, forward_period=5, threshold=0.02):
        """
        Create training labels based on forward returns
        
        Args:
            df: DataFrame with features
            forward_period: Days to look forward
            threshold: Minimum return to be labeled as 1
            
        Returns:
            DataFrame with labels
        """
        df = df.copy()
        
        # Calculate forward returns
        df['forward_return'] = df['close'].shift(-forward_period) / df['close'] - 1
        
        # Create binary labels
        df['label'] = (df['forward_return'] > threshold).astype(int)
        
        # Remove rows without forward data
        df = df[:-forward_period]
        
        return df
    
    def train(self, df):
        """
        Train the ML model
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            dict with training metrics
        """
        if df is None or len(df) < 100:
            return None
        
        # Create labels
        df = self.create_labels(df)
        
        # Prepare features and labels
        X = df[self.feature_cols].values
        y = df['label'].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_score': round(train_score, 3),
            'test_score': round(test_score, 3),
            'samples': len(X)
        }
    
    def predict_proba(self, features):
        """
        Predict probability for a set of features
        
        Args:
            features: dict or array of feature values
            
        Returns:
            float probability score
        """
        if not self.is_trained:
            return 0.5
        
        # Convert dict to array if needed
        if isinstance(features, dict):
            X = np.array([[features[col] for col in self.feature_cols]])
        else:
            X = np.array([features])
        
        # Handle NaN
        if np.isnan(X).any():
            return 0.5
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0][1]
        
        return round(proba, 3)
    
    def rank_stocks(self, stocks_features):
        """
        Rank multiple stocks based on ML score
        
        Args:
            stocks_features: dict of {symbol: features_dict}
            
        Returns:
            list of tuples (symbol, score) sorted by score
        """
        scores = []
        
        for symbol, features in stocks_features.items():
            if features is not None:
                score = self.predict_proba(features)
                scores.append((symbol, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
