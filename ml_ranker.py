"""
Advanced ML Ranking with Ensemble Models
Combines RandomForest, XGBoost, LightGBM
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False


class AdvancedMLRanker:
    """Ensemble ML model for stock ranking"""
    
    def __init__(self, use_ensemble=True):
        """
        Initialize ML ranker
        
        Args:
            use_ensemble: Use ensemble of multiple models
        """
        self.use_ensemble = use_ensemble and XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE
        
        # RandomForest (baseline)
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        
        if self.use_ensemble:
            # XGBoost
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            # LightGBM
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            
            # Voting ensemble
            self.model = VotingClassifier(
                estimators=[
                    ('rf', self.rf_model),
                    ('xgb', self.xgb_model),
                    ('lgb', self.lgb_model)
                ],
                voting='soft'  # Use probabilities
            )
        else:
            self.model = self.rf_model
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.feature_cols = [
            'rsi', 'macd_diff', 'bb_width', 'atr',
            'volume_ratio', 'stoch_k', 'adx',
            'price_vs_sma20', 'price_vs_sma50',
            'momentum_10', 'momentum_20', 'volatility_20'
        ]
    
    def create_labels(self, df, forward_period=5, threshold=0.02):
        """Create training labels"""
        df = df.copy()
        df['forward_return'] = df['close'].shift(-forward_period) / df['close'] - 1
        df['label'] = (df['forward_return'] > threshold).astype(int)
        df = df[:-forward_period]
        return df
    
    def train(self, df):
        """Train the ensemble model"""
        if df is None or len(df) < 100:
            return None
        
        df = self.create_labels(df)
        
        X = df[self.feature_cols].values
        y = df['label'].values
        
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_score': round(train_score, 3),
            'test_score': round(test_score, 3),
            'samples': len(X),
            'model_type': 'Ensemble' if self.use_ensemble else 'RandomForest'
        }
    
    def predict_proba(self, features):
        """Predict probability"""
        if not self.is_trained:
            return 0.5
        
        if isinstance(features, dict):
            X = np.array([[features[col] for col in self.feature_cols]])
        else:
            X = np.array([features])
        
        if np.isnan(X).any():
            return 0.5
        
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0][1]
        
        return round(proba, 3)
    
    def rank_stocks(self, stocks_features):
        """Rank stocks by ML score"""
        scores = []
        
        for symbol, features in stocks_features.items():
            if features is not None:
                score = self.predict_proba(features)
                scores.append((symbol, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_trained:
            return None
        
        if self.use_ensemble:
            # Average importance across models
            rf_imp = self.rf_model.feature_importances_
            xgb_imp = self.xgb_model.feature_importances_
            lgb_imp = self.lgb_model.feature_importances_
            
            avg_imp = (rf_imp + xgb_imp + lgb_imp) / 3
        else:
            avg_imp = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': avg_imp
        }).sort_values('importance', ascending=False)
        
        return importance_df
