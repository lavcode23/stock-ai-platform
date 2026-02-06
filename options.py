"""
Options Trading Module
Analyzes option chains, calculates Greeks
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st

class OptionsAnalyzer:
    """Options chain analysis and Greeks calculation"""
    
    def __init__(self):
        """Initialize options analyzer"""
        self.risk_free_rate = 0.07  # 7% risk-free rate
    
    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate Black-Scholes option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option Greeks
        
        Returns:
            dict with delta, gamma, theta, vega, rho
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        return {
            'delta': round(delta, 4),
            'gamma': round(gamma, 6),
            'theta': round(theta / 365, 2),  # Daily theta
            'vega': round(vega / 100, 2),    # Per 1% change in IV
            'rho': round(rho / 100, 2)       # Per 1% change in rate
        }
    
    def find_atm_strike(self, spot_price, strikes):
        """Find at-the-money strike"""
        return min(strikes, key=lambda x: abs(x - spot_price))
    
    def analyze_option_chain(self, spot_price, strikes, days_to_expiry, iv=0.20):
        """
        Analyze complete option chain
        
        Args:
            spot_price: Current stock price
            strikes: List of strike prices
            days_to_expiry: Days until expiration
            iv: Implied volatility (default 20%)
        
        Returns:
            DataFrame with option chain analysis
        """
        T = days_to_expiry / 365
        
        data = []
        
        for strike in strikes:
            # Call option
            call_price = self.black_scholes(spot_price, strike, T, 
                                           self.risk_free_rate, iv, 'call')
            call_greeks = self.calculate_greeks(spot_price, strike, T,
                                               self.risk_free_rate, iv, 'call')
            
            # Put option
            put_price = self.black_scholes(spot_price, strike, T,
                                          self.risk_free_rate, iv, 'put')
            put_greeks = self.calculate_greeks(spot_price, strike, T,
                                              self.risk_free_rate, iv, 'put')
            
            data.append({
                'strike': strike,
                'call_price': round(call_price, 2),
                'call_delta': call_greeks['delta'],
                'call_theta': call_greeks['theta'],
                'put_price': round(put_price, 2),
                'put_delta': put_greeks['delta'],
                'put_theta': put_greeks['theta'],
                'moneyness': 'ATM' if abs(strike - spot_price) < 50 else 
                            ('ITM' if strike < spot_price else 'OTM')
            })
        
        return pd.DataFrame(data)
    
    def suggest_strategy(self, market_outlook, spot_price, iv_percentile):
        """
        Suggest options strategy based on market outlook
        
        Args:
            market_outlook: 'bullish', 'bearish', 'neutral', 'volatile'
            spot_price: Current price
            iv_percentile: IV percentile (0-100)
        
        Returns:
            dict with strategy recommendation
        """
        strategies = {
            'bullish': {
                'low_iv': 'Buy Call',
                'high_iv': 'Bull Put Spread'
            },
            'bearish': {
                'low_iv': 'Buy Put',
                'high_iv': 'Bear Call Spread'
            },
            'neutral': {
                'low_iv': 'Iron Condor',
                'high_iv': 'Short Straddle'
            },
            'volatile': {
                'low_iv': 'Long Straddle',
                'high_iv': 'Calendar Spread'
            }
        }
        
        iv_category = 'low_iv' if iv_percentile < 50 else 'high_iv'
        strategy_name = strategies[market_outlook][iv_category]
        
        return {
            'strategy': strategy_name,
            'outlook': market_outlook,
            'iv_condition': 'Low' if iv_percentile < 50 else 'High',
            'iv_percentile': iv_percentile
        }
