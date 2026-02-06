"""
Broker Integration Module
Supports Zerodha Kite, Upstox, AliceBlue
"""

import os
from datetime import datetime
import streamlit as st

class BrokerIntegration:
    """Unified broker integration"""
    
    def __init__(self, broker_name='zerodha'):
        """
        Initialize broker connection
        
        Args:
            broker_name: 'zerodha', 'upstox', or 'aliceblue'
        """
        self.broker_name = broker_name
        self.api_key = None
        self.access_token = None
        self.connected = False
    
    def connect_zerodha(self, api_key, api_secret):
        """Connect to Zerodha Kite API"""
        try:
            from kiteconnect import KiteConnect
            
            self.kite = KiteConnect(api_key=api_key)
            
            # Generate login URL
            login_url = self.kite.login_url()
            st.info(f"🔐 Login to Zerodha: {login_url}")
            
            # After login, get request token
            request_token = st.text_input("Enter Request Token:")
            
            if request_token:
                data = self.kite.generate_session(
                    request_token, 
                    api_secret=api_secret
                )
                self.access_token = data["access_token"]
                self.kite.set_access_token(self.access_token)
                self.connected = True
                
                return True
            
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
            return False
    
    def place_order(self, symbol, transaction_type, quantity, price, 
                   order_type='LIMIT', product='MIS'):
        """
        Place order via broker
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            transaction_type: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Order price
            order_type: 'LIMIT' or 'MARKET'
            product: 'MIS' (intraday) or 'CNC' (delivery)
        """
        if not self.connected:
            st.error("❌ Broker not connected")
            return None
        
        try:
            if self.broker_name == 'zerodha':
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NSE,
                    tradingsymbol=symbol,
                    transaction_type=transaction_type,
                    quantity=quantity,
                    price=price,
                    order_type=order_type,
                    product=product
                )
                
                st.success(f"✅ Order placed! ID: {order_id}")
                return order_id
                
        except Exception as e:
            st.error(f"❌ Order failed: {str(e)}")
            return None
    
    def place_bracket_order(self, signal):
        """
        Place bracket order (entry + SL + target)
        
        Args:
            signal: Signal dict with entry, stop_loss, target
        """
        try:
            # Main order
            order_id = self.place_order(
                symbol=signal['symbol'].replace('.NS', ''),
                transaction_type='BUY',
                quantity=signal['quantity'],
                price=signal['entry'],
                order_type='LIMIT'
            )
            
            if order_id:
                # Stop loss order
                sl_order = self.place_order(
                    symbol=signal['symbol'].replace('.NS', ''),
                    transaction_type='SELL',
                    quantity=signal['quantity'],
                    price=signal['stop_loss'],
                    order_type='SL'
                )
                
                # Target order
                target_order = self.place_order(
                    symbol=signal['symbol'].replace('.NS', ''),
                    transaction_type='SELL',
                    quantity=signal['quantity'],
                    price=signal['target'],
                    order_type='LIMIT'
                )
                
                return {
                    'entry_order': order_id,
                    'sl_order': sl_order,
                    'target_order': target_order
                }
        
        except Exception as e:
            st.error(f"Bracket order failed: {str(e)}")
            return None
    
    def get_positions(self):
        """Get current open positions"""
        if not self.connected:
            return []
        
        try:
            if self.broker_name == 'zerodha':
                positions = self.kite.positions()
                return positions['net']
        except Exception as e:
            st.error(f"Failed to fetch positions: {str(e)}")
            return []
    
    def get_orders(self):
        """Get today's orders"""
        if not self.connected:
            return []
        
        try:
            if self.broker_name == 'zerodha':
                return self.kite.orders()
        except Exception as e:
            st.error(f"Failed to fetch orders: {str(e)}")
            return []
