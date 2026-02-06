"""
Alert System - WhatsApp/Telegram/Email
"""

import requests
from datetime import datetime
import streamlit as st

class AlertManager:
    """Send trading alerts via multiple channels"""
    
    def __init__(self):
        """Initialize alert manager"""
        self.telegram_bot_token = None
        self.telegram_chat_id = None
        self.whatsapp_enabled = False
    
    def setup_telegram(self, bot_token, chat_id):
        """
        Setup Telegram alerts
        
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
        """
        self.telegram_bot_token = bot_token
        self.telegram_chat_id = chat_id
    
    def send_telegram(self, message):
        """Send Telegram message"""
        if not self.telegram_bot_token:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Telegram alert failed: {str(e)}")
            return False
    
    def send_signal_alert(self, signal):
        """
        Send new signal alert
        
        Args:
            signal: Signal dict
        """
        message = f"""
🎯 <b>NEW TRADING SIGNAL</b>

📊 Symbol: {signal['symbol']}
💯 ML Score: {signal['ml_score']:.2f}

📈 Entry: ₹{signal['entry']:.2f}
🛑 Stop Loss: ₹{signal['stop_loss']:.2f}
🎯 Target: ₹{signal['target']:.2f}

📦 Quantity: {signal['quantity']}
💰 Risk: ₹{signal.get('risk_amount', 0):.0f}

⏰ {datetime.now().strftime('%I:%M %p')}
        """
        
        return self.send_telegram(message.strip())
    
    def send_entry_alert(self, symbol, price):
        """Alert when entry price is hit"""
        message = f"""
🟢 <b>ENTRY TRIGGERED</b>

{symbol} hit entry price!
Price: ₹{price:.2f}

⚡ Execute your order now!
        """
        
        return self.send_telegram(message.strip())
    
    def send_target_alert(self, symbol, exit_price, pnl):
        """Alert when target is hit"""
        message = f"""
🎯 <b>TARGET HIT!</b>

{symbol} reached target!
Exit: ₹{exit_price:.2f}
Profit: ₹{pnl:.2f}

✅ Book profits!
        """
        
        return self.send_telegram(message.strip())
    
    def send_stoploss_alert(self, symbol, exit_price, loss):
        """Alert when stop loss is hit"""
        message = f"""
🛑 <b>STOP LOSS HIT</b>

{symbol} hit stop loss
Exit: ₹{exit_price:.2f}
Loss: ₹{loss:.2f}

⚠️ Exit position
        """
        
        return self.send_telegram(message.strip())
    
    def send_daily_summary(self, metrics):
        """Send end-of-day summary"""
        message = f"""
📊 <b>DAILY SUMMARY</b>

Trades: {metrics.get('total_trades', 0)}
Win Rate: {metrics.get('win_rate', 0):.1f}%
P&L: ₹{metrics.get('total_pnl', 0):.2f}

{datetime.now().strftime('%d %b %Y')}
        """
        
        return self.send_telegram(message.strip())
