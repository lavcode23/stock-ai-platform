"""
Trading Assistant - OPTIMIZED Fast-Loading Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import time
from functools import lru_cache

# Import custom modules
from datafeed import DataFeed
from features import FeatureEngine
from ml_ranker import MLRanker
from strategy import TradingStrategy
from risk import RiskManager
from journal import TradingJournal

# Page configuration
st.set_page_config(
    page_title="Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from YAML"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize components (cached)
@st.cache_resource
def initialize_components():
    """Initialize all trading components - CACHED"""
    datafeed = DataFeed(config['symbols']['stocks'], config['symbols']['index'])
    feature_engine = FeatureEngine()
    ml_ranker = MLRanker()
    strategy = TradingStrategy(
        atr_multiplier=config['risk']['atr_stop_multiplier'],
        reward_ratio=config['risk']['reward_ratio'],
        min_score=config['ml']['min_score']
    )
    risk_manager = RiskManager(
        account_size=config['account']['initial_capital'],
        daily_risk_pct=config['account']['daily_risk_percent'],
        max_trades_per_day=config['account']['max_trades_per_day']
    )
    journal = TradingJournal()
    
    return datafeed, feature_engine, ml_ranker, strategy, risk_manager, journal

datafeed, feature_engine, ml_ranker, strategy, risk_manager, journal = initialize_components()

# Cache expensive operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_index_data_cached():
    """Get NIFTY data - cached"""
    return datafeed.get_index_data()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def train_model_cached():
    """Train ML model - cached for 30 min"""
    nifty_df = datafeed.fetch_historical_data(
        config['symbols']['index'], 
        period="2mo"  # Reduced from 3mo
    )
    
    if nifty_df is not None:
        nifty_features = feature_engine.create_ml_features(nifty_df)
        if nifty_features is not None:
            return ml_ranker.train(nifty_features)
    return None

# Sidebar (lightweight)
with st.sidebar:
    st.title("📈 Trading Assistant")
    st.markdown("---")
    
    # Account info
    st.subheader("Account")
    st.metric("Capital", f"₹{config['account']['initial_capital']:,.0f}")
    st.metric("Daily Risk", f"{config['account']['daily_risk_percent']}%")
    
    st.markdown("---")
    
    # Market status (cached)
    with st.spinner("Loading market data..."):
        index_data = get_index_data_cached()
    
    if index_data:
        st.subheader("NIFTY")
        st.metric(
            "Value",
            f"{index_data['value']:,.2f}",
            f"{index_data['change']:+.2f} ({index_data['change_pct']:+.2f}%)"
        )
        
        regime = index_data['regime']
        color = {'BULL': '🟢', 'BEAR': '🔴', 'SIDEWAYS': '🟡'}.get(regime, '⚪')
        st.info(f"Regime: {color} {regime}")
    
    st.markdown("---")
    st.caption("⚠️ Educational purposes only")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 TODAY", "🔴 MONITOR", "📓 JOURNAL", "⚙️ SETTINGS", "❓ HELP"
])

# TAB 1: TODAY (Optimized)
with tab1:
    st.header("Today's Trading Plan")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if index_data:
            regime = index_data['regime']
            if regime == 'BULL':
                st.success(f"🟢 Market: **{regime}**")
            elif regime == 'BEAR':
                st.error(f"🔴 Market: **{regime}**")
            else:
                st.warning(f"🟡 Market: **{regime}**")
    
    with col2:
        limits = risk_manager.check_daily_limits(journal.trades.to_dict('records'))
        st.metric("Trades", f"{limits['trades_taken']}/{limits['max_trades']}")
    
    with col3:
        st.metric("Risk", f"₹{limits['risk_used']:,.0f}")
    
    st.markdown("---")
    
    # Generate signals button
    if st.button("🎯 Generate Signals (Fast Mode)", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Step 1: Train model (cached)
        status.info("⚙️ Training ML model...")
        progress_bar.progress(20)
        
        ml_metrics = train_model_cached()
        if ml_metrics:
            status.success(f"✅ Model ready (Score: {ml_metrics['test_score']})")
        
        # Step 2: Fetch limited stocks (top 10 only for speed)
        status.info("📊 Analyzing top stocks...")
        progress_bar.progress(40)
        
        # Use only top 10 liquid stocks for speed
        top_stocks = config['symbols']['stocks'][:10]
        
        stocks_features = {}
        for idx, symbol in enumerate(top_stocks):
            df = datafeed.fetch_historical_data(symbol, period="2mo")
            if df is not None:
                features_df = feature_engine.create_ml_features(df)
                if features_df is not None:
                    latest_features = feature_engine.get_latest_features(features_df)
                    if latest_features:
                        stocks_features[symbol] = latest_features
            
            progress_bar.progress(40 + int((idx / len(top_stocks)) * 40))
        
        # Step 3: Rank
        status.info("🔍 Ranking stocks...")
        progress_bar.progress(85)
        
        ranked = ml_ranker.rank_stocks(stocks_features)
        
        # Step 4: Generate signals for top 5
        status.info("🎯 Generating signals...")
        progress_bar.progress(90)
        
        signals = []
        live_prices = datafeed.fetch_batch_live_prices([s[0] for s in ranked[:5]])
        
        for symbol, score in ranked[:5]:
            if symbol in live_prices and symbol in stocks_features:
                signal = strategy.generate_signal(
                    symbol,
                    live_prices[symbol],
                    stocks_features[symbol],
                    score
                )
                if signal:
                    signal = risk_manager.add_position_size_to_signal(signal)
                    signals.append(signal)
        
        progress_bar.progress(100)
        status.success(f"✅ Done! Found {len(signals)} signals")
        
        st.session_state['today_signals'] = signals
        
        time.sleep(1)
        progress_bar.empty()
        status.empty()
    
    # Display signals
    if 'today_signals' in st.session_state and st.session_state['today_signals']:
        st.subheader(f"📋 Signals ({len(st.session_state['today_signals'])})")
        
        signals_df = pd.DataFrame(st.session_state['today_signals'])
        
        display_df = signals_df[[
            'symbol', 'ml_score', 'entry', 'stop_loss', 'target',
            'quantity', 'risk_amount'
        ]].copy()
        
        display_df.columns = [
            'Symbol', 'Score', 'Entry', 'Stop', 'Target', 'Qty', 'Risk ₹'
        ]
        
        st.dataframe(
            display_df.style.format({
                'Score': '{:.2f}',
                'Entry': '₹{:.2f}',
                'Stop': '₹{:.2f}',
                'Target': '₹{:.2f}',
                'Risk ₹': '₹{:.0f}'
            }).background_gradient(subset=['Score'], cmap='RdYlGn'),
            use_container_width=True,
            height=300
        )
        
        st.info("💡 Execute manually in your broker app")

# TAB 2: MONITOR (Simplified - no auto-refresh)
with tab2:
    st.header("🔴 Live Monitor")
    
    if 'today_signals' not in st.session_state or not st.session_state['today_signals']:
        st.info("👈 Generate signals in TODAY tab first")
    else:
        if st.button("🔄 Refresh Prices"):
            signals = st.session_state['today_signals']
            symbols = [s['symbol'] for s in signals]
            
            with st.spinner("Fetching live prices..."):
                live_prices = datafeed.fetch_batch_live_prices(symbols)
            
            monitor_data = []
            
            for signal in signals:
                symbol = signal['symbol']
                
                if symbol in live_prices:
                    ltp = live_prices[symbol]['ltp']
                    status = strategy.check_signal_status(signal, ltp)
                    distance = strategy.get_distance_to_entry(signal, ltp)
                    
                    monitor_data.append({
                        'Symbol': symbol,
                        'LTP': ltp,
                        'Entry': signal['entry'],
                        'Status': status,
                        'Distance %': distance
                    })
            
            if monitor_data:
                monitor_df = pd.DataFrame(monitor_data)
                
                def color_status(val):
                    colors = {
                        'WAITING': 'background-color: #FFF3CD',
                        'TRIGGERED': 'background-color: #D1ECF1',
                        'TARGET': 'background-color: #D4EDDA',
                        'STOPPED': 'background-color: #F8D7DA'
                    }
                    return colors.get(val, '')
                
                st.dataframe(
                    monitor_df.style.applymap(color_status, subset=['Status']).format({
                        'LTP': '₹{:.2f}',
                        'Entry': '₹{:.2f}',
                        'Distance %': '{:+.2f}%'
                    }),
                    use_container_width=True
                )
        
        st.caption("Click refresh button to update prices (no auto-refresh for speed)")

# TAB 3: JOURNAL
with tab4:
    st.header("📓 Trading Journal")
    
    # Add trade
    with st.expander("➕ Add Trade"):
        col1, col2 = st.columns(2)
        
        with col1:
            trade_symbol = st.selectbox("Symbol", config['symbols']['stocks'][:10])
            entry_price = st.number_input("Entry", min_value=0.0, step=0.01)
            quantity = st.number_input("Qty", min_value=1, step=1)
        
        with col2:
            stop_loss = st.number_input("Stop", min_value=0.0, step=0.01)
            target = st.number_input("Target", min_value=0.0, step=0.01)
        
        if st.button("Add"):
            signal = {
                'symbol': trade_symbol,
                'entry': entry_price,
                'stop_loss': stop_loss,
                'target': target,
                'quantity': quantity,
                'ml_score': 0.0
            }
            journal.add_trade(signal)
            st.success("✅ Added")
            st.rerun()
    
    # Metrics
    metrics = journal.calculate_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Trades", metrics['total_trades'])
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    with col3:
        st.metric("P&L", f"₹{metrics['total_pnl']:,.0f}")
    with col4:
        st.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    
    # Closed trades
    st.subheader("Trades")
    closed = journal.get_closed_trades()
    
    if len(closed) > 0:
        st.dataframe(
            closed[['symbol', 'entry_price', 'exit_price', 'pnl']].tail(10),
            use_container_width=True
        )
        
        # Equity curve
        equity = journal.get_equity_curve()
        if len(equity) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity['exit_time'],
                y=equity['cumulative_pnl'],
                mode='lines',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(title='Equity Curve', height=300)
            st.plotly_chart(fig, use_container_width=True)

# TAB 4: SETTINGS
with tab4:
    st.header("⚙️ Settings")
    
    st.info("Edit `config.yaml` file to change settings, then restart app")
    
    st.code(f"""
Current Settings:
- Account Size: ₹{config['account']['initial_capital']:,}
- Daily Risk: {config['account']['daily_risk_percent']}%
- Max Trades: {config['account']['max_trades_per_day']}
- ATR Multiplier: {config['risk']['atr_stop_multiplier']}
- Reward Ratio: {config['risk']['reward_ratio']}
    """)

# TAB 5: HELP
with tab5:
    st.header("❓ Quick Help")
    
    st.markdown("""
    ## How to Use
    
    ### 1. Generate Signals
    - Go to TODAY tab
    - Click "Generate Signals"
    - Wait 10-15 seconds
    - Review top 5 signals
    
    ### 2. Monitor
    - Go to MONITOR tab
    - Click "Refresh" to update prices
    - Check if entry price hit
    
    ### 3. Execute
    - Open Zerodha/Upstox app
    - Place orders manually
    - Record in JOURNAL tab
    
    ### 4. Track Performance
    - Add trades to journal
    - View equity curve
    - Monitor win rate
    
    ---
    
    ## Performance Tips
    
    ✅ App analyzes only top 10 stocks (for speed)
    ✅ Data cached for 30 minutes
    ✅ No auto-refresh (click manually)
    ✅ Reduced lookback period (2 months)
    
    ---
    
    ⚠️ **Educational tool only**
    """)

st.markdown("---")
st.caption("Trading Assistant v1.0 - Optimized")
