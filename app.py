"""
Trading Assistant - PREMIUM VERSION
Complete AI-powered trading platform for Indian stocks
Features: Broker Integration, Alerts, Options, Advanced ML
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
import os

# Import custom modules
from datafeed import DataFeed
from features import FeatureEngine
from strategy import TradingStrategy
from risk import RiskManager
from journal import TradingJournal
from backtest import Backtester

# Import new premium modules (will create fallbacks if not available)
try:
    from broker_integration import BrokerIntegration
    BROKER_AVAILABLE = True
except:
    BROKER_AVAILABLE = False

try:
    from alerts import AlertManager
    ALERTS_AVAILABLE = True
except:
    ALERTS_AVAILABLE = False

try:
    from options import OptionsAnalyzer
    OPTIONS_AVAILABLE = True
except:
    OPTIONS_AVAILABLE = False

# Try to use advanced ML, fallback to basic
try:
    from ml_ranker import AdvancedMLRanker as MLRanker
    ML_TYPE = "Advanced Ensemble"
except:
    from ml_ranker import MLRanker
    ML_TYPE = "RandomForest"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Trading Assistant Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@st.cache_resource
def load_config():
    """Load configuration from YAML"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        # Fallback configuration if file not found
        return {
            'account': {
                'initial_capital': 100000,
                'daily_risk_percent': 1.0,
                'max_trades_per_day': 3
            },
            'risk': {
                'atr_stop_multiplier': 2.0,
                'reward_ratio': 2.0,
                'min_atr': 5.0
            },
            'symbols': {
                'index': '^NSEI',
                'stocks': [
                    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
                    'ICICIBANK.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS',
                    'ITC.NS', 'KOTAKBANK.NS', 'SBIN.NS'
                ]
            },
            'ml': {
                'min_score': 0.6
            }
        }

config = load_config()

# ============================================================================
# COMPONENT INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_components():
    """Initialize all trading components"""
    
    # Core components
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
    
    # Premium components (optional)
    broker = BrokerIntegration() if BROKER_AVAILABLE else None
    alerts = AlertManager() if ALERTS_AVAILABLE else None
    options_analyzer = OptionsAnalyzer() if OPTIONS_AVAILABLE else None
    
    return (datafeed, feature_engine, ml_ranker, strategy, 
            risk_manager, journal, broker, alerts, options_analyzer)

# Initialize all components
(datafeed, feature_engine, ml_ranker, strategy, 
 risk_manager, journal, broker, alerts, options_analyzer) = initialize_components()

# ============================================================================
# CACHED DATA OPERATIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_index_data_cached():
    """Get NIFTY data - cached"""
    return datafeed.get_index_data()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def train_model_cached():
    """Train ML model - cached for 30 min"""
    nifty_df = datafeed.fetch_historical_data(
        config['symbols']['index'], 
        period="2mo"
    )
    
    if nifty_df is not None:
        nifty_features = feature_engine.create_ml_features(nifty_df)
        if nifty_features is not None:
            return ml_ranker.train(nifty_features)
    return None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown('<h1 style="color: #1f77b4;">📈 Trading Assistant Pro</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Account information
    st.subheader("💰 Account")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Capital", f"₹{config['account']['initial_capital']:,.0f}")
    with col2:
        st.metric("Daily Risk", f"{config['account']['daily_risk_percent']}%")
    
    # ML Model info
    st.info(f"🤖 AI Model: {ML_TYPE}")
    
    st.markdown("---")
    
    # Market status
    st.subheader("📊 Market Status")
    
    with st.spinner("Loading market data..."):
        index_data = get_index_data_cached()
    
    if index_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "NIFTY",
                f"{index_data['value']:,.0f}",
                f"{index_data['change']:+.0f}"
            )
        
        with col2:
            change_pct = index_data['change_pct']
            st.metric(
                "Change",
                f"{change_pct:+.2f}%",
                delta_color="normal"
            )
        
        regime = index_data['regime']
        regime_colors = {
            'BULL': '🟢',
            'BEAR': '🔴',
            'SIDEWAYS': '🟡',
            'UNKNOWN': '⚪'
        }
        
        regime_emoji = regime_colors.get(regime, '⚪')
        
        if regime == 'BULL':
            st.success(f"{regime_emoji} Market Regime: **{regime}**")
        elif regime == 'BEAR':
            st.error(f"{regime_emoji} Market Regime: **{regime}**")
        else:
            st.warning(f"{regime_emoji} Market Regime: **{regime}**")
    
    st.markdown("---")
    
    # Premium features status
    st.subheader("🌟 Premium Features")
    
    features_status = {
        "Broker Integration": BROKER_AVAILABLE,
        "Alerts System": ALERTS_AVAILABLE,
        "Options Analysis": OPTIONS_AVAILABLE
    }
    
    for feature, available in features_status.items():
        if available:
            st.success(f"✅ {feature}")
        else:
            st.info(f"⚪ {feature} (Install module)")
    
    st.markdown("---")
    
    # Quick stats
    metrics = journal.calculate_metrics()
    
    st.subheader("📈 Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Trades", metrics['total_trades'])
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']:.0f}%")
    
    if metrics['total_trades'] > 0:
        st.metric("Total P&L", f"₹{metrics['total_pnl']:,.0f}")
    
    st.markdown("---")
    st.caption("⚠️ For educational purposes only")
    st.caption("Not financial advice")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Create tabs based on available features
tab_list = ["📊 TODAY", "🔴 LIVE MONITOR", "📓 JOURNAL", "📉 BACKTEST"]

if OPTIONS_AVAILABLE:
    tab_list.append("📈 OPTIONS")
if ALERTS_AVAILABLE:
    tab_list.append("🔔 ALERTS")
if BROKER_AVAILABLE:
    tab_list.append("🔌 BROKER")

tab_list.extend(["⚙️ SETTINGS", "❓ HELP"])

tabs = st.tabs(tab_list)
tab_idx = 0

# ============================================================================
# TAB 1: TODAY
# ============================================================================

with tabs[tab_idx]:
    tab_idx += 1
    
    st.header("📊 Today's Trading Plan")
    
    # Market overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if index_data:
            regime = index_data['regime']
            if regime == 'BULL':
                st.success(f"🟢 **{regime}**")
            elif regime == 'BEAR':
                st.error(f"🔴 **{regime}**")
            else:
                st.warning(f"🟡 **{regime}**")
        else:
            st.info("⚪ Loading...")
    
    with col2:
        limits = risk_manager.check_daily_limits(journal.trades.to_dict('records'))
        st.metric("Trades Today", f"{limits['trades_taken']}/{limits['max_trades']}")
    
    with col3:
        st.metric("Risk Used", f"₹{limits['risk_used']:,.0f}")
    
    with col4:
        st.metric("Risk Remaining", f"₹{limits['risk_remaining']:,.0f}")
    
    st.markdown("---")
    
    # Signal generation section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Generate Signals")
        
        num_stocks = st.slider(
            "Number of stocks to analyze",
            min_value=5,
            max_value=25,
            value=10,
            help="More stocks = longer analysis time"
        )
    
    with col2:
        st.subheader("⚡ Quick Actions")
        
        if limits['can_trade']:
            button_type = "primary"
            button_label = "🎯 Generate Signals"
        else:
            button_type = "secondary"
            button_label = "⚠️ Daily Limit Reached"
    
    generate_signals = st.button(
        button_label,
        type=button_type,
        disabled=not limits['can_trade'],
        use_container_width=True
    )
    
    if generate_signals:
        # Progress tracking
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Step 1: Train model
        status.info("⚙️ Training AI model...")
        progress_bar.progress(15)
        
        ml_metrics = train_model_cached()
        
        if ml_metrics:
            status.success(f"✅ Model ready - {ml_metrics.get('model_type', 'ML')} (Score: {ml_metrics['test_score']})")
        else:
            status.warning("⚠️ Using untrained model")
        
        time.sleep(0.5)
        
        # Step 2: Fetch and analyze stocks
        status.info("📊 Analyzing stocks...")
        progress_bar.progress(30)
        
        stocks_to_analyze = config['symbols']['stocks'][:num_stocks]
        stocks_features = {}
        
        for idx, symbol in enumerate(stocks_to_analyze):
            df = datafeed.fetch_historical_data(symbol, period="2mo")
            if df is not None:
                features_df = feature_engine.create_ml_features(df)
                if features_df is not None:
                    latest_features = feature_engine.get_latest_features(features_df)
                    if latest_features:
                        stocks_features[symbol] = latest_features
            
            # Update progress
            progress = 30 + int((idx / len(stocks_to_analyze)) * 50)
            progress_bar.progress(progress)
            status.info(f"📊 Analyzed {idx + 1}/{len(stocks_to_analyze)} stocks...")
        
        # Step 3: Rank stocks
        status.info("🔍 Ranking stocks by AI score...")
        progress_bar.progress(85)
        
        ranked = ml_ranker.rank_stocks(stocks_features)
        
        # Step 4: Generate signals
        status.info("🎯 Generating trading signals...")
        progress_bar.progress(90)
        
        signals = []
        top_stocks = ranked[:5]
        live_prices = datafeed.fetch_batch_live_prices([s[0] for s in top_stocks])
        
        for symbol, score in top_stocks:
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
        status.success(f"✅ Analysis complete! Found {len(signals)} signals")
        
        # Store in session state
        st.session_state['today_signals'] = signals
        st.session_state['all_ranked'] = ranked
        
        # Send alerts if configured
        if ALERTS_AVAILABLE and alerts and len(signals) > 0:
            for signal in signals:
                alerts.send_signal_alert(signal)
        
        time.sleep(1)
        progress_bar.empty()
        status.empty()
        
        st.rerun()
    
    # Display signals
    if 'today_signals' in st.session_state and st.session_state['today_signals']:
        st.markdown("---")
        st.subheader(f"📋 Trading Signals ({len(st.session_state['today_signals'])})")
        
        signals = st.session_state['today_signals']
        
        # Create signals dataframe
        signals_df = pd.DataFrame(signals)
        
        # Display as cards for top signals
        for idx, signal in enumerate(signals[:3]):
            with st.expander(f"🎯 #{idx + 1} - {signal['symbol']} (Score: {signal['ml_score']:.2f})", expanded=(idx == 0)):
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Entry", f"₹{signal['entry']:.2f}")
                    st.metric("ML Score", f"{signal['ml_score']:.2f}")
                
                with col2:
                    st.metric("Stop Loss", f"₹{signal['stop_loss']:.2f}")
                    st.metric("RSI", f"{signal.get('rsi', 0):.1f}")
                
                with col3:
                    st.metric("Target", f"₹{signal['target']:.2f}")
                    st.metric("ADX", f"{signal.get('adx', 0):.1f}")
                
                with col4:
                    st.metric("Quantity", signal['quantity'])
                    st.metric("Risk", f"₹{signal.get('risk_amount', 0):.0f}")
                
                # Risk/Reward visualization
                risk = signal['entry'] - signal['stop_loss']
                reward = signal['target'] - signal['entry']
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Risk', 'Reward'],
                    y=[risk, reward],
                    marker_color=['#ff4444', '#44ff44'],
                    text=[f'₹{risk:.2f}', f'₹{reward:.2f}'],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Risk vs Reward",
                    height=250,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Table view for all signals
        st.markdown("### 📊 All Signals")
        
        display_df = signals_df[[
            'symbol', 'ml_score', 'entry', 'stop_loss', 'target',
            'quantity', 'risk_amount', 'reward_amount'
        ]].copy()
        
        display_df.columns = [
            'Symbol', 'Score', 'Entry', 'Stop', 'Target',
            'Qty', 'Risk ₹', 'Reward ₹'
        ]
        
        st.dataframe(
            display_df.style.format({
                'Score': '{:.2f}',
                'Entry': '₹{:.2f}',
                'Stop': '₹{:.2f}',
                'Target': '₹{:.2f}',
                'Risk ₹': '₹{:.0f}',
                'Reward ₹': '₹{:.0f}'
            }).background_gradient(subset=['Score'], cmap='RdYlGn'),
            use_container_width=True,
            height=300
        )
        
        # Pre-trade checklist
        st.markdown("---")
        st.subheader("✅ Pre-Trade Checklist")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.checkbox("☑️ Market regime favorable", value=True)
            st.checkbox("☑️ Risk limits checked")
        
        with col2:
            st.checkbox("☑️ Signals reviewed")
            st.checkbox("☑️ Stop losses noted")
        
        with col3:
            st.checkbox("☑️ Position sizes calculated")
            st.checkbox("☑️ Ready to execute")
        
        # Download signals as CSV
        csv = signals_df.to_csv(index=False)
        st.download_button(
            "📥 Download Signals CSV",
            csv,
            "signals.csv",
            "text/csv",
            key='download-csv'
        )
    
    else:
        st.info("👆 Click 'Generate Signals' to start analysis")
        
        # Show example
        with st.expander("ℹ️ What will you get?"):
            st.markdown("""
            **The AI will provide:**
            - 🎯 Top 5 stocks ranked by ML score
            - 💰 Exact entry price
            - 🛑 ATR-based stop loss
            - 🎯 Target price (2:1 R:R)
            - 📦 Position size (risk-based)
            - 📊 Technical indicators (RSI, ADX)
            
            **You decide:**
            - Whether to take the trade
            - When to enter (market/limit order)
            - Manual execution in your broker
            """)

# ============================================================================
# TAB 2: LIVE MONITOR
# ============================================================================

with tabs[tab_idx]:
    tab_idx += 1
    
    st.header("🔴 Live Price Monitor")
    
    if 'today_signals' not in st.session_state or not st.session_state['today_signals']:
        st.info("👈 Generate signals in the TODAY tab first")
        
        st.markdown("### 💡 What is Live Monitor?")
        st.markdown("""
        - Real-time price tracking for your signals
        - Auto-detect entry triggers
        - Monitor stop loss and target levels
        - Get instant action recommendations
        - Track distance to entry price
        """)
    
    else:
        # Refresh controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
        
        with col2:
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.rerun()
        
        with col3:
            st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # Fetch live prices
        signals = st.session_state['today_signals']
        symbols = [s['symbol'] for s in signals]
        
        with st.spinner("Fetching live prices..."):
            live_prices = datafeed.fetch_batch_live_prices(symbols)
        
        # Monitor data
        monitor_data = []
        
        for signal in signals:
            symbol = signal['symbol']
            
            if symbol in live_prices:
                ltp = live_prices[symbol]['ltp']
                status = strategy.check_signal_status(signal, ltp)
                distance = strategy.get_distance_to_entry(signal, ltp)
                action = strategy.get_action_message(signal, ltp)
                
                # Calculate potential P&L if triggered
                if status == 'TRIGGERED':
                    current_pnl = (ltp - signal['entry']) * signal['quantity']
                else:
                    current_pnl = 0
                
                monitor_data.append({
                    'Symbol': symbol,
                    'LTP': ltp,
                    'Entry': signal['entry'],
                    'Stop': signal['stop_loss'],
                    'Target': signal['target'],
                    'Qty': signal['quantity'],
                    'Status': status,
                    'Distance %': distance,
                    'Current P&L': current_pnl,
                    'Action': action
                })
                
                # Send alerts on status change
                if ALERTS_AVAILABLE and alerts:
                    if status == 'TRIGGERED' and not signal.get('entry_alert_sent'):
                        alerts.send_entry_alert(symbol, ltp)
                        signal['entry_alert_sent'] = True
                    
                    elif status == 'TARGET' and not signal.get('target_alert_sent'):
                        alerts.send_target_alert(symbol, ltp, 
                                                (ltp - signal['entry']) * signal['quantity'])
                        signal['target_alert_sent'] = True
                    
                    elif status == 'STOPPED' and not signal.get('stop_alert_sent'):
                        alerts.send_stoploss_alert(symbol, ltp,
                                                  (signal['entry'] - ltp) * signal['quantity'])
                        signal['stop_alert_sent'] = True
        
        if monitor_data:
            monitor_df = pd.DataFrame(monitor_data)
            
            # Status summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                waiting = len([d for d in monitor_data if d['Status'] == 'WAITING'])
                st.metric("⏳ Waiting", waiting)
            
            with col2:
                triggered = len([d for d in monitor_data if d['Status'] == 'TRIGGERED'])
                st.metric("🟢 Active", triggered)
            
            with col3:
                targets = len([d for d in monitor_data if d['Status'] == 'TARGET'])
                st.metric("🎯 Targets", targets)
            
            with col4:
                stopped = len([d for d in monitor_data if d['Status'] == 'STOPPED'])
                st.metric("🛑 Stopped", stopped)
            
            st.markdown("---")
            
            # Detailed monitor table
            def color_status(val):
                colors = {
                    'WAITING': 'background-color: #FFF3CD; color: #856404',
                    'TRIGGERED': 'background-color: #D1ECF1; color: #0C5460',
                    'TARGET': 'background-color: #D4EDDA; color: #155724',
                    'STOPPED': 'background-color: #F8D7DA; color: #721C24'
                }
                return colors.get(val, '')
            
            def color_pnl(val):
                if val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
                return ''
            
            styled_df = monitor_df.style.applymap(
                color_status, subset=['Status']
            ).applymap(
                color_pnl, subset=['Current P&L']
            ).format({
                'LTP': '₹{:.2f}',
                'Entry': '₹{:.2f}',
                'Stop': '₹{:.2f}',
                'Target': '₹{:.2f}',
                'Distance %': '{:+.2f}%',
                'Current P&L': '₹{:.0f}'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Action items
            st.markdown("---")
            st.subheader("⚡ Action Items")
            
            for data in monitor_data:
                if data['Status'] != 'WAITING':
                    if data['Status'] == 'TRIGGERED':
                        st.success(f"🟢 **{data['Symbol']}**: {data['Action']}")
                    elif data['Status'] == 'TARGET':
                        st.success(f"🎯 **{data['Symbol']}**: {data['Action']}")
                    elif data['Status'] == 'STOPPED':
                        st.error(f"🛑 **{data['Symbol']}**: {data['Action']}")
        
        else:
            st.warning("Unable to fetch live prices. Check your internet connection.")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(30)
            st.rerun()

# ============================================================================
# TAB 3: JOURNAL
# ============================================================================

with tabs[tab_idx]:
    tab_idx += 1
    
    st.header("📓 Trading Journal")
    
    # Quick add trade
    with st.expander("➕ Add Paper Trade", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_symbol = st.selectbox("Symbol", config['symbols']['stocks'][:10])
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01, value=1000.0)
        
        with col2:
            stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01, value=950.0)
            target = st.number_input("Target", min_value=0.0, step=0.01, value=1100.0)
        
        with col3:
            quantity = st.number_input("Quantity", min_value=1, step=1, value=10)
            notes = st.text_input("Notes (optional)")
        
        if st.button("💾 Add Trade", use_container_width=True):
            signal = {
                'symbol': trade_symbol,
                'entry': entry_price,
                'stop_loss': stop_loss,
                'target': target,
                'quantity': quantity,
                'ml_score': 0.0
            }
            journal.add_trade(signal, notes)
            st.success("✅ Trade added to journal")
            time.sleep(1)
            st.rerun()
    
    # Close trade
    open_trades = journal.get_open_trades()
    
    if len(open_trades) > 0:
        with st.expander("❌ Close Trade", expanded=False):
            trade_to_close = st.selectbox(
                "Select Trade to Close",
                open_trades['trade_id'].tolist(),
                format_func=lambda x: f"#{x} - {open_trades[open_trades['trade_id']==x]['symbol'].iloc[0]} @ ₹{open_trades[open_trades['trade_id']==x]['entry_price'].iloc[0]:.2f}"
            )
            
            exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01, value=1000.0, key='exit_price')
            
            if st.button("✅ Close Trade", use_container_width=True):
                journal.close_trade(trade_to_close, exit_price)
                st.success("✅ Trade closed")
                time.sleep(1)
                st.rerun()
    
    st.markdown("---")
    
    # Performance metrics
    metrics = journal.calculate_metrics()
    
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", metrics['total_trades'])
        st.metric("Avg Win", f"₹{metrics['avg_win']:,.0f}")
    
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        st.metric("Avg Loss", f"₹{metrics['avg_loss']:,.0f}")
    
    with col3:
        st.metric("Total P&L", f"₹{metrics['total_pnl']:,.0f}")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    with col4:
        st.metric("Open Positions", len(open_trades))
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
    
    st.markdown("---")
    
    # Trades table
    st.subheader("📋 Trade History")
    
    closed_trades = journal.get_closed_trades()
    
    if len(closed_trades) > 0:
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            filter_symbol = st.multiselect(
                "Filter by Symbol",
                options=closed_trades['symbol'].unique(),
                default=None
            )
        
        with col2:
            show_last = st.slider("Show last N trades", 5, 50, 20)
        
        # Apply filters
        filtered_trades = closed_trades.copy()
        
        if filter_symbol:
            filtered_trades = filtered_trades[filtered_trades['symbol'].isin(filter_symbol)]
        
        # Display table
        display_closed = filtered_trades.tail(show_last)[[
            'trade_id', 'symbol', 'entry_time', 'entry_price',
            'exit_time', 'exit_price', 'quantity', 'pnl', 'pnl_pct'
        ]].copy()
        
        def color_pnl(val):
            if val > 0:
                return 'color: green; font-weight: bold'
            elif val < 0:
                return 'color: red; font-weight: bold'
            return ''
        
        st.dataframe(
            display_closed.style.applymap(
                color_pnl, subset=['pnl', 'pnl_pct']
            ).format({
                'entry_price': '₹{:.2f}',
                'exit_price': '₹{:.2f}',
                'pnl': '₹{:.2f}',
                'pnl_pct': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Equity curve
            equity_curve = journal.get_equity_curve()
            
            if len(equity_curve) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_curve['exit_time'],
                    y=equity_curve['cumulative_pnl'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=2),
                    fill='tozeroy'
                ))
                fig.update_layout(
                    title='Equity Curve',
                    xaxis_title='Date',
                    yaxis_title='Cumulative P&L (₹)',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # P&L Distribution
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=closed_trades['pnl'],
                nbinsx=20,
                marker_color='lightblue',
                marker_line_color='darkblue',
                marker_line_width=1
            ))
            fig2.update_layout(
                title='P&L Distribution',
                xaxis_title='P&L (₹)',
                yaxis_title='Frequency',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Win/Loss pie chart
        wins = len(closed_trades[closed_trades['pnl'] > 0])
        losses = len(closed_trades[closed_trades['pnl'] < 0])
        
        fig3 = go.Figure(data=[go.Pie(
            labels=['Wins', 'Losses'],
            values=[wins, losses],
            marker_colors=['green', 'red'],
            hole=0.3
        )])
        fig3.update_layout(
            title='Win/Loss Ratio',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.info("📝 No closed trades yet. Start trading to see your performance!")

# ============================================================================
# TAB 4: BACKTEST
# ============================================================================

with tabs[tab_idx]:
    tab_idx += 1
    
    st.header("📉 Strategy Backtesting")
    
    st.info("💡 Test your strategy on historical data to see how it would have performed")
    
    col1, col2 = st.columns(2)
    
    with col1:
        backtest_symbol = st.selectbox(
            "Select Stock for Backtest",
            config['symbols']['stocks'],
            key='backtest_symbol'
        )
        
        backtest_period = st.selectbox(
            "Backtest Period",
            ["6mo", "1y", "2y", "5y"],
            index=1
        )
    
    with col2:
        st.markdown("### Parameters")
        st.info(f"ATR Multiplier: {config['risk']['atr_stop_multiplier']}")
        st.info(f"Reward Ratio: {config['risk']['reward_ratio']}")
        st.info(f"Risk per Trade: 0.5%")
    
    if st.button("🔬 Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running backtest... This may take a minute..."):
            
            # Fetch data
            df = datafeed.fetch_historical_data(backtest_symbol, period=backtest_period)
            
            if df is not None and len(df) > 100:
                # Initialize backtester
                backtester = Backtester(
                    initial_capital=config['account']['initial_capital'],
                    risk_per_trade=0.5
                )
                
                # Run backtest
                results = backtester.run_backtest(
                    df,
                    atr_multiplier=config['risk']['atr_stop_multiplier'],
                    reward_ratio=config['risk']['reward_ratio']
                )
                
                if results:
                    st.session_state['backtest_results'] = results
                    st.success("✅ Backtest complete!")
                else:
                    st.error("❌ Backtest failed - insufficient data or no trades generated")
            else:
                st.error("❌ Could not fetch sufficient historical data")
    
    # Display results
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
        st.markdown("---")
        st.subheader("📊 Backtest Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", results['total_trades'])
            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
        
        with col2:
            st.metric("Total Return", f"{results['return_pct']:.1f}%")
            st.metric("Final Capital", f"₹{results['final_capital']:,.0f}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            st.metric("Avg Win", f"₹{results['avg_win']:,.0f}")
        
        with col4:
            st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.0f}")
            st.metric("Avg Loss", f"₹{results['avg_loss']:,.0f}")
        
        # Performance assessment
        if results['return_pct'] > 10 and results['sharpe_ratio'] > 1:
            st.success("🎉 Excellent performance! Strategy shows strong potential")
        elif results['return_pct'] > 0 and results['win_rate'] > 50:
            st.info("👍 Positive performance. Consider optimization")
        else:
            st.warning("⚠️ Strategy needs improvement. Review parameters")
        
        st.markdown("---")
        
        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=results['equity_curve'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='green', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=f'Backtest Equity Curve - {backtest_symbol}',
            xaxis_title='Trade Number',
            yaxis_title='Cumulative P&L (₹)',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade distribution
        if 'trades' in results:
            col1, col2 = st.columns(2)
            
            with col1:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=results['trades']['pnl'],
                    nbinsx=30,
                    marker_color='lightblue',
                    marker_line_color='darkblue',
                    marker_line_width=1
                ))
                fig2.update_layout(
                    title='P&L Distribution',
                    xaxis_title='P&L (₹)',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Win/Loss by ML score
                trades_with_score = results['trades'].copy()
                trades_with_score['win'] = trades_with_score['pnl'] > 0
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=trades_with_score['score'],
                    y=trades_with_score['pnl'],
                    mode='markers',
                    marker=dict(
                        color=trades_with_score['win'],
                        colorscale=[[0, 'red'], [1, 'green']],
                        size=8
                    ),
                    text=trades_with_score.index,
                    hovertemplate='Score: %{x:.2f}<br>P&L: ₹%{y:.0f}<extra></extra>'
                ))
                fig3.update_layout(
                    title='ML Score vs P&L',
                    xaxis_title='ML Score',
                    yaxis_title='P&L (₹)',
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# TAB: OPTIONS (If available)
# ============================================================================

if OPTIONS_AVAILABLE:
    with tabs[tab_idx]:
        tab_idx += 1
        
        st.header("📈 Options Analysis")
        
        st.info("💡 Analyze option chains and calculate Greeks for Indian stocks")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            option_symbol = st.selectbox(
                "Select Stock",
                config['symbols']['stocks'][:10],
                key='option_symbol'
            )
            
            spot_price = st.number_input(
                "Spot Price",
                min_value=0.0,
                value=3500.0,
                step=10.0
            )
        
        with col2:
            days_to_expiry = st.slider(
                "Days to Expiry",
                min_value=1,
                max_value=30,
                value=7
            )
            
            iv = st.slider(
                "Implied Volatility (%)",
                min_value=10,
                max_value=50,
                value=20
            ) / 100
        
        with col3:
            market_outlook = st.selectbox(
                "Market Outlook",
                ['bullish', 'bearish', 'neutral', 'volatile']
            )
            
            iv_percentile = st.slider(
                "IV Percentile",
                min_value=0,
                max_value=100,
                value=50,
                help="Current IV vs historical IV"
            )
        
        if st.button("🔍 Analyze Options Chain", type="primary", use_container_width=True):
            with st.spinner("Calculating option prices and Greeks..."):
                
                # Generate strikes around spot price
                strikes = [spot_price + i * 50 for i in range(-5, 6)]
                
                # Analyze option chain
                chain = options_analyzer.analyze_option_chain(
                    spot_price, strikes, days_to_expiry, iv
                )
                
                st.subheader("📊 Option Chain")
                
                # Format and display
                st.dataframe(
                    chain.style.format({
                        'call_price': '₹{:.2f}',
                        'put_price': '₹{:.2f}',
                        'call_delta': '{:.4f}',
                        'call_theta': '{:.2f}',
                        'put_delta': '{:.4f}',
                        'put_theta': '{:.2f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                st.markdown("---")
                
                # Strategy suggestion
                suggestion = options_analyzer.suggest_strategy(
                    market_outlook, spot_price, iv_percentile
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"💡 **Recommended Strategy:** {suggestion['strategy']}")
                    st.info(f"📊 Market Outlook: {suggestion['outlook'].title()}")
                
                with col2:
                    st.info(f"📈 IV Condition: {suggestion['iv_condition']}")
                    st.info(f"📊 IV Percentile: {suggestion['iv_percentile']:.0f}th")
                
                # Strategy explanation
                st.markdown("---")
                st.subheader("📚 Strategy Explanation")
                
                strategy_info = {
                    'Buy Call': "Bullish strategy. Buy when expecting upward move. Best in low IV.",
                    'Buy Put': "Bearish strategy. Buy when expecting downward move. Best in low IV.",
                    'Bull Put Spread': "Moderately bullish. Sell higher strike put, buy lower strike put. Best in high IV.",
                    'Bear Call Spread': "Moderately bearish. Sell lower strike call, buy higher strike call. Best in high IV.",
                    'Long Straddle': "Expect big move in either direction. Buy ATM call and put. Best in low IV.",
                    'Short Straddle': "Expect no movement. Sell ATM call and put. Best in high IV. High risk!",
                    'Iron Condor': "Neutral strategy. Expect range-bound movement. Best in low IV.",
                    'Calendar Spread': "Profit from time decay. Sell near-term, buy far-term. Best in high IV."
                }
                
                st.info(strategy_info.get(suggestion['strategy'], "Strategy description not available"))

# ============================================================================
# TAB: ALERTS (If available)
# ============================================================================

if ALERTS_AVAILABLE:
    with tabs[tab_idx]:
        tab_idx += 1
        
        st.header("🔔 Alert System")
        
        st.info("💡 Get instant notifications via Telegram when signals trigger")
        
        # Telegram setup
        st.subheader("📱 Telegram Configuration")
        
        with st.expander("📖 How to setup Telegram alerts", expanded=False):
            st.markdown("""
            ### Step-by-step guide:
            
            1. **Create a Telegram Bot:**
               - Open Telegram and search for `@BotFather`
               - Send `/newbot` command
               - Follow instructions to create bot
               - Copy the **Bot Token** (looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)
            
            2. **Get Your Chat ID:**
               - Search for `@userinfobot` on Telegram
               - Start a chat
               - It will send you your **Chat ID** (looks like: `123456789`)
            
            3. **Start your bot:**
               - Find your bot on Telegram
               - Send `/start` to activate it
            
            4. **Enter credentials below**
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            bot_token = st.text_input(
                "🤖 Bot Token",
                type="password",
                help="Get from @BotFather"
            )
        
        with col2:
            chat_id = st.text_input(
                "💬 Chat ID",
                help="Get from @userinfobot"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Settings", use_container_width=True):
                if bot_token and chat_id:
                    alerts.setup_telegram(bot_token, chat_id)
                    st.session_state['alerts_configured'] = True
                    st.success("✅ Telegram alerts configured!")
                else:
                    st.error("❌ Please enter both Bot Token and Chat ID")
        
        with col2:
            if st.button("📤 Send Test Alert", use_container_width=True):
                if 'alerts_configured' in st.session_state:
                    success = alerts.send_telegram(
                        "🔔 <b>Test Alert</b>\n\nYour Trading Assistant is now connected!"
                    )
                    if success:
                        st.success("✅ Test alert sent! Check your Telegram")
                    else:
                        st.error("❌ Failed to send alert. Check your credentials")
                else:
                    st.warning("⚠️ Please save settings first")
        
        st.markdown("---")
        
        # Alert preferences
        st.subheader("⚙️ Alert Preferences")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.checkbox("📊 New signals generated", value=True, key='alert_signals')
            st.checkbox("🟢 Entry price triggered", value=True, key='alert_entry')
        
        with col2:
            st.checkbox("🎯 Target reached", value=True, key='alert_target')
            st.checkbox("🛑 Stop loss hit", value=True, key='alert_stop')
        
        with col3:
            st.checkbox("📈 Daily summary (5 PM)", value=True, key='alert_daily')
            st.checkbox("⚠️ Risk limit warnings", value=True, key='alert_risk')
        
        st.markdown("---")
        
        # Alert history
        st.subheader("📜 Recent Alerts")
        
        if 'alert_history' not in st.session_state:
            st.session_state['alert_history'] = []
        
        if len(st.session_state['alert_history']) > 0:
            for alert in st.session_state['alert_history'][-10:]:
                st.text(alert)
        else:
            st.info("No alerts sent yet")

# ============================================================================
# TAB: BROKER (If available)
# ============================================================================

if BROKER_AVAILABLE:
    with tabs[tab_idx]:
        tab_idx += 1
        
        st.header("🔌 Broker Integration")
        
        st.info("💡 Connect your broker for one-click order execution")
        
        # Broker selection
        broker_name = st.selectbox(
            "Select Broker",
            ["Zerodha Kite", "Upstox", "AliceBlue"]
        )
        
        if broker_name == "Zerodha Kite":
            st.markdown("### 📝 Zerodha Setup")
            
            with st.expander("📖 How to get Zerodha API credentials", expanded=False):
                st.markdown("""
                1. Go to [Kite Connect](https://kite.trade/)
                2. Login with your Zerodha credentials
                3. Create a new app
                4. Note down your **API Key** and **API Secret**
                5. Enter them below
                
                **Note:** You'll need to login once daily via the generated URL
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                api_key = st.text_input("API Key", type="password")
            
            with col2:
                api_secret = st.text_input("API Secret", type="password")
            
            if st.button("🔗 Connect to Zerodha", use_container_width=True):
                if api_key and api_secret:
                    success = broker.connect_zerodha(api_key, api_secret)
                    if success:
                        st.session_state['broker_connected'] = True
                else:
                    st.error("❌ Please enter API credentials")
        
        st.markdown("---")
        
        # If connected, show trading interface
        if broker and broker.connected:
            st.success("✅ Connected to broker")
            
            # Quick order form
            st.subheader("⚡ Quick Order")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                order_symbol = st.selectbox(
                    "Symbol",
                    config['symbols']['stocks'][:10],
                    key='order_symbol'
                )
                
                transaction_type = st.selectbox(
                    "Type",
                    ["BUY", "SELL"]
                )
            
            with col2:
                order_qty = st.number_input(
                    "Quantity",
                    min_value=1,
                    value=1,
                    step=1
                )
                
                order_price = st.number_input(
                    "Price",
                    min_value=0.0,
                    value=3500.0,
                    step=1.0
                )
            
            with col3:
                order_type = st.selectbox(
                    "Order Type",
                    ["LIMIT", "MARKET"]
                )
                
                product = st.selectbox(
                    "Product",
                    ["MIS", "CNC"],
                    help="MIS=Intraday, CNC=Delivery"
                )
            
            if st.button("📤 Place Order", type="primary", use_container_width=True):
                order_id = broker.place_order(
                    symbol=order_symbol.replace('.NS', ''),
                    transaction_type=transaction_type,
                    quantity=order_qty,
                    price=order_price,
                    order_type=order_type,
                    product=product
                )
                
                if order_id:
                    st.balloons()
            
            st.markdown("---")
            
            # Bracket order from signals
            if 'today_signals' in st.session_state and len(st.session_state['today_signals']) > 0:
                st.subheader("🎯 Place from Signals")
                
                signal_options = [
                    f"{s['symbol']} - Entry: ₹{s['entry']:.2f}"
                    for s in st.session_state['today_signals']
                ]
                
                selected_signal_idx = st.selectbox(
                    "Select Signal",
                    range(len(signal_options)),
                    format_func=lambda x: signal_options[x]
                )
                
                selected_signal = st.session_state['today_signals'][selected_signal_idx]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Entry", f"₹{selected_signal['entry']:.2f}")
                
                with col2:
                    st.metric("Stop Loss", f"₹{selected_signal['stop_loss']:.2f}")
                
                with col3:
                    st.metric("Target", f"₹{selected_signal['target']:.2f}")
                
                if st.button("🚀 Place Bracket Order", use_container_width=True):
                    result = broker.place_bracket_order(selected_signal)
                    
                    if result:
                        st.success("✅ Bracket order placed successfully!")
                        st.json(result)
            
            st.markdown("---")
            
            # Show positions
            st.subheader("📊 Open Positions")
            
            positions = broker.get_positions()
            
            if positions and len(positions) > 0:
                positions_df = pd.DataFrame(positions)
                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("No open positions")
            
            st.markdown("---")
            
            # Show orders
            st.subheader("📋 Today's Orders")
            
            orders = broker.get_orders()
            
            if orders and len(orders) > 0:
                orders_df = pd.DataFrame(orders)
                st.dataframe(orders_df, use_container_width=True)
            else:
                st.info("No orders placed today")
        
        else:
            st.warning("⚠️ Not connected to broker. Enter credentials above.")
            
            st.markdown("### 🎯 Why Connect Broker?")
            st.markdown("""
            - **One-click execution** from signals
            - **Bracket orders** with automatic SL and target
            - **Real-time position tracking**
            - **Order management** from single dashboard
            - **No need** to open broker app separately
            """)

# ============================================================================
# TAB: SETTINGS
# ============================================================================

with tabs[tab_idx]:
    tab_idx += 1
    
    st.header("⚙️ Settings & Configuration")
    
    # Account settings
    st.subheader("💰 Account Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_capital = st.number_input(
            "Account Size (₹)",
            min_value=10000,
            max_value=10000000,
            value=config['account']['initial_capital'],
            step=10000
        )
        
        new_risk_pct = st.slider(
            "Daily Risk %",
            min_value=0.5,
            max_value=5.0,
            value=config['account']['daily_risk_percent'],
            step=0.1,
            help="Maximum percentage of account to risk per day"
        )
    
    with col2:
        new_max_trades = st.slider(
            "Max Trades per Day",
            min_value=1,
            max_value=10,
            value=config['account']['max_trades_per_day'],
            help="Maximum number of trades allowed per day"
        )
        
        new_min_score = st.slider(
            "Min ML Score",
            min_value=0.5,
            max_value=0.9,
            value=config['ml']['min_score'],
            step=0.05,
            help="Minimum ML score to generate signal"
        )
    
    st.markdown("---")
    
    # Risk settings
    st.subheader("⚖️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_atr_mult = st.slider(
            "ATR Stop Multiplier",
            min_value=1.0,
            max_value=4.0,
            value=config['risk']['atr_stop_multiplier'],
            step=0.5,
            help="Multiplier for ATR-based stop loss"
        )
    
    with col2:
        new_reward = st.slider(
            "Reward Ratio (R:R)",
            min_value=1.0,
            max_value=5.0,
            value=config['risk']['reward_ratio'],
            step=0.5,
            help="Target profit as multiple of risk"
        )
    
    st.markdown("---")
    
    # Save button
    if st.button("💾 Save All Settings", type="primary", use_container_width=True):
        # Update config
        config['account']['initial_capital'] = new_capital
        config['account']['daily_risk_percent'] = new_risk_pct
        config['account']['max_trades_per_day'] = new_max_trades
        config['risk']['atr_stop_multiplier'] = new_atr_mult
        config['risk']['reward_ratio'] = new_reward
        config['ml']['min_score'] = new_min_score
        
        # Save to file
        try:
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f)
            
            st.success("✅ Settings saved! Restart the app to apply changes.")
            st.balloons()
        except Exception as e:
            st.error(f"❌ Failed to save: {str(e)}")
    
    st.markdown("---")
    
    # Display current config
    st.subheader("📄 Current Configuration")
    
    with st.expander("View config.yaml"):
        st.code(yaml.dump(config, default_flow_style=False), language='yaml')
    
    st.markdown("---")
    
    # Advanced settings
    st.subheader("🔧 Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Enable debug mode", value=False)
        st.checkbox("Log all trades to file", value=True)
    
    with col2:
        st.checkbox("Enable experimental features", value=False)
        st.checkbox("Auto-save journal", value=True)
    
    # Clear cache
    if st.button("🗑️ Clear Cache & Reset", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared! Refresh to reload.")

# ============================================================================
# TAB: HELP
# ============================================================================

with tabs[tab_idx]:
    tab_idx += 1
    
    st.header("❓ Help & Documentation")
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    with st.expander("📖 How to use Trading Assistant", expanded=True):
        st.markdown("""
        ### Step-by-Step Workflow:
        
        #### 1️⃣ **Generate Signals** (TODAY tab)
        - Check market regime
        - Click "Generate Signals"
        - Wait 10-15 seconds for AI analysis
        - Review top 5 stock signals
        
        #### 2️⃣ **Monitor Live Prices** (MONITOR tab)
        - Click "Refresh" to update prices
        - Watch for entry triggers
        - Check status: WAITING → TRIGGERED → TARGET/STOPPED
        - Follow action recommendations
        
        #### 3️⃣ **Execute Trades** (Manual or via Broker)
        - **Option A:** Open Zerodha/Upstox app manually
        - **Option B:** Use BROKER tab (if configured)
        - Place orders based on signals
        - Set stop loss and target alerts
        
        #### 4️⃣ **Record in Journal** (JOURNAL tab)
        - Add trade details
        - Close trades when exited
        - Track performance
        - Review equity curve
        
        #### 5️⃣ **Optimize Strategy** (BACKTEST tab)
        - Test on historical data
        - Check win rate and returns
        - Adjust parameters in SETTINGS
        """)
    
    # Features guide
    st.markdown("---")
    st.subheader("🌟 Features Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📊 Signal Generation"):
            st.markdown("""
            **What it does:**
            - Analyzes technical indicators
            - Ranks stocks using AI (ML score 0-1)
            - Filters top opportunities
            - Calculates entry, stop, target
            
            **How to use:**
            - Look for ML score > 0.6
            - Check RSI (50-70 is good)
            - Verify market regime matches
            - Review risk/reward ratio
            """)
        
        with st.expander("🔴 Live Monitor"):
            st.markdown("""
            **What it does:**
            - Fetches real-time prices
            - Compares with entry/stop/target
            - Shows distance to entry
            - Provides action guidance
            
            **Status meanings:**
            - **WAITING**: Price below entry
            - **TRIGGERED**: Trade active
            - **TARGET**: Profit target hit
            - **STOPPED**: Stop loss hit
            """)
        
        with st.expander("📓 Trading Journal"):
            st.markdown("""
            **What it does:**
            - Records all trades
            - Calculates P&L
            - Tracks performance metrics
            - Displays equity curve
            
            **Metrics explained:**
            - **Win Rate**: % of profitable trades
            - **Sharpe Ratio**: Risk-adjusted returns
            - **Max Drawdown**: Largest peak-to-trough loss
            """)
    
    with col2:
        with st.expander("📈 Options Analysis"):
            st.markdown("""
            **What it does:**
            - Calculates option prices
            - Computes Greeks (Delta, Theta, etc.)
            - Suggests strategies
            - Analyzes IV conditions
            
            **Strategy types:**
            - **Directional**: Buy Call/Put
            - **Spreads**: Bull/Bear spreads
            - **Neutral**: Iron Condor, Straddle
            """)
        
        with st.expander("🔔 Alert System"):
            st.markdown("""
            **What it does:**
            - Sends Telegram notifications
            - Alerts on entry triggers
            - Notifies target/stop hits
            - Daily summary reports
            
            **Setup:**
            1. Create Telegram bot (@BotFather)
            2. Get Chat ID (@userinfobot)
            3. Enter in ALERTS tab
            4. Test connection
            """)
        
        with st.expander("🔌 Broker Integration"):
            st.markdown("""
            **What it does:**
            - One-click order placement
            - Bracket orders (entry+SL+target)
            - Position tracking
            - Order management
            
            **Supported:**
            - Zerodha Kite
            - Upstox
            - AliceBlue
            """)
    
    # FAQ
    st.markdown("---")
    st.subheader("🤔 Frequently Asked Questions")
    
    with st.expander("Q: How accurate are the signals?"):
        st.markdown("""
        **A:** Signals are based on AI and technical analysis. Past backtests show 55-65% win rate, 
        but **no strategy is 100% accurate**. Always:
        - Use proper risk management
        - Never risk more than 1% per trade
        - Combine with your own analysis
        - Start with paper trading
        """)
    
    with st.expander("Q: Can I use this for intraday trading?"):
        st.markdown("""
        **A:** Yes! The platform supports both:
        - **Intraday (MIS)**: Use smaller timeframes, tighter stops
        - **Swing (CNC)**: Use daily signals, hold 2-5 days
        
        Adjust ATR multiplier and reward ratio accordingly in SETTINGS.
        """)
    
    with st.expander("Q: How much capital do I need?"):
        st.markdown("""
        **A:** Recommended minimum:
        - **Paper trading**: Any amount (practice mode)
        - **Live trading**: ₹50,000 - ₹1,00,000
        
        With ₹1 lakh and 1% daily risk:
        - Risk per trade: ₹500
        - Typical position: 5-15 shares
        - 3 trades max per day
        """)
    
    with st.expander("Q: Is this auto-trading?"):
        st.markdown("""
        **A:** No! This is a **trading assistant**, not auto-trading:
        - Generates signals (AI-powered)
        - You review signals
        - You decide to trade or skip
        - You execute manually (or via broker integration)
        - You manage the trade
        
        **You are always in control!**
        """)
    
    with st.expander("Q: What if I lose money?"):
        st.markdown("""
        **A:** Trading involves risk. To minimize losses:
        
        1. **Always use stop losses** (mandatory!)
        2. **Never risk more than 1%** of account per trade
        3. **Max 3 trades per day** (avoid overtrading)
        4. **Start with paper trading** (practice first)
        5. **Review your journal** (learn from mistakes)
        
        **Remember:** This is an educational tool, not financial advice.
        """)
    
    # Technical info
    st.markdown("---")
    st.subheader("🔧 Technical Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Model:**")
        st.code(f"""
Type: {ML_TYPE}
Features: 12 technical indicators
Training: 70/30 split
Lookback: 60 days
Min Score: {config['ml']['min_score']}
        """)
    
    with col2:
        st.markdown("**Risk Management:**")
        st.code(f"""
Stop Loss: {config['risk']['atr_stop_multiplier']}× ATR
Reward Ratio: {config['risk']['reward_ratio']}:1
Daily Risk: {config['account']['daily_risk_percent']}%
Max Trades: {config['account']['max_trades_per_day']}/day
        """)
    
    # Contact & support
    st.markdown("---")
    st.subheader("💬 Support & Feedback")
    
    st.info("""
    **Need help?**
    - 📧 Report issues on GitHub
    - 💬 Join community discussions
    - 📚 Read code documentation
    - ⭐ Star the repository
    """)
    
    # Disclaimer
    st.markdown("---")
    st.error("""
    ⚠️ **IMPORTANT DISCLAIMER**
    
    This tool is for **EDUCATIONAL PURPOSES ONLY**.
    
    - NOT financial advice
    - NO guarantees of profit
    - Trading involves substantial risk
    - You may lose your entire investment
    - Always do your own research
    - Start with paper trading
    - Never trade with money you can't afford to lose
    
    **Use at your own risk.**
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("📈 Trading Assistant Pro v2.0")

with footer_col2:
    st.caption("🤖 Powered by AI & Machine Learning")

with footer_col3:
    st.caption("⚠️ Educational Tool - Not Financial Advice")
