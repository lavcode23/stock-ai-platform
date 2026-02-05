"""
Trading Assistant - Main Streamlit Application
Professional trading app for Indian stocks
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import time

# Import custom modules
from datafeed import DataFeed
from features import FeatureEngine
from ml_ranker import MLRanker
from strategy import TradingStrategy
from risk import RiskManager
from journal import TradingJournal
from backtest import Backtester

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

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all trading components"""
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

# Sidebar
with st.sidebar:
    st.title("📈 Trading Assistant")
    st.markdown("---")
    
    # Account info
    st.subheader("Account")
    st.metric("Capital", f"₹{config['account']['initial_capital']:,.0f}")
    st.metric("Daily Risk", f"{config['account']['daily_risk_percent']}%")
    st.metric("Max Trades/Day", config['account']['max_trades_per_day'])
    
    st.markdown("---")
    
    # Market status
    index_data = datafeed.get_index_data()
    if index_data:
        st.subheader("NIFTY")
        st.metric(
            "Value",
            f"{index_data['value']:,.2f}",
            f"{index_data['change']:+.2f} ({index_data['change_pct']:+.2f}%)"
        )
        
        regime = index_data['regime']
        color = {'BULL': '🟢', 'BEAR': '🔴', 'SIDEWAYS': '🟡'}.get(regime, '⚪')
        st.info(f"Market Regime: {color} {regime}")
    
    st.markdown("---")
    st.caption("⚠️ For educational purposes only")
    st.caption("No guarantees. Trade at your own risk.")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 TODAY", "🔴 LIVE MONITOR", "🏢 SECTORS", 
    "📓 JOURNAL", "📉 BACKTEST", "⚙️ SETTINGS", "❓ EXPLAIN"
])

# TAB 1: TODAY
with tab1:
    st.header("Today's Trading Plan")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if index_data:
            regime = index_data['regime']
            if regime == 'BULL':
                st.success(f"🟢 Market Regime: **{regime}** - Long bias recommended")
            elif regime == 'BEAR':
                st.error(f"🔴 Market Regime: **{regime}** - Caution advised")
            else:
                st.warning(f"🟡 Market Regime: **{regime}** - Wait for clarity")
    
    with col2:
        limits = risk_manager.check_daily_limits(journal.trades.to_dict('records'))
        st.metric("Trades Today", f"{limits['trades_taken']}/{limits['max_trades']}")
    
    with col3:
        st.metric("Risk Used", f"₹{limits['risk_used']:,.0f}")
    
    st.markdown("---")
    
    # Generate signals button
    if st.button("🎯 Generate Today's Signals", type="primary", use_container_width=True):
        with st.spinner("Analyzing stocks..."):
            
            # Train ML model on NIFTY
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Training ML model on NIFTY...")
            nifty_df = datafeed.fetch_historical_data(
                config['symbols']['index'], 
                period=config['data']['history_period']
            )
            
            if nifty_df is not None:
                nifty_features = feature_engine.create_ml_features(nifty_df)
                if nifty_features is not None:
                    ml_metrics = ml_ranker.train(nifty_features)
                    if ml_metrics:
                        st.success(f"✅ Model trained: {ml_metrics['samples']} samples, Test score: {ml_metrics['test_score']}")
            
            progress_bar.progress(20)
            
            # Fetch and analyze stocks
            status_text.text("Fetching stock data...")
            stocks_data = {}
            stocks_features = {}
            
            for idx, symbol in enumerate(config['symbols']['stocks']):
                df = datafeed.fetch_historical_data(symbol, period=config['data']['history_period'])
                if df is not None:
                    features_df = feature_engine.create_ml_features(df)
                    if features_df is not None:
                        latest_features = feature_engine.get_latest_features(features_df)
                        if latest_features:
                            stocks_data[symbol] = df
                            stocks_features[symbol] = latest_features
                
                progress_bar.progress(20 + int((idx / len(config['symbols']['stocks'])) * 60))
            
            status_text.text("Ranking stocks...")
            
            # Rank stocks
            ranked = ml_ranker.rank_stocks(stocks_features)
            
            progress_bar.progress(85)
            status_text.text("Generating signals...")
            
            # Generate signals for top stocks
            signals = []
            live_prices = datafeed.fetch_batch_live_prices([s[0] for s in ranked[:10]])
            
            for symbol, score in ranked[:10]:
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
            status_text.text("Done!")
            
            # Store in session state
            st.session_state['today_signals'] = signals
            
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
    
    # Display signals
    if 'today_signals' in st.session_state and st.session_state['today_signals']:
        st.subheader(f"📋 Signals Generated ({len(st.session_state['today_signals'])})")
        
        signals_df = pd.DataFrame(st.session_state['today_signals'])
        
        # Format for display
        display_df = signals_df[[
            'symbol', 'ml_score', 'entry', 'stop_loss', 'target',
            'quantity', 'risk_amount', 'reward_amount', 'rsi', 'adx'
        ]].copy()
        
        display_df.columns = [
            'Symbol', 'ML Score', 'Entry', 'Stop', 'Target',
            'Qty', 'Risk ₹', 'Reward ₹', 'RSI', 'ADX'
        ]
        
        # Style the dataframe
        st.dataframe(
            display_df.style.format({
                'ML Score': '{:.3f}',
                'Entry': '₹{:.2f}',
                'Stop': '₹{:.2f}',
                'Target': '₹{:.2f}',
                'Risk ₹': '₹{:.0f}',
                'Reward ₹': '₹{:.0f}',
                'RSI': '{:.1f}',
                'ADX': '{:.1f}'
            }).background_gradient(subset=['ML Score'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Pre-trade checklist
        st.markdown("---")
        st.subheader("✅ Pre-Trade Checklist")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Market regime favorable")
            st.checkbox("Risk limits not exceeded")
            st.checkbox("Signals reviewed and understood")
        
        with col2:
            st.checkbox("Stop loss levels marked")
            st.checkbox("Position sizes calculated")
            st.checkbox("Ready to execute manually")

# TAB 2: LIVE MONITOR
with tab2:
    st.header("🔴 Live Monitor")
    
    if 'today_signals' not in st.session_state or not st.session_state['today_signals']:
        st.info("👈 Generate signals in the TODAY tab first")
    else:
        # Auto-refresh
        placeholder = st.empty()
        
        # Refresh button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        with col2:
            st.caption("Auto-refreshes every 10 seconds")
        
        # Monitor signals
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
                action = strategy.get_action_message(signal, ltp)
                
                monitor_data.append({
                    'Symbol': symbol,
                    'LTP': ltp,
                    'Entry': signal['entry'],
                    'Stop': signal['stop_loss'],
                    'Target': signal['target'],
                    'Qty': signal['quantity'],
                    'Status': status,
                    'Distance %': distance,
                    'Action': action
                })
        
        if monitor_data:
            monitor_df = pd.DataFrame(monitor_data)
            
            # Color code status
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
                    'Stop': '₹{:.2f}',
                    'Target': '₹{:.2f}',
                    'Distance %': '{:+.2f}%'
                }),
                use_container_width=True,
                height=500
            )
            
            # Summary
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                waiting = len([s for s in monitor_data if s['Status'] == 'WAITING'])
                st.metric("Waiting", waiting)
            
            with col2:
                triggered = len([s for s in monitor_data if s['Status'] == 'TRIGGERED'])
                st.metric("Active", triggered)
            
            with col3:
                targets = len([s for s in monitor_data if s['Status'] == 'TARGET'])
                st.metric("Targets Hit", targets)
            
            with col4:
                stopped = len([s for s in monitor_data if s['Status'] == 'STOPPED'])
                st.metric("Stopped Out", stopped)
        
        # Auto-refresh after 10 seconds
        time.sleep(10)
        st.rerun()

# TAB 3: SECTORS
with tab3:
    st.header("🏢 Sector Analysis")
    
    if st.button("📊 Analyze Sectors"):
        with st.spinner("Analyzing sectors..."):
            
            sector_scores = {}
            
            for sector_name, sector_symbols in config['sectors'].items():
                scores = []
                
                for symbol in sector_symbols:
                    if symbol in config['symbols']['stocks']:
                        df = datafeed.fetch_historical_data(symbol, period="3mo")
                        if df is not None:
                            features_df = feature_engine.create_ml_features(df)
                            if features_df is not None:
                                latest_features = feature_engine.get_latest_features(features_df)
                                if latest_features:
                                    score = ml_ranker.predict_proba(latest_features)
                                    scores.append(score)
                
                if scores:
                    sector_scores[sector_name] = {
                        'avg_score': np.mean(scores),
                        'max_score': np.max(scores),
                        'count': len(scores),
                        'strong_stocks': sum([1 for s in scores if s > 0.6])
                    }
            
            st.session_state['sector_scores'] = sector_scores
    
    if 'sector_scores' in st.session_state:
        sector_data = st.session_state['sector_scores']
        
        # Create dataframe
        sector_df = pd.DataFrame([
            {
                'Sector': name,
                'Avg Score': data['avg_score'],
                'Max Score': data['max_score'],
                'Stocks': data['count'],
                'Strong (>0.6)': data['strong_stocks']
            }
            for name, data in sector_data.items()
        ]).sort_values('Avg Score', ascending=False)
        
        # Display table
        st.dataframe(
            sector_df.style.format({
                'Avg Score': '{:.3f}',
                'Max Score': '{:.3f}'
            }).background_gradient(subset=['Avg Score'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Chart
        fig = px.bar(
            sector_df,
            x='Sector',
            y='Avg Score',
            color='Avg Score',
            color_continuous_scale='RdYlGn',
            title='Sector Strength'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: JOURNAL
with tab4:
    st.header("📓 Trading Journal")
    
    # Add trade manually
    with st.expander("➕ Add Paper Trade"):
        col1, col2 = st.columns(2)
        
        with col1:
            trade_symbol = st.selectbox("Symbol", config['symbols']['stocks'])
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01)
            quantity = st.number_input("Quantity", min_value=1, step=1)
        
        with col2:
            stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01)
            target = st.number_input("Target", min_value=0.0, step=0.01)
            notes = st.text_input("Notes (optional)")
        
        if st.button("Add Trade"):
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
            st.rerun()
    
    # Close trade
    open_trades = journal.get_open_trades()
    
    if len(open_trades) > 0:
        with st.expander("❌ Close Trade"):
            trade_to_close = st.selectbox(
                "Select Trade",
                open_trades['trade_id'].tolist(),
                format_func=lambda x: f"#{x} - {open_trades[open_trades['trade_id']==x]['symbol'].iloc[0]}"
            )
            
            exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01, key='exit_price')
            
            if st.button("Close Trade"):
                journal.close_trade(trade_to_close, exit_price)
                st.success("✅ Trade closed")
                st.rerun()
    
    st.markdown("---")
    
    # Performance metrics
    metrics = journal.calculate_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", metrics['total_trades'])
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    with col3:
        st.metric("Total P&L", f"₹{metrics['total_pnl']:,.0f}")
    with col4:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Win", f"₹{metrics['avg_win']:,.0f}")
    with col2:
        st.metric("Avg Loss", f"₹{metrics['avg_loss']:,.0f}")
    with col3:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
    
    st.markdown("---")
    
    # Trades table
    st.subheader("Closed Trades")
    
    closed_trades = journal.get_closed_trades()
    
    if len(closed_trades) > 0:
        display_closed = closed_trades[[
            'trade_id', 'symbol', 'entry_time', 'entry_price',
            'exit_price', 'quantity', 'pnl', 'pnl_pct'
        ]].copy()
        
        st.dataframe(
            display_closed.style.format({
                'entry_price': '₹{:.2f}',
                'exit_price': '₹{:.2f}',
                'pnl': '₹{:.2f}',
                'pnl_pct': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Equity curve
        equity_curve = journal.get_equity_curve()
        
        if len(equity_curve) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve['exit_time'],
                y=equity_curve['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Cumulative P&L (₹)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # PnL histogram
            fig2 = px.histogram(
                closed_trades,
                x='pnl',
                nbins=20,
                title='P&L Distribution'
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No closed trades yet")

# TAB 5: BACKTEST
with tab5:
    st.header("📉 Historical Backtest")
    
    backtest_symbol = st.selectbox("Select Stock", config['symbols']['stocks'], key='backtest_symbol')
    
    if st.button("🔬 Run Backtest"):
        with st.spinner("Running backtest..."):
            
            # Fetch data
            df = datafeed.fetch_historical_data(backtest_symbol, period="1y")
            
            if df is not None:
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
                else:
                    st.error("Backtest failed - insufficient data")
            else:
                st.error("Could not fetch data")
    
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", results['total_trades'])
        with col2:
            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
        with col3:
            st.metric("Total Return", f"{results['return_pct']:.1f}%")
        with col4:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Capital", f"₹{results['final_capital']:,.0f}")
        with col2:
            st.metric("Avg Win", f"₹{results['avg_win']:,.0f}")
        with col3:
            st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.0f}")
        
        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=results['equity_curve'],
            mode='lines',
            name='Equity',
            line=dict(color='green', width=2)
        ))
        fig.update_layout(
            title='Backtest Equity Curve',
            xaxis_title='Trade Number',
            yaxis_title='Cumulative P&L (₹)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Trades distribution
        if 'trades' in results:
            fig2 = px.histogram(
                results['trades'],
                x='pnl',
                nbins=20,
                title='P&L Distribution'
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

# TAB 6: SETTINGS
with tab6:
    st.header("⚙️ Settings")
    
    st.subheader("Account Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_capital = st.number_input(
            "Account Size (₹)",
            min_value=10000,
            value=config['account']['initial_capital'],
            step=10000
        )
        
        new_risk_pct = st.slider(
            "Daily Risk %",
            min_value=0.5,
            max_value=3.0,
            value=config['account']['daily_risk_percent'],
            step=0.1
        )
    
    with col2:
        new_max_trades = st.slider(
            "Max Trades per Day",
            min_value=1,
            max_value=10,
            value=config['account']['max_trades_per_day']
        )
        
        new_atr_mult = st.slider(
            "ATR Stop Multiplier",
            min_value=1.0,
            max_value=4.0,
            value=config['risk']['atr_stop_multiplier'],
            step=0.5
        )
    
    new_reward = st.slider(
        "Reward Ratio (R:R)",
        min_value=1.0,
        max_value=5.0,
        value=config['risk']['reward_ratio'],
        step=0.5
    )
    
    if st.button("💾 Save Settings"):
        config['account']['initial_capital'] = new_capital
        config['account']['daily_risk_percent'] = new_risk_pct
        config['account']['max_trades_per_day'] = new_max_trades
        config['risk']['atr_stop_multiplier'] = new_atr_mult
        config['risk']['reward_ratio'] = new_reward
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        st.success("✅ Settings saved! Restart the app to apply changes.")

# TAB 7: EXPLAIN
with tab7:
    st.header("❓ How It Works")
    
    st.markdown("""
    ## Trading Assistant Workflow
    
    ### 1️⃣ Data Collection
    - Fetches live Indian stock data using `yfinance`
    - Analyzes NIFTY index for market regime
    - Downloads 3-month historical data for stocks
    
    ### 2️⃣ Feature Engineering
    - Calculates 12+ technical indicators:
        - RSI, MACD, Bollinger Bands
        - ATR, ADX, Stochastic
        - Moving averages and momentum
    - Creates ML-ready features
    
    ### 3️⃣ Machine Learning
    - Trains RandomForest classifier on NIFTY
    - Predicts probability of upward moves
    - Ranks stocks by ML score (0-1)
    
    ### 4️⃣ Signal Generation
    - Filters stocks with score > 0.6
    - Calculates entry, stop (ATR-based), target (R:R ratio)
    - Applies position sizing based on risk
    
    ### 5️⃣ Risk Management
    - Max 1% account risk per day
    - Max 3 trades per day
    - ATR-based stops
    - Risk-based position sizing
    
    ### 6️⃣ Execution (Manual)
    - User reviews signals in TODAY tab
    - Monitors live prices in LIVE MONITOR
    - Executes trades manually in broker app
    - Records trades in JOURNAL
    
    ### 7️⃣ Performance Tracking
    - Tracks all trades in CSV
    - Calculates win rate, Sharpe, drawdown
    - Shows equity curve
    - Analyzes P&L distribution
    
    ---
    
    ## Key Features
    
    ✅ **No Auto-Trading** - You control execution  
    ✅ **Risk Managed** - Built-in position sizing  
    ✅ **ML Powered** - Data-driven stock selection  
    ✅ **Live Monitoring** - Real-time price tracking  
    ✅ **Paper Trading** - Practice without risk  
    ✅ **Performance Analytics** - Track your progress  
    
    ---
    
    ## Disclaimer
    
    ⚠️ **For Educational Purposes Only**
    
    - This tool provides trading assistance, not advice
    - Past performance does not guarantee future results
    - Trading involves substantial risk of loss
    - Always do your own research
    - Start with paper trading
    - Never risk more than you can afford to lose
    
    ---
    
    ## Tips for Success
    
    1. **Start Small** - Use paper trading first
    2. **Follow Rules** - Respect risk limits
    3. **Be Disciplined** - Stick to stops and targets
    4. **Review Trades** - Learn from the journal
    5. **Stay Patient** - Wait for quality setups
    6. **Manage Risk** - Never over-leverage
    
    ---
    
    ## Technical Stack
    
    - **Frontend**: Streamlit
    - **Data**: yfinance
    - **ML**: scikit-learn (RandomForest)
    - **Charts**: Plotly
    - **Storage**: CSV files
    - **Language**: Python 3.8+
    
    ---
    
    **Happy Trading! 📈**
    """)

# Footer
st.markdown("---")
st.caption("Trading Assistant v1.0 | Educational Tool | Not Financial Advice")
