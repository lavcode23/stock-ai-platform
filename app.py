"""
Trading Assistant - Fast-Loading Pro Cockpit (FULL app.py replacement)
- Fixes tab bugs
- Adds auto-refresh monitor (optional)
- Adds risk gate + sector summary + "why this trade" + candlestick
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
from datetime import datetime
import time

# Optional auto-refresh (recommended)
AUTOREFRESH_OK = True
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    AUTOREFRESH_OK = False

# Import custom modules
from datafeed import DataFeed
from features import FeatureEngine
from ml_ranker import MLRanker
from strategy import TradingStrategy
from risk import RiskManager
from journal import TradingJournal

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Config
# -----------------------------
@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# -----------------------------
# Initialize components (cached)
# -----------------------------
@st.cache_resource
def initialize_components():
    datafeed = DataFeed(config["symbols"]["stocks"], config["symbols"]["index"])
    feature_engine = FeatureEngine()
    ml_ranker = MLRanker()
    strategy = TradingStrategy(
        atr_multiplier=config["risk"]["atr_stop_multiplier"],
        reward_ratio=config["risk"]["reward_ratio"],
        min_score=config["ml"]["min_score"],
    )
    risk_manager = RiskManager(
        account_size=config["account"]["initial_capital"],
        daily_risk_pct=config["account"]["daily_risk_percent"],
        max_trades_per_day=config["account"]["max_trades_per_day"],
    )
    journal = TradingJournal()
    return datafeed, feature_engine, ml_ranker, strategy, risk_manager, journal

datafeed, feature_engine, ml_ranker, strategy, risk_manager, journal = initialize_components()

# -----------------------------
# Cache expensive operations
# -----------------------------
@st.cache_data(ttl=3600)
def get_index_data_cached():
    return datafeed.get_index_data()

@st.cache_data(ttl=1800)
def train_model_cached():
    nifty_df = datafeed.fetch_historical_data(config["symbols"]["index"], period="2mo")
    if nifty_df is not None and not nifty_df.empty:
        nifty_features = feature_engine.create_ml_features(nifty_df)
        if nifty_features is not None and not nifty_features.empty:
            return ml_ranker.train(nifty_features)
    return None

@st.cache_data(ttl=900)
def fetch_candles_cached(symbol: str, period="3mo"):
    df = datafeed.fetch_historical_data(symbol, period=period)
    if df is None:
        return pd.DataFrame()
    return df.dropna()

# -----------------------------
# Helpers
# -----------------------------
def format_rupee(x):
    try:
        return f"₹{float(x):,.0f}"
    except Exception:
        return str(x)

def risk_gate_reduce(signals: list, max_daily_risk: float):
    """Return (signals_after_gate, status_text)."""
    if not signals:
        return signals, "NO_SIGNALS"

    # Sort by score descending, keep best until within risk
    s = sorted(signals, key=lambda d: float(d.get("ml_score", 0)), reverse=True)
    kept = []
    total = 0.0
    for item in s:
        r = float(item.get("risk_amount", 0) or 0)
        if total + r <= max_daily_risk:
            kept.append(item)
            total += r

    if total <= max_daily_risk and len(kept) == len(s):
        return kept, "PASS"
    if len(kept) == 0:
        return kept, "BLOCK"
    return kept, "REDUCED"

def make_candlestick(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        st.info("No candle data available for this symbol.")
        return
    df = df.tail(120).copy()
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
            )
        ]
    )
    fig.update_layout(height=420, title=title, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Sidebar (lightweight)
# -----------------------------
with st.sidebar:
    st.title("📈 Trading Assistant")
    st.markdown("---")

    st.subheader("Account")
    st.metric("Capital", f"₹{config['account']['initial_capital']:,.0f}")
    st.metric("Daily Risk", f"{config['account']['daily_risk_percent']}%")
    st.metric("Max Trades", f"{config['account']['max_trades_per_day']}")

    st.markdown("---")

    with st.spinner("Loading market data..."):
        index_data = get_index_data_cached()

    if index_data:
        st.subheader("NIFTY")
        st.metric(
            "Value",
            f"{index_data['value']:,.2f}",
            f"{index_data['change']:+.2f} ({index_data['change_pct']:+.2f}%)",
        )

        regime = index_data.get("regime", "UNKNOWN")
        color = {"BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🟡"}.get(regime, "⚪")
        st.info(f"Regime: {color} {regime}")

    st.markdown("---")
    st.caption("⚠️ Educational purposes only")

# -----------------------------
# Tabs (FIXED)
# -----------------------------
tab_today, tab_monitor, tab_journal, tab_settings, tab_help = st.tabs(
    ["📊 TODAY", "🔴 MONITOR", "📓 JOURNAL", "⚙️ SETTINGS", "❓ HELP"]
)

# =========================================================
# TAB 1: TODAY
# =========================================================
with tab_today:
    st.header("Today's Trading Plan")

    col1, col2, col3 = st.columns([2, 1, 1])

    # Market regime display
    with col1:
        if index_data:
            regime = index_data.get("regime", "UNKNOWN")
            if regime == "BULL":
                st.success(f"🟢 Market: **{regime}**")
            elif regime == "BEAR":
                st.error(f"🔴 Market: **{regime}**")
            else:
                st.warning(f"🟡 Market: **{regime}**")
        else:
            st.info("Market regime unavailable.")

    # Risk/trade limits
    limits = risk_manager.check_daily_limits(journal.trades.to_dict("records") if hasattr(journal, "trades") else [])
    with col2:
        st.metric("Trades", f"{limits.get('trades_taken',0)}/{limits.get('max_trades',0)}")
    with col3:
        st.metric("Risk Used", f"₹{limits.get('risk_used',0):,.0f}")

    st.markdown("---")
    # -------- Risk Gate Visual --------
max_daily_risk = config["account"]["initial_capital"] * config["account"]["daily_risk_percent"] / 100
used_risk = limits.get("risk_used", 0)

risk_pct = min(used_risk / max_daily_risk, 1.0) if max_daily_risk > 0 else 0

st.markdown("### 🛡️ Risk Gate")

st.progress(risk_pct)

st.caption(
    f"Used: ₹{used_risk:,.0f} / Allowed: ₹{max_daily_risk:,.0f} "
    f"({risk_pct*100:.0f}%)"
)

if used_risk >= max_daily_risk:
    st.error("⛔ Daily risk exhausted. No more trades today.")


    # Action controls
    gen_col, opt_col = st.columns([1.3, 1])
    with gen_col:
        fast_mode = st.checkbox("Fast Mode (top 10 stocks, 2 months)", value=True)
    with opt_col:
        show_why_panel = st.checkbox("Show 'Why this trade?'", value=True)

    # Generate signals
    if st.button("🎯 Generate Signals", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status = st.empty()

        # Step 1: Train model (cached)
        status.info("⚙️ Training ML model...")
        progress_bar.progress(15)
        ml_metrics = train_model_cached()
        if ml_metrics:
            status.success(f"✅ Model ready (Test score: {ml_metrics.get('test_score','NA')})")
        else:
            status.warning("Model training returned empty metrics (continuing).")

        # Step 2: Choose stocks universe
        status.info("📊 Fetching & building features...")
        progress_bar.progress(30)

        stock_list = config["symbols"]["stocks"]
        if fast_mode:
            stock_list = stock_list[:10]

        period = "2mo" if fast_mode else "6mo"

        stocks_features = {}
        for idx, symbol in enumerate(stock_list):
            df = datafeed.fetch_historical_data(symbol, period=period)
            if df is not None and not df.empty:
                features_df = feature_engine.create_ml_features(df)
                if features_df is not None and not features_df.empty:
                    latest_features = feature_engine.get_latest_features(features_df)
                    if latest_features:
                        stocks_features[symbol] = latest_features

            progress_bar.progress(30 + int((idx / max(1, len(stock_list))) * 40))

        # Step 3: Rank
        status.info("🔍 Ranking stocks...")
        progress_bar.progress(75)
        ranked = ml_ranker.rank_stocks(stocks_features) or []

        # Step 4: Signals
        status.info("🎯 Generating signals...")
        progress_bar.progress(85)

        top_k = 5
        ranked_top = ranked[:top_k]

        live_prices = datafeed.fetch_batch_live_prices([s[0] for s in ranked_top]) if ranked_top else {}

        signals = []
        for symbol, score in ranked_top:
            if symbol in live_prices and symbol in stocks_features:
                # live_prices[symbol] might be dict or float depending on your DataFeed
                lp = live_prices[symbol]
                # normalize
                current_price = lp["ltp"] if isinstance(lp, dict) and "ltp" in lp else lp

                signal = strategy.generate_signal(symbol, current_price, stocks_features[symbol], score)
                if signal:
                    signal = risk_manager.add_position_size_to_signal(signal)
                    # Ensure expected keys exist
                    signal["ml_score"] = float(signal.get("ml_score", score))
                    signals.append(signal)

        # Risk gate
        max_daily_risk = config["account"]["initial_capital"] * (config["account"]["daily_risk_percent"] / 100.0)
        gated, gate_status = risk_gate_reduce(signals, max_daily_risk)

        progress_bar.progress(100)
        status.success(f"✅ Done! Signals: {len(signals)} | After Risk Gate: {len(gated)} ({gate_status})")

        st.session_state["today_signals"] = gated
        st.session_state["gate_status"] = gate_status
        st.session_state["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        time.sleep(0.4)
        progress_bar.empty()
        status.empty()

    # Display signals
    if st.session_state.get("today_signals"):
        st.subheader(f"📋 Signals ({len(st.session_state['today_signals'])})")
        st.caption(f"Generated at: {st.session_state.get('generated_at','-')} | Risk Gate: {st.session_state.get('gate_status','-')}")

        signals_df = pd.DataFrame(st.session_state["today_signals"])

        # Sector summary (simple)
        if "sector" in signals_df.columns:
            sec = signals_df["sector"].value_counts().reset_index()
            sec.columns = ["Sector", "Signal Count"]
            st.markdown("### 🧊 Sector Participation")
            st.dataframe(sec, use_container_width=True, height=180)
        else:
            st.markdown("### 🧊 Sector Participation")
            st.info("No 'sector' column in signals. (You can add sector mapping later.)")

        # Main table
        show_cols = [c for c in ["symbol", "ml_score", "entry", "stop_loss", "target", "quantity", "risk_amount"] if c in signals_df.columns]
        display_df = signals_df[show_cols].copy()

        rename_map = {
            "symbol": "Symbol",
            "ml_score": "Score",
            "entry": "Entry",
            "stop_loss": "Stop",
            "target": "Target",
            "quantity": "Qty",
            "risk_amount": "Risk ₹",
        }
        display_df = display_df.rename(columns=rename_map)

        st.dataframe(
            display_df.style.format({
                "Score": "{:.2f}",
                "Entry": "₹{:.2f}",
                "Stop": "₹{:.2f}",
                "Target": "₹{:.2f}",
                "Risk ₹": "₹{:.0f}",
            }).background_gradient(subset=["Score"] if "Score" in display_df.columns else [], cmap="RdYlGn"),
            use_container_width=True,
            height=320
        )

        st.info("💡 Execute manually in Zerodha/Upstox. Do NOT enter before entry triggers.")
        # -------- Copy Orders --------
st.markdown("## 📋 Copy Orders (Zerodha / Upstox)")

for sig in st.session_state["today_signals"]:
    symbol = sig["symbol"]
    qty = int(sig.get("quantity", 0))
    entry = float(sig.get("entry", 0))
    stop = float(sig.get("stop_loss", 0))
    target = float(sig.get("target", 0))

    zerodha = f"""
BUY {symbol}
QTY {qty}
LIMIT {entry:.2f}

STOP LOSS:
SELL {symbol}
QTY {qty}
SL {stop:.2f}

TARGET:
SELL {symbol}
QTY {qty}
LIMIT {target:.2f}
"""

    upstox = f"""
{symbol}
Buy Qty: {qty}
Entry: {entry:.2f}
StopLoss: {stop:.2f}
Target: {target:.2f}
"""

    with st.expander(f"📄 {symbol} Order Copy"):
        st.text_area("Zerodha Format", zerodha, height=120)
        st.text_area("Upstox Format", upstox, height=80)


        # Why this trade + Candlestick
        if show_why_panel and "Symbol" in display_df.columns:
            st.markdown("---")
            st.subheader("🧠 Why this trade? (Explain + Chart)")

            pick = st.selectbox("Select a symbol", display_df["Symbol"].tolist(), index=0)
            row = signals_df[signals_df["symbol"] == pick].iloc[0].to_dict()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ML Score", f"{float(row.get('ml_score', 0)):.2f}")
            c2.metric("Entry", f"₹{float(row.get('entry', 0)):.2f}")
            c3.metric("Stop", f"₹{float(row.get('stop_loss', 0)):.2f}")
            c4.metric("Target", f"₹{float(row.get('target', 0)):.2f}")

            st.write("**Decision summary:**")
            bullets = []
            bullets.append(f"- Model confidence is **{float(row.get('ml_score', 0)):.2f}** (higher = better).")
            bullets.append(f"- Stop-loss is set using ATR logic (controls risk).")
            bullets.append(f"- Target is based on Reward:Risk ratio from config.")
            if index_data:
                bullets.append(f"- Market regime: **{index_data.get('regime','UNKNOWN')}** (better to trade in BULL/SIDEWAYS).")
            for b in bullets:
                st.write(b)

            candles = fetch_candles_cached(pick, period="3mo")
            make_candlestick(candles, f"{pick} Candlestick (last ~3 months)")

# =========================================================
# TAB 2: MONITOR (Auto refresh optional)
# =========================================================
with tab_monitor:
    st.header("🔴 Live Monitor")

    if not st.session_state.get("today_signals"):
        st.info("👈 Generate signals in TODAY tab first")
    else:
        signals = st.session_state["today_signals"]
        symbols = [s["symbol"] for s in signals if "symbol" in s]

        # Auto refresh toggle (works if streamlit-autorefresh installed)
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            auto = st.checkbox("Auto-refresh (every 10s)", value=True, disabled=not AUTOREFRESH_OK)
            if not AUTOREFRESH_OK:
                st.caption("Install `streamlit-autorefresh` to enable auto refresh.")
        with c2:
            refresh_secs = st.selectbox("Refresh interval", [5, 10, 15, 30], index=1)
        with c3:
            manual = st.button("🔄 Manual Refresh")

        if auto and AUTOREFRESH_OK:
            st_autorefresh(interval=refresh_secs * 1000, key="monitor_refresh")

        if auto or manual:
            with st.spinner("Fetching live prices..."):
                live_prices = datafeed.fetch_batch_live_prices(symbols)

            monitor_rows = []
            for sig in signals:
                symbol = sig["symbol"]
                if symbol not in live_prices:
                    continue

                lp = live_prices[symbol]
                ltp = lp["ltp"] if isinstance(lp, dict) and "ltp" in lp else lp

                status = strategy.check_signal_status(sig, ltp)
                distance = strategy.get_distance_to_entry(sig, ltp)

                # Action guidance
                action = "WAIT"
                if status == "WAITING":
                    action = "WAIT: Buy only if entry breaks"
                elif status == "TRIGGERED":
                    action = "ACTION: Entry hit → place SL + Target"
                elif status == "TARGET":
                    action = "BOOK: Target hit"
                elif status == "STOPPED":
                    action = "EXIT: Stop hit"
                else:
                    action = "CHECK"

                monitor_rows.append({
                    "Symbol": symbol,
                    "LTP": float(ltp),
                    "Entry": float(sig.get("entry", 0)),
                    "Stop": float(sig.get("stop_loss", 0)),
                    "Target": float(sig.get("target", 0)),
                    "Status": status,
                    "Distance %": float(distance),
                    "Action": action,
                })

            if monitor_rows:
                monitor_df = pd.DataFrame(monitor_rows)

                def color_status(val):
                    colors = {
                        "WAITING": "background-color: #FFF3CD",
                        "TRIGGERED": "background-color: #D1ECF1",
                        "TARGET": "background-color: #D4EDDA",
                        "STOPPED": "background-color: #F8D7DA",
                    }
                    return colors.get(val, "")

                st.dataframe(
                    monitor_df.style.applymap(color_status, subset=["Status"]).format({
                        "LTP": "₹{:.2f}",
                        "Entry": "₹{:.2f}",
                        "Stop": "₹{:.2f}",
                        "Target": "₹{:.2f}",
                        "Distance %": "{:+.2f}%",
                    }),
                    use_container_width=True,
                    height=420
                )

                st.caption("If Status becomes TRIGGERED → enter manually and place SL immediately.")
            else:
                st.warning("No monitor rows generated. Try manual refresh.")
        else:
            st.info("Toggle auto-refresh or click Manual Refresh to start monitoring.")

# =========================================================
# TAB 3: JOURNAL (FIXED - was wrong in your code)
# =========================================================
with tab_journal:
    st.header("📓 Trading Journal")

    # Add trade (manual)
    with st.expander("➕ Add Trade (Manual paper entry)"):
        col1, col2 = st.columns(2)

        with col1:
            trade_symbol = st.selectbox("Symbol", config["symbols"]["stocks"][:10])
            entry_price = st.number_input("Entry", min_value=0.0, step=0.05, format="%.2f")
            quantity = st.number_input("Qty", min_value=1, step=1)

        with col2:
            stop_loss = st.number_input("Stop", min_value=0.0, step=0.05, format="%.2f")
            target = st.number_input("Target", min_value=0.0, step=0.05, format="%.2f")

        if st.button("Add Trade to Journal"):
            signal = {
                "symbol": trade_symbol,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "quantity": quantity,
                "ml_score": 0.0,
            }
            journal.add_trade(signal)
            st.success("✅ Added")
            st.rerun()

    # Metrics
    metrics = journal.calculate_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", metrics.get("total_trades", 0))
    c2.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
    c3.metric("P&L", f"₹{metrics.get('total_pnl', 0):,.0f}")
    c4.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")

    st.subheader("Recent Closed Trades")
    closed = journal.get_closed_trades()
    if closed is not None and len(closed) > 0:
        cols = [c for c in ["symbol", "entry_price", "exit_price", "pnl"] if c in closed.columns]
        st.dataframe(closed[cols].tail(15), use_container_width=True)

        # Equity curve
        equity = journal.get_equity_curve()
        if equity is not None and len(equity) > 0:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=equity["exit_time"],
                    y=equity["cumulative_pnl"],
                    mode="lines",
                    line=dict(color="green", width=2),
                )
            )
            fig.update_layout(title="Equity Curve", height=320)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades yet. Add paper trades above.")

# =========================================================
# TAB 4: SETTINGS (FIXED)
# =========================================================
with tab_settings:
    st.header("⚙️ Settings")

    st.info("Edit `config.yaml` to change settings, then redeploy/restart the app.")
    st.code(
        f"""
Current Settings:
- Account Size: ₹{config['account']['initial_capital']:,}
- Daily Risk: {config['account']['daily_risk_percent']}%
- Max Trades: {config['account']['max_trades_per_day']}
- ATR Multiplier: {config['risk']['atr_stop_multiplier']}
- Reward Ratio: {config['risk']['reward_ratio']}
- Min ML Score: {config['ml']['min_score']}
""".strip()
    )

    st.caption("Tip: If Streamlit Cloud is used, push config.yaml changes to GitHub and it redeploys automatically.")

# =========================================================
# TAB 5: HELP
# =========================================================
with tab_help:
    st.header("❓ Quick Help")

    st.markdown(
        """
## How to Use

### 1) Generate Signals
- Go to **TODAY** tab
- Click **Generate Signals**
- Review Entry / Stop / Target / Qty

### 2) Monitor Live
- Go to **MONITOR**
- Turn on **Auto-refresh**
- When Status becomes **TRIGGERED**:
  - Place BUY order manually in Zerodha/Upstox
  - Place STOP immediately
  - Set target alert

### 3) Record Trades
- Go to **JOURNAL**
- Add paper trades (entry/exit)
- Track Win rate + Equity curve

---

## Trading Discipline Rules (must)
- Never enter early (wait for TRIGGERED)
- Always place stop-loss
- Total daily risk must be within limit
- If the market regime is BEAR, avoid longs

⚠️ Educational tool only. Markets are risky.
"""
    )

st.markdown("---")
st.caption("Trading Assistant v1.1 - Fast Pro Cockpit")
