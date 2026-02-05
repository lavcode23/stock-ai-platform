# 📈 Trading Assistant

A professional, beginner-friendly trading assistant for Indian stocks built with Python and Streamlit.

## Features

- ✅ Live Indian stock data (NIFTY + top stocks)
- ✅ ML-powered stock ranking (RandomForest)
- ✅ Automated signal generation with entry, stop, target
- ✅ ATR-based stop losses
- ✅ Risk-based position sizing
- ✅ Daily risk limits (1% max risk, 3 trades/day)
- ✅ Live price monitoring
- ✅ Sector strength analysis
- ✅ Paper trading journal
- ✅ Performance metrics (win rate, Sharpe, drawdown)
- ✅ Historical backtesting
- ✅ Professional Streamlit UI

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
cd stocks-reader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open browser at: `http://localhost:8501`

## Usage

### TODAY Tab
1. Check market regime (Bull/Bear/Sideways)
2. Click "Generate Today's Signals"
3. Review generated signals with entry, stop, target
4. Check pre-trade checklist

### LIVE MONITOR Tab
1. Auto-refreshes every 10 seconds
2. Shows live prices vs entry/stop/target
3. Displays status (WAITING/TRIGGERED/TARGET/STOPPED)
4. Provides action recommendations

### SECTORS Tab
1. Click "Analyze Sectors"
2. View sector strength rankings
3. See which sectors have most strong stocks

### JOURNAL Tab
1. Add paper trades manually
2. Close trades with exit price
3. View performance metrics
4. Analyze equity curve and P&L distribution

### BACKTEST Tab
1. Select a stock
2. Run historical backtest
3. View results and equity curve

### SETTINGS Tab
1. Adjust account size
2. Modify risk parameters
3. Change ATR multiplier and reward ratio

## Configuration

Edit `config.yaml` to customize:

- Account size and risk limits
- Stock symbols to track
- Sector classifications
- ML parameters
- Risk management rules

## Risk Management

- **Max Daily Risk**: 1% of account
- **Max Trades/Day**: 3 trades
- **Stop Loss**: ATR-based (2x ATR default)
- **Position Sizing**: Risk-based calculation
- **Reward Ratio**: 2:1 default (configurable)

## Disclaimer

⚠️ **IMPORTANT**: This tool is for **educational purposes only**.

- No guarantees of profitability
- Past performance ≠ future results
- Trading involves substantial risk
- Always do your own research
- Start with paper trading
- Never risk more than you can afford to lose

This is NOT financial advice. Use at your own risk.

## Technical Details

### Data Source
- Yahoo Finance via `yfinance`
- Indian stocks (NSE)
- Historical and live data

### ML Model
- RandomForest classifier
- Trained on historical features
- Predicts probability of upward move
- 70/30 train/test split

### Technical Indicators
- RSI (14)
- MACD (12, 26, 9)
- Bollinger Bands (20, 2)
- ATR (14)
- ADX (14)
- Stochastic (14, 3)
- Moving averages (10, 20, 50)
- Volume ratio
- Momentum indicators

### Position Sizing Formula
```
Quantity = (Account × Risk%) / (Entry - Stop)
```

## File Structure
```
stocks-reader/
├── app.py              # Main Streamlit application
├── datafeed.py         # Data fetching (yfinance)
├── features.py         # Technical indicators
├── ml_ranker.py        # ML model (RandomForest)
├── strategy.py         # Signal generation
├── risk.py             # Risk management
├── journal.py          # Trade journaling
├── backtest.py         # Historical backtesting
├── config.yaml         # Configuration
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Support

This is a learning project. For issues or questions:
1. Check the EXPLAIN tab in the app
2. Review the code comments
3. Modify as needed for your use case

## License

MIT License - Free to use and modify

---

**Happy Trading! Remember: Education first, profits second.** 📚📈
