from fastapi import FastAPI
import yfinance as yf

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Stock AI Platform running"}

@app.get("/candles")
def get_candles(ticker: str = "AAPL", period: str = "3mo"):
    df = yf.download(ticker, period=period, interval="1d")
    df = df.reset_index()

    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": row["Date"].strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"])
        })

    return candles
