
from fastapi import FastAPI
import yfinance as yf

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Stock AI Platform running"}

@app.get("/price")
def get_price(ticker: str):
    data = yf.Ticker(ticker).history(period="1mo")
    return data.reset_index().to_dict(orient="records")
