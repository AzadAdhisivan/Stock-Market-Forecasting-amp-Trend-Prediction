import yfinance as yf
import os

tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
data = yf.download(tickers, start="2021-06-22", end="2026-06-22")

# Save each stock as its own CSV
for ticker in tickers:
    df = data.xs(ticker, axis=1, level="Ticker")
    df = df.round(2)
    df.to_csv(f"../data/{ticker}_historical.csv")

# print confirmation message
print("Done! Files saved to:")
for ticker in tickers:
    print(f"  ../data/{ticker}_historical.csv")