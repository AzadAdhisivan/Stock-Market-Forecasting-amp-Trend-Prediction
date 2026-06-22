import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
import numpy as np


scaler = MinMaxScaler()

for file in os.listdir("../data/"):
    if file.endswith(".csv"):
        filepath = f"../data/{file}"
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        
        ticker = file.replace("_historical.csv", "")
        
        # Normalize closing price
        df["Close_Normalized"] = scaler.fit_transform(df[["Close"]])
        
        # Individual histogram
        plt.figure(figsize=(10, 5))
        plt.hist(df["Close_Normalized"], bins=50, color="steelblue", edgecolor="black")
        plt.title(f"{ticker} - Normalized Closing Price Distribution")
        plt.xlabel("Normalized Price (0-1)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        
        os.makedirs("../plots", exist_ok=True)
        plt.savefig(f"../plots/{ticker}_normalized_histogram.png")
        plt.show()
        
        print(f"Saved: plots/{ticker}_normalized_histogram.png")


for file in os.listdir("../data/"):
    if file.endswith(".csv"):
        filepath = f"../data/{file}"
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        
        ticker = file.replace("_historical.csv", "")
        
        # Convert volume to millions
        df["Volume_M"] = df["Volume"] / 1_000_000
        
        plt.figure(figsize=(8, 6))
        plt.boxplot(df["Volume_M"], patch_artist=True,
                    boxprops=dict(facecolor="steelblue", color="black"),
                    medianprops=dict(color="red", linewidth=2),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    flierprops=dict(marker="o", color="orange", markersize=4))
        
        plt.title(f"{ticker} - Volume Boxplot")
        plt.ylabel("Volume (M)")
        plt.xticks([1], [ticker])
        plt.tight_layout()
        
        os.makedirs("../plots", exist_ok=True)
        plt.savefig(f"../plots/{ticker}_volume_boxplot.png")
        plt.show()
        
        print(f"Saved: plots/{ticker}_volume_boxplot.png")


for file in os.listdir("../data/"):
    if file.endswith(".csv"):
        filepath = f"../data/{file}"
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        
        ticker = file.replace("_historical.csv", "")
        
        # Calculate daily returns in %
        df["Daily_Returns"] = df["Close"].pct_change() * 100
        df = df.dropna()
        
        # KDE distribution plot
        returns = df["Daily_Returns"]
        kde = gaussian_kde(returns)
        x = np.linspace(returns.min(), returns.max(), 1000)
        y = kde(x)
        
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, color="steelblue", linewidth=2)
        plt.fill_between(x, y, alpha=0.3, color="steelblue")
        plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Return")
        plt.axvline(returns.mean(), color="green", linestyle="--", linewidth=1.5, label=f"Mean: {returns.mean():.2f}%")
        
        plt.title(f"{ticker} - Daily Returns Distribution")
        plt.xlabel("Daily Return (%)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        
        os.makedirs("../plots", exist_ok=True)
        plt.savefig(f"../plots/{ticker}_daily_returns_distribution.png")
        plt.show()
        
        print(f"Saved: plots/{ticker}_daily_returns_distribution.png")


for file in os.listdir("../data/"):
    if file.endswith(".csv"):
        filepath = f"../data/{file}"
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        
        ticker = file.replace("_historical.csv", "")
        
        # Calculate rolling means
        df["RM_7"] = df["Close"].rolling(window=7).mean()
        df["RM_30"] = df["Close"].rolling(window=30).mean()
        df["RM_90"] = df["Close"].rolling(window=90).mean()
        
        # Calculate daily returns for volatility
        df["Daily_Returns"] = df["Close"].pct_change() * 100
        
        # Define periods
        # Bullish: 7-day above 90-day rolling mean
        # Bearish: 7-day below 90-day rolling mean
        # Volatile: daily return > 3% or < -3%
        df["Bullish"] = df["RM_7"] > df["RM_90"]
        df["Bearish"] = df["RM_7"] < df["RM_90"]
        df["Volatile"] = df["Daily_Returns"].abs() > 3
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(f"{ticker} - Time Series Analysis", fontsize=16)
        
        # ── Plot 1: Closing Price + Rolling Means + Shading ──
        ax1.plot(df.index, df["Close"], color="black", linewidth=1, label="Close Price", zorder=3)
        ax1.plot(df.index, df["RM_7"], color="blue", linewidth=1, linestyle="--", label="7-Day RM", zorder=3)
        ax1.plot(df.index, df["RM_30"], color="orange", linewidth=1, linestyle="--", label="30-Day RM", zorder=3)
        ax1.plot(df.index, df["RM_90"], color="red", linewidth=1.5, linestyle="--", label="90-Day RM", zorder=3)
        
        # Shade bullish periods (green)
        ax1.fill_between(df.index, df["Close"].min(), df["Close"].max(),
                         where=df["Bullish"], alpha=0.1, color="green", label="Bullish")
        
        # Shade bearish periods (red)
        ax1.fill_between(df.index, df["Close"].min(), df["Close"].max(),
                         where=df["Bearish"], alpha=0.1, color="red", label="Bearish")
        
        # Shade volatile periods (yellow)
        ax1.fill_between(df.index, df["Close"].min(), df["Close"].max(),
                         where=df["Volatile"], alpha=0.3, color="yellow", label="Volatile")
        
        ax1.set_title("Closing Price + Rolling Means + Trend Periods")
        ax1.set_ylabel("Price (USD)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # ── Plot 2: Volume Bar Chart ──
        df["Volume_M"] = df["Volume"] / 1_000_000
        ax2.bar(df.index, df["Volume_M"], color="steelblue", alpha=0.7, width=1)
        ax2.set_title("Volume Over Time")
        ax2.set_ylabel("Volume (M)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs("../plots", exist_ok=True)
        plt.savefig(f"../plots/{ticker}_timeseries.png")
        plt.show()
        
        print(f"Saved: plots/{ticker}_timeseries.png")