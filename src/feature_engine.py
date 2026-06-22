import pandas as pd
import os

for file in os.listdir("../data/"):
    if file.endswith(".csv"):
        filepath = f"../data/{file}"
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        
        ticker = file.replace("_historical.csv", "")
        
        # 1. Daily Return
        df["Daily_Return"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
        
        # 2. Moving Averages
        df["MA7"] = df["Close"].rolling(window=7).mean()
        df["MA30"] = df["Close"].rolling(window=30).mean()
        df["MA90"] = df["Close"].rolling(window=90).mean()
        
        # 3. Price Difference
        df["Price_Difference"] = df["Close"] - df["Open"]
        
        # 4. High-Low Spread
        df["High_Low_Spread"] = df["High"] - df["Low"]
        
        # 5. Lag Features
        df["Lag_1"] = df["Close"].shift(1)
        df["Lag_3"] = df["Close"].shift(3)
        df["Lag_7"] = df["Close"].shift(7)
        
        # 6. Rolling Volatility
        df["Volatility_7"] = df["Daily_Return"].rolling(window=7).std()
        df["Volatility_30"] = df["Daily_Return"].rolling(window=30).std()
        
        # 7. Momentum Indicator
        df["Momentum_7"] = df["Close"] - df["Close"].shift(7)
        
        # 8. Trend Label (Target Variable)
        df["Trend_Label"] = (df["Close"] > df["Close"].shift(1)).astype(int)
        
        # Drop NaN rows created by rolling/shift
        df = df.dropna()
        
        # Save to new CSV
        os.makedirs("../data/features", exist_ok=True)
        df.to_csv(f"../data/features/{ticker}_features.csv")
        
        print(f"\n{'='*50}")
        print(f"  {ticker} Feature Engineering Done")
        print(f"{'='*50}")
        print(f"Total Features: {len(df.columns)}")
        print(f"Total Rows: {len(df)}")
        print(f"Features: {list(df.columns)}")