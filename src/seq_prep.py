import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

WINDOW_SIZE = 30

for file in os.listdir("../data/features/"):
    if file.endswith(".csv"):
        filepath = f"../data/features/{file}"
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        
        ticker = file.replace("_features.csv", "")
        
        # ── Step 1: Select Feature Columns ──
        # Drop Trend_Label (classification target, not needed for price prediction)
        feature_cols = [col for col in df.columns if col != "Trend_Label"]
        target_col = "Close"
        
        print(f"\n{'='*50}")
        print(f"  {ticker} Sequence Preparation")
        print(f"{'='*50}")
        print(f"Features used: {feature_cols}")
        
        # ── Step 2: Normalize ──
        scaler_X = MinMaxScaler()
        scaler_Y = MinMaxScaler()
        
        scaled_features = scaler_X.fit_transform(df[feature_cols])
        scaled_target = scaler_Y.fit_transform(df[[target_col]])
        
        # ── Step 3: Create Sequences ──
        X = []
        Y = []
        
        for i in range(WINDOW_SIZE, len(scaled_features)):
            # Previous 30 days of all features
            X.append(scaled_features[i - WINDOW_SIZE:i])
            # Next day close price
            Y.append(scaled_target[i])
        
        # ── Step 4: Convert to Numpy Arrays ──
        X = np.array(X)
        Y = np.array(Y).flatten()
        
        # ── Step 5: Save ──
        os.makedirs("../data/sequences", exist_ok=True)
        np.save(f"../data/sequences/{ticker}_X.npy", X)
        np.save(f"../data/sequences/{ticker}_Y.npy", Y)
        
        # Save scalers for inverse transform later
        os.makedirs("../data/scalers", exist_ok=True)
        import joblib
        joblib.dump(scaler_X, f"../data/scalers/{ticker}_scaler_X.pkl")
        joblib.dump(scaler_Y, f"../data/scalers/{ticker}_scaler_Y.pkl")
        
        # ── Step 6: Document ──
        print(f"\nWindow Size    : {WINDOW_SIZE} days")
        print(f"Total Samples  : {len(X)}")
        print(f"X Shape        : {X.shape} (samples, timesteps, features)")
        print(f"Y Shape        : {Y.shape} (samples,)")
        print(f"\nSaved:")
        print(f"  ../data/sequences/{ticker}_X.npy")
        print(f"  ../data/sequences/{ticker}_Y.npy")
        print(f"  ../data/scalers/{ticker}_scaler_X.pkl")
        print(f"  ../data/scalers/{ticker}_scaler_Y.pkl")