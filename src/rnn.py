import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
EPOCHS = 200
BATCH_SIZE = 32

results = {}

for ticker in tickers:
    print(f"\n{'='*50}")
    print(f"  Training {ticker} RNN Model")
    print(f"{'='*50}")

    # ── Step 1: Load Sequences ──
    X = np.load(f"../data/sequences/{ticker}_X.npy")
    Y = np.load(f"../data/sequences/{ticker}_Y.npy")

    # ── Step 2: Load Classification Target ──
    df = pd.read_csv(f"../data/features/{ticker}_features.csv", index_col="Date", parse_dates=True)
    df = df.dropna()
    Y_class = df["Trend_Label"].values[30:]

    # ── Step 3: Train/Test Split (80/20) ──
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    Y_reg_train, Y_reg_test = Y[:split], Y[split:]
    Y_cls_train, Y_cls_test = Y_class[:split], Y_class[split:]

    print(f"Train samples  : {len(X_train)}")
    print(f"Test samples   : {len(X_test)}")

    # ── Step 4: Build Regression RNN ──
    reg_model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        SimpleRNN(64, activation="relu", return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")
    ])

    reg_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    reg_model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    os.makedirs("../models", exist_ok=True)
    checkpoint_reg = ModelCheckpoint(
        f"../models/{ticker}_rnn_regression_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    )

    reg_history = reg_model.fit(
        X_train, Y_reg_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint_reg],
        verbose=1
    )

    # ── Step 5: Regression Evaluation ──
    Y_reg_pred = reg_model.predict(X_test).flatten()

    scaler_Y = joblib.load(f"../data/scalers/{ticker}_scaler_Y.pkl")
    Y_reg_test_real = scaler_Y.inverse_transform(Y_reg_test.reshape(-1, 1)).flatten()
    Y_reg_pred_real = scaler_Y.inverse_transform(Y_reg_pred.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(Y_reg_test_real, Y_reg_pred_real)
    rmse = np.sqrt(mean_squared_error(Y_reg_test_real, Y_reg_pred_real))
    r2   = r2_score(Y_reg_test_real, Y_reg_pred_real)

    print(f"\n📈 Regression Metrics:")
    print(f"  MAE  : ${mae:.2f}")
    print(f"  RMSE : ${rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    # ── Step 6: Build Classification RNN ──
    cls_model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        SimpleRNN(64, activation="relu", return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    cls_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_cls = ModelCheckpoint(
        f"../models/{ticker}_rnn_classification_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=0
    )

    cls_history = cls_model.fit(
        X_train, Y_cls_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint_cls],
        verbose=1
    )

    # ── Step 7: Classification Evaluation ──
    Y_cls_pred_prob = cls_model.predict(X_test).flatten()
    Y_cls_pred = (Y_cls_pred_prob > 0.5).astype(int)

    acc = accuracy_score(Y_cls_test, Y_cls_pred)
    f1  = f1_score(Y_cls_test, Y_cls_pred, zero_division=0)

    print(f"\n📊 Classification Metrics:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    # ── Step 8: Save Models ──
    reg_model.save(f"../models/{ticker}_rnn_regression_model.keras")
    cls_model.save(f"../models/{ticker}_rnn_classification_model.keras")

    print(f"\n✅ Models saved:")
    print(f"  ../models/{ticker}_rnn_regression_model.keras")
    print(f"  ../models/{ticker}_rnn_classification_model.keras")

    # ── Step 9: Plot Training History ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ticker} - RNN Training History", fontsize=14)

    ax1.plot(reg_history.history["loss"], label="Train Loss")
    ax1.plot(reg_history.history["val_loss"], label="Val Loss")
    ax1.set_title("Regression Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(cls_history.history["accuracy"], label="Train Accuracy")
    ax2.plot(cls_history.history["val_accuracy"], label="Val Accuracy")
    ax2.set_title("Classification Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("../plots", exist_ok=True)
    plt.savefig(f"../plots/{ticker}_rnn_training_history.png")
    plt.show()

    # Store results
    results[ticker] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Accuracy": acc,
        "F1_Score": f1
    }

# ── Step 10: Summary Table ──
print(f"\n{'='*70}")
print("  RNN FINAL RESULTS SUMMARY")
print(f"{'='*70}")
summary_df = pd.DataFrame(results).T
print(summary_df.round(4))
summary_df.to_csv("../models/rnn_results_summary.csv")
print("\nSaved: ../models/rnn_results_summary.csv")

# ── Step 11: Compare ANN vs RNN ──
print(f"\n{'='*70}")
print("  ANN vs RNN COMPARISON")
print(f"{'='*70}")

try:
    ann_results = pd.read_csv("../models/results_summary.csv", index_col=0)
    rnn_results = pd.read_csv("../models/rnn_results_summary.csv", index_col=0)

    for ticker in tickers:
        print(f"\n{ticker}:")
        print(f"  {'Metric':<12} {'ANN':>10} {'RNN':>10} {'Winner':>10}")
        print(f"  {'-'*44}")
        for metric in ["MAE", "RMSE", "R2", "Accuracy", "F1_Score"]:
            if metric in ann_results.columns and metric in rnn_results.columns:
                ann_val = ann_results.loc[ticker, metric]
                rnn_val = rnn_results.loc[ticker, metric]
                if metric in ["MAE", "RMSE"]:
                    winner = "RNN" if rnn_val < ann_val else "ANN"
                else:
                    winner = "RNN" if rnn_val > ann_val else "ANN"
                print(f"  {metric:<12} {ann_val:>10.4f} {rnn_val:>10.4f} {winner:>10}")
except:
    print("ANN results not found for comparison")