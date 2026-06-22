import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
EPOCHS = 200
BATCH_SIZE = 32

results = {}

for ticker in tickers:
    print(f"\n{'='*50}")
    print(f"  Training {ticker} LSTM Model")
    print(f"{'='*50}")

    # ── Step 1: Load Sequences ──
    X = np.load(f"../data/sequences/{ticker}_X.npy")
    Y = np.load(f"../data/sequences/{ticker}_Y.npy")

    # ── Step 2: Load Classification Target ──
    df = pd.read_csv(f"../data/features/{ticker}_features.csv",
                     index_col="Date", parse_dates=True)
    df = df.dropna()
    Y_class = df["Trend_Label"].values[30:]

    # ── Step 3: Train/Test Split (80/20) ──
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    Y_reg_train, Y_reg_test = Y[:split], Y[split:]
    Y_cls_train, Y_cls_test = Y_class[:split], Y_class[split:]

    print(f"Train samples  : {len(X_train)}")
    print(f"Test samples   : {len(X_test)}")
    print(f"X shape        : {X.shape}")

    # ── Step 4: Build Regression LSTM ──
    reg_model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(128, return_sequences=True,
             kernel_regularizer=l2(0.001)),   # ← first LSTM layer
        Dropout(0.3),
        LSTM(64, return_sequences=False,
             kernel_regularizer=l2(0.001)),   # ← second LSTM layer
        Dropout(0.3),
        Dense(32, activation="relu",
              kernel_regularizer=l2(0.001)),
        Dense(1, activation="linear")          # ← regression output
    ])

    reg_model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"]
    )

    reg_model.summary()

    early_stop_reg = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    os.makedirs("../models", exist_ok=True)
    checkpoint_reg = ModelCheckpoint(
        f"../models/{ticker}_lstm_regression_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    )

    print(f"\nTraining Regression LSTM for {ticker}...")
    reg_history = reg_model.fit(
        X_train, Y_reg_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop_reg, checkpoint_reg],
        verbose=1
    )

    # ── Step 5: Regression Evaluation ──
    Y_reg_pred = reg_model.predict(X_test).flatten()

    scaler_Y = joblib.load(f"../data/scalers/{ticker}_scaler_Y.pkl")
    Y_reg_test_real = scaler_Y.inverse_transform(
        Y_reg_test.reshape(-1, 1)).flatten()
    Y_reg_pred_real = scaler_Y.inverse_transform(
        Y_reg_pred.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(Y_reg_test_real, Y_reg_pred_real)
    rmse = np.sqrt(mean_squared_error(Y_reg_test_real, Y_reg_pred_real))
    r2   = r2_score(Y_reg_test_real, Y_reg_pred_real)

    print(f"\n📈 Regression Metrics:")
    print(f"  MAE  : ${mae:.2f}")
    print(f"  RMSE : ${rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    # ── Step 6: Build Classification LSTM ──
    cls_model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(128, return_sequences=True,
             kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        LSTM(64, return_sequences=False,
             kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(32, activation="relu",
              kernel_regularizer=l2(0.001)),
        Dense(1, activation="sigmoid")         # ← classification output
    ])

    cls_model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop_cls = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True
    )

    checkpoint_cls = ModelCheckpoint(
        f"../models/{ticker}_lstm_classification_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=0
    )

    print(f"\nTraining Classification LSTM for {ticker}...")
    cls_history = cls_model.fit(
        X_train, Y_cls_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop_cls, checkpoint_cls],
        verbose=1
    )

    # ── Step 7: Classification Evaluation ──
    Y_cls_pred_prob = cls_model.predict(X_test).flatten()
    Y_cls_pred = (Y_cls_pred_prob > 0.5).astype(int)

    acc       = accuracy_score(Y_cls_test, Y_cls_pred)
    precision = precision_score(Y_cls_test, Y_cls_pred, zero_division=0)
    recall    = recall_score(Y_cls_test, Y_cls_pred, zero_division=0)
    f1        = f1_score(Y_cls_test, Y_cls_pred, zero_division=0)
    roc_auc   = roc_auc_score(Y_cls_test, Y_cls_pred_prob)

    print(f"\n📊 Classification Metrics:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")

    # ── Step 8: Save Models ──
    reg_model.save(f"../models/{ticker}_lstm_regression_model.keras")
    cls_model.save(f"../models/{ticker}_lstm_classification_model.keras")

    print(f"\n✅ Models saved:")
    print(f"  ../models/{ticker}_lstm_regression_model.keras")
    print(f"  ../models/{ticker}_lstm_classification_model.keras")

    # ── Step 9: Plot Training History ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{ticker} - LSTM Training History", fontsize=14)

    # Regression loss
    axes[0, 0].plot(reg_history.history["loss"], label="Train Loss")
    axes[0, 0].plot(reg_history.history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Regression Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Regression MAE
    axes[0, 1].plot(reg_history.history["mae"], label="Train MAE")
    axes[0, 1].plot(reg_history.history["val_mae"], label="Val MAE")
    axes[0, 1].set_title("Regression MAE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Classification accuracy
    axes[1, 0].plot(cls_history.history["accuracy"], label="Train Accuracy")
    axes[1, 0].plot(cls_history.history["val_accuracy"], label="Val Accuracy")
    axes[1, 0].set_title("Classification Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Classification loss
    axes[1, 1].plot(cls_history.history["loss"], label="Train Loss")
    axes[1, 1].plot(cls_history.history["val_loss"], label="Val Loss")
    axes[1, 1].set_title("Classification Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("../plots", exist_ok=True)
    plt.savefig(f"../plots/{ticker}_lstm_training_history.png")
    plt.show()

    # Store results
    results[ticker] = {
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Accuracy": acc, "Precision": precision,
        "Recall": recall, "F1": f1, "ROC_AUC": roc_auc
    }

# ── Step 10: Final Summary ──
print(f"\n{'='*70}")
print("  LSTM FINAL RESULTS SUMMARY")
print(f"{'='*70}")
summary_df = pd.DataFrame(results).T
print(summary_df.round(4))
summary_df.to_csv("../models/lstm_results_summary.csv")
print("\nSaved: ../models/lstm_results_summary.csv")

# ── Step 11: ANN vs RNN vs LSTM Comparison ──
print(f"\n{'='*70}")
print("  ANN vs RNN vs LSTM COMPARISON")
print(f"{'='*70}")

try:
    ann_results  = pd.read_csv("../models/results_summary.csv", index_col=0)
    rnn_results  = pd.read_csv("../models/rnn_results_summary.csv", index_col=0)
    lstm_results = pd.read_csv("../models/lstm_results_summary.csv", index_col=0)

    for ticker in tickers:
        print(f"\n{ticker}:")
        print(f"  {'Metric':<12} {'ANN':>10} {'RNN':>10} {'LSTM':>10} {'Winner':>10}")
        print(f"  {'-'*54}")
        for metric in ["MAE", "RMSE", "R2", "Accuracy"]:
            if all(metric in df.columns for df in [ann_results, rnn_results, lstm_results]):
                ann_val  = ann_results.loc[ticker, metric]
                rnn_val  = rnn_results.loc[ticker, metric]
                lstm_val = lstm_results.loc[ticker, metric]

                if metric in ["MAE", "RMSE"]:
                    best = min(ann_val, rnn_val, lstm_val)
                    winner = "ANN" if best == ann_val else "RNN" if best == rnn_val else "LSTM"
                else:
                    best = max(ann_val, rnn_val, lstm_val)
                    winner = "ANN" if best == ann_val else "RNN" if best == rnn_val else "LSTM"

                print(f"  {metric:<12} {ann_val:>10.4f} {rnn_val:>10.4f} {lstm_val:>10.4f} {winner:>10}")
except Exception as e:
    print(f"Comparison error: {e}")