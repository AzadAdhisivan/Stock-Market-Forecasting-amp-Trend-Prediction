import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
EPOCHS = 100
BATCH_SIZE = 32

results = {}

for ticker in tickers:
    print(f"\n{'='*50}")
    print(f"  Training {ticker} Model")
    print(f"{'='*50}")

    # ── Step 1: Load Sequences ──
    X = np.load(f"../data/sequences/{ticker}_X.npy")
    Y = np.load(f"../data/sequences/{ticker}_Y.npy")

    # ── Step 2: Load Features for Classification Target ──
    df = pd.read_csv(f"../data/features/{ticker}_features.csv", index_col="Date", parse_dates=True)
    df = df.dropna()
    Y_class = df["Trend_Label"].values[30:]  # align with sequences

    # ── Step 3: Train/Test Split (80/20) ──
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    Y_reg_train, Y_reg_test = Y[:split], Y[split:]
    Y_cls_train, Y_cls_test = Y_class[:split], Y_class[split:]

    print(f"Train samples  : {len(X_train)}")
    print(f"Test samples   : {len(X_test)}")

    # ── Step 4: Build Regression Model ──
    reg_model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")  # Regression output
    ])

    reg_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    reg_history = reg_model.fit(
        X_train, Y_reg_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    # ── Step 5: Regression Evaluation ──
    Y_reg_pred = reg_model.predict(X_test).flatten()

    # Inverse transform to real prices
    scaler_Y = joblib.load(f"../data/scalers/{ticker}_scaler_Y.pkl")
    Y_reg_test_real = scaler_Y.inverse_transform(Y_reg_test.reshape(-1, 1)).flatten()
    Y_reg_pred_real = scaler_Y.inverse_transform(Y_reg_pred.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(Y_reg_test_real, Y_reg_pred_real)
    rmse = np.sqrt(mean_squared_error(Y_reg_test_real, Y_reg_pred_real))
    r2   = r2_score(Y_reg_test_real, Y_reg_pred_real)

    print(f"\n Regression Metrics:")
    print(f"  MAE  : ${mae:.2f}")
    print(f"  RMSE : ${rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    # ── Step 6: Build Classification Model ──
    cls_model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")  # Classification output
    ])

    cls_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    cls_history = cls_model.fit(
        X_train, Y_cls_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    # ── Step 7: Classification Evaluation ──
    Y_cls_pred_prob = cls_model.predict(X_test).flatten()
    Y_cls_pred = (Y_cls_pred_prob > 0.5).astype(int)

    acc       = accuracy_score(Y_cls_test, Y_cls_pred)
    precision = precision_score(Y_cls_test, Y_cls_pred)
    recall    = recall_score(Y_cls_test, Y_cls_pred)
    f1        = f1_score(Y_cls_test, Y_cls_pred)
    roc_auc   = roc_auc_score(Y_cls_test, Y_cls_pred_prob)

    print(f"\n Classification Metrics:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")

    # ── Step 8: Save Models ──
    os.makedirs("../models", exist_ok=True)
    reg_model.save(f"../models/{ticker}_regression_model.keras")
    cls_model.save(f"../models/{ticker}_classification_model.keras")

    print(f"\n Models saved:")
    print(f"  ../models/{ticker}_regression_model.keras")
    print(f"  ../models/{ticker}_classification_model.keras")

    # ── Step 9: Plot Training Loss ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ticker} - Training History", fontsize=14)

    ax1.plot(reg_history.history["loss"], label="Train Loss")
    ax1.plot(reg_history.history["val_loss"], label="Val Loss")
    ax1.set_title("Regression Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(cls_history.history["accuracy"], label="Train Accuracy")
    ax2.plot(cls_history.history["val_accuracy"], label="Val Accuracy")
    ax2.set_title("Classification Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    os.makedirs("../plots", exist_ok=True)
    plt.savefig(f"../plots/{ticker}_training_history.png")
    plt.show()

    # Store results
    results[ticker] = {
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Accuracy": acc, "Precision": precision,
        "Recall": recall, "F1": f1, "ROC_AUC": roc_auc
    }

# ── Step 10: Summary Table ──
print(f"\n{'='*70}")
print("  FINAL RESULTS SUMMARY")
print(f"{'='*70}")
summary_df = pd.DataFrame(results).T
print(summary_df.round(4))
summary_df.to_csv("../models/results_summary.csv")
print("\nSaved: ../models/results_summary.csv")