import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ── Fix random seeds for reproducibility ──
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Load AAPL data ──
X = np.load("../data/sequences/AAPL_X.npy")
Y = np.load("../data/sequences/AAPL_Y.npy")
scaler_Y = joblib.load("../data/scalers/AAPL_scaler_Y.pkl")

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

os.makedirs("../plots", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# ── Evaluation helper ──
def evaluate(model, X_test, Y_test, scaler_Y):
    Y_pred = model.predict(X_test, verbose=0).flatten()
    Y_test_real = scaler_Y.inverse_transform(Y_test.reshape(-1, 1)).flatten()
    Y_pred_real = scaler_Y.inverse_transform(Y_pred.reshape(-1, 1)).flatten()
    mae  = mean_absolute_error(Y_test_real, Y_pred_real)
    rmse = np.sqrt(mean_squared_error(Y_test_real, Y_pred_real))
    r2   = r2_score(Y_test_real, Y_pred_real)
    return mae, rmse, r2

# ── Model builder ──
def build_model(num_layers, hidden_units, learning_rate, seq_len):
    """
    Build a dense model for AAPL price prediction.

    Parameters
    ----------
    num_layers   : int   – number of hidden Dense layers
    hidden_units : int   – units per hidden layer
    learning_rate: float – Adam learning rate
    seq_len      : int   – input sequence length (used to slice X)
    """
    # Reset seed before each build so every config starts from the same
    # weight initialisation → fair apples-to-apples comparison.
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    model = Sequential()
    model.add(Input(shape=(seq_len, X.shape[2])))
    model.add(Flatten())
    for _ in range(num_layers):
        model.add(Dense(hidden_units, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=Adam(learning_rate), loss="mse", metrics=["mae"])
    return model

# ══════════════════════════════════════════════════════
# 20. HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  20. HYPERPARAMETER TUNING")
print("=" * 60)

# ── Search Space ──
param_grid = {
    "num_layers"   : [1, 2, 3],
    "hidden_units" : [32, 64, 128],
    "batch_size"   : [16, 32, 64],
    "learning_rate": [0.01, 0.001, 0.0001],
    "seq_len"      : [10, 20, 30],
}

# Total combinations = 3^5 = 243 — too many for full grid search.
# We use a staged search:
#   Stage A – tune architecture  (layers × units, others fixed)
#   Stage B – tune training      (batch × LR, best arch fixed)
#   Stage C – tune seq_len       (best arch + training fixed)

ES = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

tuning_results = []   # all individual runs
best_params    = {}   # accumulated best values

# ──────────────────────────────────────────────────────
# STAGE A – Architecture: num_layers × hidden_units
# ──────────────────────────────────────────────────────
print("\n" + "-" * 50)
print("  STAGE A: Architecture Search  (layers × units)")
print("-" * 50)

# Fixed defaults for this stage
FIXED_BATCH = 32
FIXED_LR    = 0.001
FIXED_SEQ   = X.shape[1]   # use full default sequence length

stage_a_results = []
total_a = len(param_grid["num_layers"]) * len(param_grid["hidden_units"])
run_idx = 0

for num_layers, hidden_units in itertools.product(
        param_grid["num_layers"], param_grid["hidden_units"]):
    run_idx += 1
    print(f"\n  [{run_idx}/{total_a}] layers={num_layers}, "
          f"units={hidden_units}  (batch={FIXED_BATCH}, lr={FIXED_LR})")

    model = build_model(num_layers, hidden_units, FIXED_LR, FIXED_SEQ)
    model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=FIXED_BATCH,
        validation_split=0.1,
        callbacks=[ES],
        verbose=0,
    )
    mae, rmse, r2 = evaluate(model, X_test, Y_test, scaler_Y)
    print(f"    → MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")

    row = dict(stage="A", num_layers=num_layers, hidden_units=hidden_units,
               batch_size=FIXED_BATCH, learning_rate=FIXED_LR,
               seq_len=FIXED_SEQ, mae=mae, rmse=rmse, r2=r2)
    stage_a_results.append(row)
    tuning_results.append(row)

best_a = max(stage_a_results, key=lambda x: x["r2"])
best_params["num_layers"]  = best_a["num_layers"]
best_params["hidden_units"] = best_a["hidden_units"]
print(f"\n  ✔ Best architecture: "
      f"layers={best_params['num_layers']}, "
      f"units={best_params['hidden_units']}  "
      f"(R²={best_a['r2']:.4f})")

# ──────────────────────────────────────────────────────
# STAGE B – Training: batch_size × learning_rate
# ──────────────────────────────────────────────────────
print("\n" + "-" * 50)
print("  STAGE B: Training Search  (batch_size × learning_rate)")
print("-" * 50)

stage_b_results = []
total_b = len(param_grid["batch_size"]) * len(param_grid["learning_rate"])
run_idx = 0

# Carry the Stage A winner forward so Stage B includes it without re-running.
# Re-running would give a different number (random variance) and could
# incorrectly make a worse config look like the winner.
stage_b_results.append({
    **best_a,
    "stage": "B",
})
print(f"  [carried] batch={FIXED_BATCH}, lr={FIXED_LR}  "
      f"← Stage A winner (R²={best_a['r2']:.4f})")

for batch_size, learning_rate in itertools.product(
        param_grid["batch_size"], param_grid["learning_rate"]):
    # Skip the combo already carried from Stage A
    if batch_size == FIXED_BATCH and learning_rate == FIXED_LR:
        continue
    run_idx += 1
    print(f"\n  [{run_idx}/{total_b}] batch={batch_size}, lr={learning_rate}  "
          f"(layers={best_params['num_layers']}, "
          f"units={best_params['hidden_units']})")

    model = build_model(best_params["num_layers"],
                        best_params["hidden_units"],
                        learning_rate, FIXED_SEQ)
    model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[ES],
        verbose=0,
    )
    mae, rmse, r2 = evaluate(model, X_test, Y_test, scaler_Y)
    print(f"    → MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")

    row = dict(stage="B",
               num_layers=best_params["num_layers"],
               hidden_units=best_params["hidden_units"],
               batch_size=batch_size, learning_rate=learning_rate,
               seq_len=FIXED_SEQ, mae=mae, rmse=rmse, r2=r2)
    stage_b_results.append(row)
    tuning_results.append(row)

best_b = max(stage_b_results, key=lambda x: x["r2"])
best_params["batch_size"]    = best_b["batch_size"]
best_params["learning_rate"] = best_b["learning_rate"]
print(f"\n  ✔ Best training: "
      f"batch={best_params['batch_size']}, "
      f"lr={best_params['learning_rate']}  "
      f"(R²={best_b['r2']:.4f})")

# ──────────────────────────────────────────────────────
# STAGE C – Sequence Length
# ──────────────────────────────────────────────────────
print("\n" + "-" * 50)
print("  STAGE C: Sequence Length Search")
print("-" * 50)

stage_c_results = []
total_c = len(param_grid["seq_len"])

for idx, seq_len in enumerate(param_grid["seq_len"], 1):
    # Clip sequences to requested length from the end of the time axis
    actual_seq = min(seq_len, X.shape[1])
    X_tr_s = X_train[:, :actual_seq, :]
    X_te_s = X_test[:,  :actual_seq, :]

    print(f"\n  [{idx}/{total_c}] seq_len={seq_len}  "
          f"(using first {actual_seq} steps, "
          f"layers={best_params['num_layers']}, "
          f"units={best_params['hidden_units']}, "
          f"batch={best_params['batch_size']}, "
          f"lr={best_params['learning_rate']})")

    model = build_model(best_params["num_layers"],
                        best_params["hidden_units"],
                        best_params["learning_rate"],
                        actual_seq)
    model.fit(
        X_tr_s, Y_train,
        epochs=100,
        batch_size=best_params["batch_size"],
        validation_split=0.1,
        callbacks=[ES],
        verbose=0,
    )
    mae, rmse, r2 = evaluate(model, X_te_s, Y_test, scaler_Y)
    print(f"    → MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")

    row = dict(stage="C",
               num_layers=best_params["num_layers"],
               hidden_units=best_params["hidden_units"],
               batch_size=best_params["batch_size"],
               learning_rate=best_params["learning_rate"],
               seq_len=actual_seq, mae=mae, rmse=rmse, r2=r2)
    stage_c_results.append(row)
    tuning_results.append(row)

best_c = max(stage_c_results, key=lambda x: x["r2"])
best_params["seq_len"] = best_c["seq_len"]
print(f"\n  ✔ Best seq_len: {best_params['seq_len']}  "
      f"(R²={best_c['r2']:.4f})")

# ──────────────────────────────────────────────────────
# BEST MODEL — retrain with all best parameters
# ──────────────────────────────────────────────────────
print("\n" + "-" * 50)
print("  BEST MODEL — Final Training")
print("-" * 50)
print(f"  num_layers   : {best_params['num_layers']}")
print(f"  hidden_units : {best_params['hidden_units']}")
print(f"  batch_size   : {best_params['batch_size']}")
print(f"  learning_rate: {best_params['learning_rate']}")
print(f"  seq_len      : {best_params['seq_len']}")

best_seq = best_params["seq_len"]
X_tr_best = X_train[:, :best_seq, :]
X_te_best = X_test[:,  :best_seq, :]

best_model = build_model(
    best_params["num_layers"],
    best_params["hidden_units"],
    best_params["learning_rate"],
    best_seq,
)
es_best = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
history_best = best_model.fit(
    X_tr_best, Y_train,
    epochs=150,
    batch_size=best_params["batch_size"],
    validation_split=0.1,
    callbacks=[es_best],
    verbose=0,
)
mae_best, rmse_best, r2_best = evaluate(best_model, X_te_best, Y_test, scaler_Y)
print(f"\n  Best Model → MAE: ${mae_best:.2f} | "
      f"RMSE: ${rmse_best:.2f} | R²: {r2_best:.4f}")

best_model.save("../models/best_hypertuned_model.keras")
print("  Saved: ../models/best_hypertuned_model.keras")

# ──────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  HYPERPARAMETER TUNING SUMMARY")
print("=" * 60)

print(f"\n  {'Parameter':<20} {'Best Value':>15}")
print(f"  {'-'*37}")
for k, v in best_params.items():
    print(f"  {k:<20} {str(v):>15}")

print(f"\n  {'Metric':<20} {'Best Model':>15}")
print(f"  {'-'*37}")
print(f"  {'MAE':<20} ${mae_best:>13.2f}")
print(f"  {'RMSE':<20} ${rmse_best:>13.2f}")
print(f"  {'R²':<20} {r2_best:>15.4f}")

# ──────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────

# ── Plot 1: Stage A heatmap – R² by layers × units ──
a_df = pd.DataFrame(stage_a_results)
pivot_a = a_df.pivot(index="num_layers", columns="hidden_units", values="r2")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Hyperparameter Tuning — AAPL Price Prediction", fontsize=14)

im0 = axes[0].imshow(pivot_a.values, cmap="YlGn", aspect="auto")
axes[0].set_xticks(range(len(pivot_a.columns)))
axes[0].set_yticks(range(len(pivot_a.index)))
axes[0].set_xticklabels(pivot_a.columns)
axes[0].set_yticklabels(pivot_a.index)
axes[0].set_xlabel("Hidden Units")
axes[0].set_ylabel("Num Layers")
axes[0].set_title("Stage A: Architecture\n(R² score)")
for i in range(len(pivot_a.index)):
    for j in range(len(pivot_a.columns)):
        axes[0].text(j, i, f"{pivot_a.values[i, j]:.3f}",
                     ha="center", va="center", fontsize=9)
plt.colorbar(im0, ax=axes[0])

# ── Plot 2: Stage B heatmap – R² by batch × LR ──
b_df = pd.DataFrame(stage_b_results)
pivot_b = b_df.pivot(index="batch_size", columns="learning_rate", values="r2")

im1 = axes[1].imshow(pivot_b.values, cmap="YlOrRd", aspect="auto")
axes[1].set_xticks(range(len(pivot_b.columns)))
axes[1].set_yticks(range(len(pivot_b.index)))
axes[1].set_xticklabels([str(c) for c in pivot_b.columns])
axes[1].set_yticklabels(pivot_b.index)
axes[1].set_xlabel("Learning Rate")
axes[1].set_ylabel("Batch Size")
axes[1].set_title("Stage B: Training Params\n(R² score)")
for i in range(len(pivot_b.index)):
    for j in range(len(pivot_b.columns)):
        axes[1].text(j, i, f"{pivot_b.values[i, j]:.3f}",
                     ha="center", va="center", fontsize=9)
plt.colorbar(im1, ax=axes[1])

# ── Plot 3: Stage C bar – R² by seq_len ──
c_df = pd.DataFrame(stage_c_results)
axes[2].bar([str(s) for s in c_df["seq_len"]], c_df["r2"],
            color=["#4C72B0", "#DD8452", "#55A868"])
axes[2].set_xlabel("Sequence Length")
axes[2].set_ylabel("R²")
axes[2].set_title("Stage C: Sequence Length\n(R² score)")
axes[2].grid(True, alpha=0.3)
for i, (seq, r2) in enumerate(zip(c_df["seq_len"], c_df["r2"])):
    axes[2].text(i, r2 + 0.001, f"{r2:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("../plots/hyperparameter_tuning.png", dpi=150)
plt.show()
print("\nSaved: plots/hyperparameter_tuning.png")

# ── Plot 4: Best model training curve ──
fig2, ax = plt.subplots(figsize=(10, 5))
ax.plot(history_best.history["loss"],     label="Train Loss", color="blue")
ax.plot(history_best.history["val_loss"], label="Val Loss",   color="blue",
        linestyle="--")
ax.set_title(f"Best Model Training Curve\n"
             f"layers={best_params['num_layers']}, "
             f"units={best_params['hidden_units']}, "
             f"batch={best_params['batch_size']}, "
             f"lr={best_params['learning_rate']}, "
             f"seq={best_params['seq_len']}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../plots/best_model_training_curve.png", dpi=150)
plt.show()
print("Saved: plots/best_model_training_curve.png")

# ──────────────────────────────────────────────────────
# SAVE RESULTS TO CSV
# ──────────────────────────────────────────────────────
results_df = pd.DataFrame(tuning_results)
results_df.to_csv("../models/hyperparameter_tuning_results.csv", index=False)
print("Saved: ../models/hyperparameter_tuning_results.csv")

best_summary_df = pd.DataFrame([{
    "num_layers"   : best_params["num_layers"],
    "hidden_units" : best_params["hidden_units"],
    "batch_size"   : best_params["batch_size"],
    "learning_rate": best_params["learning_rate"],
    "seq_len"      : best_params["seq_len"],
    "MAE"          : round(mae_best, 4),
    "RMSE"         : round(rmse_best, 4),
    "R2"           : round(r2_best, 4),
}])
best_summary_df.to_csv("../models/best_hyperparameters.csv", index=False)
print("Saved: ../models/best_hyperparameters.csv")

# ── Final console summary ──
print("\n" + "=" * 60)
print("  20. HYPERPARAMETER TUNING — COMPLETE")
print("=" * 60)
print(f"\n  Best Parameters:")
for k, v in best_params.items():
    print(f"    {k:<20}: {v}")
print(f"\n  Best Model Performance:")
print(f"    MAE  : ${mae_best:.2f}")
print(f"    RMSE : ${rmse_best:.2f}")
print(f"    R²   : {r2_best:.4f}")