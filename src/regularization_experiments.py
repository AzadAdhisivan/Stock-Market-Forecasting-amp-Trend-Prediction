import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Flatten,
                                     BatchNormalization, Input)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# ── Load AAPL data for all experiments ──
X = np.load("../data/sequences/AAPL_X.npy")
Y = np.load("../data/sequences/AAPL_Y.npy")
scaler_Y = joblib.load("../data/scalers/AAPL_scaler_Y.pkl")

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

os.makedirs("../plots", exist_ok=True)
os.makedirs("../models", exist_ok=True)

def evaluate(model, X_test, Y_test, scaler_Y):
    Y_pred = model.predict(X_test, verbose=0).flatten()
    Y_test_real = scaler_Y.inverse_transform(
        Y_test.reshape(-1, 1)).flatten()
    Y_pred_real = scaler_Y.inverse_transform(
        Y_pred.reshape(-1, 1)).flatten()
    mae  = mean_absolute_error(Y_test_real, Y_pred_real)
    rmse = np.sqrt(mean_squared_error(Y_test_real, Y_pred_real))
    r2   = r2_score(Y_test_real, Y_pred_real)
    return mae, rmse, r2

# ══════════════════════════════════════════════════════
# 13. REGULARIZATION TECHNIQUES
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  13. REGULARIZATION TECHNIQUES")
print("="*60)

reg_results = {}

# ── Model 1: No Regularization (Baseline) ──
print("\n[1/5] Training Baseline (No Regularization)...")
baseline = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])
baseline.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
h1 = baseline.fit(X_train, Y_train, epochs=100, batch_size=32,
                   validation_split=0.1, verbose=0)
mae, rmse, r2 = evaluate(baseline, X_test, Y_test, scaler_Y)
reg_results["Baseline"] = {"mae": mae, "rmse": rmse, "r2": r2,
                            "val_loss": h1.history["val_loss"]}
print(f"  MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")

# ── Model 2: Dropout Only ──
print("\n[2/5] Training with Dropout Only...")
dropout_model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])
dropout_model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
h2 = dropout_model.fit(X_train, Y_train, epochs=100, batch_size=32,
                        validation_split=0.1, verbose=0)
mae, rmse, r2 = evaluate(dropout_model, X_test, Y_test, scaler_Y)
reg_results["Dropout"] = {"mae": mae, "rmse": rmse, "r2": r2,
                           "val_loss": h2.history["val_loss"]}
print(f"  MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")

# ── Model 3: Early Stopping Only ──
print("\n[3/5] Training with Early Stopping Only...")
early_model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])
early_model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
es = EarlyStopping(monitor="val_loss", patience=10,
                   restore_best_weights=True)
h3 = early_model.fit(X_train, Y_train, epochs=100, batch_size=32,
                      validation_split=0.1, callbacks=[es], verbose=0)
mae, rmse, r2 = evaluate(early_model, X_test, Y_test, scaler_Y)
reg_results["Early Stopping"] = {"mae": mae, "rmse": rmse, "r2": r2,
                                  "val_loss": h3.history["val_loss"]}
print(f"  MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")

# ── Model 4: Batch Normalization Only ──
print("\n[4/5] Training with Batch Normalization Only...")
bn_model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Flatten(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dense(1, activation="linear")
])
bn_model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
h4 = bn_model.fit(X_train, Y_train, epochs=100, batch_size=32,
                   validation_split=0.1, verbose=0)
mae, rmse, r2 = evaluate(bn_model, X_test, Y_test, scaler_Y)
reg_results["Batch Norm"] = {"mae": mae, "rmse": rmse, "r2": r2,
                              "val_loss": h4.history["val_loss"]}
print(f"  MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")

# ── Model 5: All Combined ──
print("\n[5/5] Training with All Regularization Combined...")
combined_model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Flatten(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dense(1, activation="linear")
])
combined_model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
es2 = EarlyStopping(monitor="val_loss", patience=10,
                    restore_best_weights=True)
h5 = combined_model.fit(X_train, Y_train, epochs=100, batch_size=32,
                         validation_split=0.1, callbacks=[es2], verbose=0)
mae, rmse, r2 = evaluate(combined_model, X_test, Y_test, scaler_Y)
reg_results["All Combined"] = {"mae": mae, "rmse": rmse, "r2": r2,
                                "val_loss": h5.history["val_loss"]}
print(f"  MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")

# ── Regularization Summary ──
print("\n" + "="*60)
print("  REGULARIZATION COMPARISON SUMMARY")
print("="*60)
print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print(f"  {'-'*46}")
for name, res in reg_results.items():
    print(f"  {name:<20} ${res['mae']:>6.2f} ${res['rmse']:>6.2f} "
          f"{res['r2']:>8.4f}")

# ── Plot Regularization Comparison ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Regularization Techniques Comparison", fontsize=14)

colors = ["red", "blue", "green", "orange", "purple"]
names  = list(reg_results.keys())

for i, (name, res) in enumerate(reg_results.items()):
    axes[0].plot(res["val_loss"], label=name, color=colors[i])

axes[0].set_title("Validation Loss Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Val Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

metrics = ["mae", "rmse", "r2"]
x = np.arange(len(names))
width = 0.25

for i, metric in enumerate(metrics):
    values = [reg_results[n][metric] for n in names]
    axes[1].bar(x + i * width, values, width, label=metric.upper())

axes[1].set_title("MAE / RMSE / R² Comparison")
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(names, rotation=15)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../plots/regularization_comparison.png")
plt.show()
print("\nSaved: plots/regularization_comparison.png")

# ══════════════════════════════════════════════════════
# 14. VANISHING GRADIENT PROBLEM
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  14. VANISHING GRADIENT PROBLEM")
print("="*60)

# ── Deep Sigmoid Network ──
print("\nTraining Deep Sigmoid Network (8 layers)...")
sigmoid_model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Flatten(),
    Dense(256, activation="sigmoid"),
    Dense(128, activation="sigmoid"),
    Dense(64,  activation="sigmoid"),
    Dense(32,  activation="sigmoid"),
    Dense(16,  activation="sigmoid"),
    Dense(8,   activation="sigmoid"),
    Dense(4,   activation="sigmoid"),
    Dense(1,   activation="linear")
])
sigmoid_model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
h_sigmoid = sigmoid_model.fit(X_train, Y_train, epochs=100, batch_size=32,
                               validation_split=0.1, verbose=1)

# ── Deep ReLU Network (for comparison) ──
print("\nTraining Deep ReLU Network (8 layers)...")
relu_model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64,  activation="relu"),
    Dense(32,  activation="relu"),
    Dense(16,  activation="relu"),
    Dense(8,   activation="relu"),
    Dense(4,   activation="relu"),
    Dense(1,   activation="linear")
])
relu_model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
h_relu = relu_model.fit(X_train, Y_train, epochs=100, batch_size=32,
                         validation_split=0.1, verbose=0)

# ── Extract Gradients ──
print("\nExtracting gradients from each layer...")

@tf.function
def get_gradients(model, X_batch, Y_batch):
    with tf.GradientTape() as tape:
        pred = model(X_batch, training=True)
        loss = tf.reduce_mean(tf.square(pred - Y_batch))
    return tape.gradient(loss, model.trainable_variables)

X_sample = tf.constant(X_train[:32], dtype=tf.float32)
Y_sample = tf.constant(Y_train[:32], dtype=tf.float32)

sigmoid_grads = get_gradients(sigmoid_model, X_sample, Y_sample)
relu_grads    = get_gradients(relu_model, X_sample, Y_sample)

sigmoid_grad_norms = [tf.norm(g).numpy() for g in sigmoid_grads if g is not None]
relu_grad_norms    = [tf.norm(g).numpy() for g in relu_grads if g is not None]

print("\n  Sigmoid Gradient Norms per layer:")
for i, g in enumerate(sigmoid_grad_norms):
    print(f"    Layer {i+1}: {g:.8f} {'← near zero! vanishing' if g < 0.001 else ''}")

print("\n  ReLU Gradient Norms per layer:")
for i, g in enumerate(relu_grad_norms):
    print(f"    Layer {i+1}: {g:.8f}")

# ── Evaluate both ──
mae_s, rmse_s, r2_s = evaluate(sigmoid_model, X_test, Y_test, scaler_Y)
mae_r, rmse_r, r2_r = evaluate(relu_model,    X_test, Y_test, scaler_Y)

print(f"\n  Sigmoid → MAE: ${mae_s:.2f} | R²: {r2_s:.4f}")
print(f"  ReLU    → MAE: ${mae_r:.2f} | R²: {r2_r:.4f}")

# ── Plot Vanishing Gradient ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Vanishing Gradient Problem — Sigmoid vs ReLU", fontsize=14)

axes[0].plot(h_sigmoid.history["loss"],     label="Sigmoid Train", color="red")
axes[0].plot(h_sigmoid.history["val_loss"], label="Sigmoid Val",   color="red",
             linestyle="--")
axes[0].plot(h_relu.history["loss"],        label="ReLU Train",    color="blue")
axes[0].plot(h_relu.history["val_loss"],    label="ReLU Val",      color="blue",
             linestyle="--")
axes[0].set_title("Training Loss Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(len(sigmoid_grad_norms)), sigmoid_grad_norms, color="red")
axes[1].set_title("Sigmoid Gradient Norms\n(near zero = vanishing)")
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Gradient Norm")
axes[1].grid(True, alpha=0.3)

axes[2].bar(range(len(relu_grad_norms)), relu_grad_norms, color="blue")
axes[2].set_title("ReLU Gradient Norms\n(healthy gradients)")
axes[2].set_xlabel("Layer")
axes[2].set_ylabel("Gradient Norm")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../plots/vanishing_gradient.png")
plt.show()
print("\nSaved: plots/vanishing_gradient.png")

# ── Document Findings ──
print("\n  VANISHING GRADIENT FINDINGS:")
print(f"  Sigmoid avg gradient : {np.mean(sigmoid_grad_norms):.8f}")
print(f"  ReLU avg gradient    : {np.mean(relu_grad_norms):.8f}")
print(f"  Sigmoid R²           : {r2_s:.4f} (slow/poor learning)")
print(f"  ReLU R²              : {r2_r:.4f} (fast/better learning)")

# ══════════════════════════════════════════════════════
# 15. EXPLODING GRADIENT PROBLEM
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  15. EXPLODING GRADIENT PROBLEM")
print("="*60)

# ── Deep Network WITHOUT Gradient Clipping ──
print("\nTraining Deep Network WITHOUT Gradient Clipping...")
explode_losses = []

exploding_model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Flatten(),
    Dense(512, activation="relu", kernel_initializer="glorot_uniform"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64,  activation="relu"),
    Dense(32,  activation="relu"),
    Dense(16,  activation="relu"),
    Dense(8,   activation="relu"),
    Dense(1,   activation="linear")
])

# High learning rate to trigger exploding gradients
exploding_model.compile(
    optimizer=Adam(learning_rate=0.1),  # ← very high LR
    loss="mse",
    metrics=["mae"]
)

h_explode = exploding_model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ── Deep Network WITH Gradient Clipping ──
print("\nTraining Deep Network WITH Gradient Clipping...")
clipped_model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Flatten(),
    Dense(512, activation="relu", kernel_initializer="glorot_uniform"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64,  activation="relu"),
    Dense(32,  activation="relu"),
    Dense(16,  activation="relu"),
    Dense(8,   activation="relu"),
    Dense(1,   activation="linear")
])

# Same high LR but with gradient clipping
clipped_model.compile(
    optimizer=Adam(learning_rate=0.1, clipnorm=1.0),  # ← clip gradients
    loss="mse",
    metrics=["mae"]
)

h_clipped = clipped_model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ── Evaluate both ──
mae_e, rmse_e, r2_e = evaluate(exploding_model, X_test, Y_test, scaler_Y)
mae_c, rmse_c, r2_c = evaluate(clipped_model,   X_test, Y_test, scaler_Y)

print(f"\n  Without Clipping → MAE: ${mae_e:.2f} | R²: {r2_e:.4f}")
print(f"  With Clipping    → MAE: ${mae_c:.2f} | R²: {r2_c:.4f}")

# ── Plot Exploding Gradient ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Exploding Gradient Problem — Before vs After Clipping",
             fontsize=14)

axes[0].plot(h_explode.history["loss"],     label="Train Loss", color="red")
axes[0].plot(h_explode.history["val_loss"], label="Val Loss",   color="red",
             linestyle="--")
axes[0].set_title("WITHOUT Gradient Clipping\n(unstable loss)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, min(max(h_explode.history["loss"]), 1.0)])

axes[1].plot(h_clipped.history["loss"],     label="Train Loss", color="blue")
axes[1].plot(h_clipped.history["val_loss"], label="Val Loss",   color="blue",
             linestyle="--")
axes[1].set_title("WITH Gradient Clipping\n(stable loss)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../plots/exploding_gradient.png")
plt.show()
print("\nSaved: plots/exploding_gradient.png")

# ── Document Findings ──
print("\n  EXPLODING GRADIENT FINDINGS:")
print(f"  Without clipping → R²: {r2_e:.4f} "
      f"({'unstable/NaN' if np.isnan(r2_e) else 'poor'})")
print(f"  With clipping    → R²: {r2_c:.4f} (stable)")

# ══════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  COMPLETE FINDINGS SUMMARY")
print("="*60)

print("\n  13. Regularization:")
for name, res in reg_results.items():
    print(f"    {name:<20} R²: {res['r2']:.4f}")

print(f"\n  14. Vanishing Gradient:")
print(f"    Sigmoid (8 layers) R²: {r2_s:.4f} ← slow learning")
print(f"    ReLU    (8 layers) R²: {r2_r:.4f} ← healthy gradients")

print(f"\n  15. Exploding Gradient:")
print(f"    Without clipping   R²: {r2_e:.4f} ← unstable")
print(f"    With clipping      R²: {r2_c:.4f} ← stable")

# ── Save All Results to CSV ──

# 13. Regularization CSV
reg_df = pd.DataFrame([
    {
        "Model": name,
        "MAE": res["mae"],
        "RMSE": res["rmse"],
        "R2": res["r2"]
    }
    for name, res in reg_results.items()
])
reg_df.to_csv("../models/regularization_results.csv", index=False)
print("Saved: ../models/regularization_results.csv")

# 14. Vanishing Gradient CSV
vanishing_df = pd.DataFrame({
    "Activation": ["Sigmoid", "ReLU"],
    "MAE":  [mae_s, mae_r],
    "RMSE": [rmse_s, rmse_r],
    "R2":   [r2_s, r2_r],
    "Avg_Gradient_Norm": [
        np.mean(sigmoid_grad_norms),
        np.mean(relu_grad_norms)
    ],
    "Min_Gradient_Norm": [
        np.min(sigmoid_grad_norms),
        np.min(relu_grad_norms)
    ],
    "Observation": [
        "Slow learning, gradients near zero",
        "Healthy gradients, faster learning"
    ]
})
vanishing_df.to_csv("../models/vanishing_gradient_results.csv", index=False)
print("Saved: ../models/vanishing_gradient_results.csv")

# 15. Exploding Gradient CSV
exploding_df = pd.DataFrame({
    "Model": ["Without Clipping", "With Clipping"],
    "MAE":  [mae_e, mae_c],
    "RMSE": [rmse_e, rmse_c],
    "R2":   [r2_e, r2_c],
    "Learning_Rate": [0.1, 0.1],
    "Clip_Norm": [None, 1.0],
    "Observation": [
        "Unstable loss, large weight updates",
        "Stable loss, controlled gradients"
    ]
})
exploding_df.to_csv("../models/exploding_gradient_results.csv", index=False)
print("Saved: ../models/exploding_gradient_results.csv")