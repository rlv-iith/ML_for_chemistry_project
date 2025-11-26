# ==============================================================================
# BATTERY CYCLE LIFE PREDICTION (Severson et al. Reproduced - Optimized)
# ==============================================================================
import matplotlib
matplotlib.use('Agg')  # Fixes "Tcl/Tk" crash on Windows by saving to file only

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
CSV_FILE = "battery_data_combined.csv"  # The combined file you created
START_CYCLE = 10
MID_CYCLE = 55
END_CYCLE = 100
SMOOTH_WINDOW = 9  # Averages 9 cycles to remove sensor noise

print(f"--- Battery Life Prediction (Smoothing Window: {SMOOTH_WINDOW}) ---")

# 1. LOAD DATA
try:
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} rows. Processing features...")
except FileNotFoundError:
    print(f"CRITICAL ERROR: '{CSV_FILE}' not found.")
    print("Please run the 'process_all_batches.py' script first to merge the files.")
    exit()

# 2. FEATURE EXTRACTION (With Smoothing)
features = []
targets = []
cell_ids = []

# Group by battery to process one timeline at a time
for cell_id, cell_data in df.groupby("cell_id"):
    cell_data = cell_data.sort_values("cycle")
    
    # 2a. Filter short-lived batteries
    # If it died before cycle 105, we can't measure cycle 100 accurately
    max_life = cell_data['cycle'].max()
    if max_life < END_CYCLE + 5:
        continue

    try:
        # 2b. APPLY SMOOTHING (The Magic Step)
        # We use a rolling mean to wash out the jagged sensor noise
        cell_data['Q_Smooth'] = cell_data['QDischarge'].rolling(
            window=SMOOTH_WINDOW, center=True, min_periods=1
        ).mean()

        # 2c. EXTRACT POINTS
        q_10 = cell_data.loc[cell_data['cycle'] == START_CYCLE, 'Q_Smooth'].values[0]
        q_55 = cell_data.loc[cell_data['cycle'] == MID_CYCLE, 'Q_Smooth'].values[0]
        q_100 = cell_data.loc[cell_data['cycle'] == END_CYCLE, 'Q_Smooth'].values[0]
        ir_100 = cell_data.loc[cell_data['cycle'] == END_CYCLE, 'IR'].values[0]

        # 2d. COMPUTE FEATURES
        
        # Feature 1: Log10(Capacity Loss)
        # Log-space is crucial because capacity fade follows a power law
        loss = abs(q_100 - q_10)
        if loss == 0: loss = 1e-6 # Avoid log(0) error
        f_log_delta_q = np.log10(loss)

        # Feature 2: Curvature
        # Difference between actual mid-point and a linear straight line
        # This tells us if degradation is accelerating or constant
        linear_projection = (q_10 + q_100) / 2
        f_curvature = q_55 - linear_projection

        # Feature 3: Internal Resistance (Log scaled)
        f_log_ir = np.log10(ir_100)

        # 2e. TARGET
        # We predict Log10(Life) to keep error percentage balanced
        target = np.log10(max_life)

        features.append([f_log_delta_q, f_curvature, f_log_ir])
        targets.append(target)
        cell_ids.append(cell_id)

    except (IndexError, ValueError):
        # Skip cells with gaps in data
        continue

# Convert to ML-ready arrays
feature_names = ['Log(Delta_Q)', 'Curvature', 'Log(IR_100)']
X = pd.DataFrame(features, columns=feature_names)
y = np.array(targets)

print(f"Extracted valid features for {len(X)} cells.")

# 3. SPLIT AND SCALE
# Split 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler (Vital for ElasticNet to weight features fairly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAIN MODEL (ElasticNetCV)
# Auto-tunes regularization (alpha/l1) to find the sweet spot
print("Training AI Model...")
model = ElasticNetCV(cv=5, random_state=42, max_iter=5000)
model.fit(X_train_scaled, y_train)

# 5. PREDICT
preds_log = model.predict(X_test_scaled)

# Convert back from Log-Space to Real Cycles
preds_cycles = 10**preds_log
y_test_cycles = 10**y_test

# 6. EVALUATE
rmse = np.sqrt(mean_squared_error(y_test_cycles, preds_cycles))
r2 = r2_score(y_test, preds_log) # Scored on Log Scale

print("\n" + "="*30)
print(f" FINAL RESULTS (Smooth Window {SMOOTH_WINDOW})")
print("="*30)
print(f"RMSE:  {rmse:.2f} cycles")
print(f"R^2:   {r2:.4f}  (> 0.6 is good for Summary Data)")
print("-" * 30)
print("Feature Coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name:15s}: {coef:.5f}")
print("="*30)

# 7. SAVE PLOT
plt.figure(figsize=(9, 7))
# Scatter Plot
plt.scatter(y_test_cycles, preds_cycles, color='#2980b9', alpha=0.8, s=80, edgecolors='k', label='Predicted Batteries')

# Perfect Line
min_val = min(y_test_cycles.min(), preds_cycles.min()) * 0.9
max_val = max(y_test_cycles.max(), preds_cycles.max()) * 1.1
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Fit')

plt.xlabel("Actual Cycle Life", fontsize=12)
plt.ylabel("Predicted Cycle Life", fontsize=12)
plt.title(f"Battery Life Prediction (Using Cycles 10-100)\nModel R²: {r2:.3f}", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

filename = "final_results_smoothed.png"
plt.savefig(filename)
print(f"\n[Done] Results graph saved to '{filename}'")

# ==============================================================================
# 7. PROFESSIONAL VISUALIZATION (Seaborn Style)
# ==============================================================================
import seaborn as sns

# Set a professional style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2) # Make text readable for reports

# --- PLOT 1: Actual vs Predicted (The "Money Shot") ---
plt.figure(figsize=(10, 8))

# Color points by how accurate they are (Error magnitude)
errors = np.abs(y_test_cycles - preds_cycles)
sc = plt.scatter(y_test_cycles, preds_cycles, c=errors, cmap='viridis_r', 
                 alpha=0.8, s=100, edgecolors='w', linewidth=0.5)

# The Perfect Fit Line
min_val = 200 # Set lower bound for neatness
max_val = 2300
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')

# Labels and Titles
plt.colorbar(sc, label='Absolute Error (Cycles)')
plt.xlabel("Actual Cycle Life (Observed)", fontweight='bold')
plt.ylabel("Predicted Cycle Life (AI Model)", fontweight='bold')
plt.title(f"Battery Cycle Life Prediction (Cycles 10-100)\nModel: ElasticNet | $R^2={r2:.3f}$", fontsize=16)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.legend()

# Add a text box with stats
stats_text = (f"RMSE: {int(rmse)} cycles\n"
              f"$R^2$: {r2:.3f}\n"
              f"Test Size: {len(y_test)}")
plt.text(max_val*0.05 + min_val, max_val*0.9, stats_text, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.savefig("1_prediction_accuracy.png", dpi=300) # dpi=300 is High Res
print("\n[GRAPH] Saved '1_prediction_accuracy.png' (High Res)")


# --- PLOT 2: Residual Analysis (Where is the model wrong?) ---
# Ideally, errors should be random (centered on 0).
# If this plot shows a pattern (curve), the model is missing physics.
residuals = preds_cycles - y_test_cycles

plt.figure(figsize=(10, 6))
plt.axhline(0, color='black', lw=2, linestyle='--') # Zero line
sns.scatterplot(x=y_test_cycles, y=residuals, color='#c0392b', s=100, alpha=0.7)

plt.xlabel("Actual Battery Life (Cycles)")
plt.ylabel("Prediction Error (Predicted - Actual)")
plt.title("Residual Plot: Where does the model fail?", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)

# Highlight bounds
limit = np.max(np.abs(residuals)) * 1.1
plt.ylim(-limit, limit)

plt.tight_layout()
plt.savefig("2_residuals.png", dpi=300)
print("[GRAPH] Saved '2_residuals.png' (Error Analysis)")


# --- PLOT 3: Feature Importance (Which Physics Matter?) ---
plt.figure(figsize=(10, 5))

# Create a clean dataframe for plotting
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
})

# Sort by magnitude
coef_df['Abs_Val'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('Abs_Val', ascending=False)

# Color bar: Red for Negative correlation (kills battery), Green for Positive
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in coef_df['Coefficient']]

sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette=colors)

plt.axvline(0, color='black', linewidth=1)
plt.title("Feature Importance: What drives battery degradation?", fontsize=16)
plt.xlabel("Coefficient Strength (Normalized Impact)")
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("3_feature_importance.png", dpi=300)
print("[GRAPH] Saved '3_feature_importance.png' (Interpretation)")

print("\n--- All visualizations completed successfully. ---")