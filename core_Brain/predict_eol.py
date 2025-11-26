# 1. FIX PLOTTING CRASH (Must be first)
import matplotlib
matplotlib.use('Agg') # Save to file instead of trying to open a window

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "battery_data_combined.csv"
START_CYCLE = 10
MID_CYCLE = 55
END_CYCLE = 100
# ==========================================

print(f"--- Battery EOL Prediction (Severson et al. Style) ---")

# 1. LOAD DATA
try:
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: Could not find '{CSV_FILE}'. Run the converter script first.")
    exit()

# 2. FEATURE ENGINEERING
print("Extracting features (Cycles 10-100)...")

features = []
targets = []
cell_ids = []

for cell_id, cell_data in df.groupby("cell_id"):
    # Ensure data is sorted by cycle
    cell_data = cell_data.sort_values("cycle")
    
    # Filter: Battery must have lived at least past END_CYCLE
    max_life = cell_data['cycle'].max()
    if max_life < END_CYCLE:
        continue

    try:
        # A. EXTRACT RAW VALUES
        q_10 = cell_data.loc[cell_data['cycle'] == START_CYCLE, 'QDischarge'].values[0]
        q_100 = cell_data.loc[cell_data['cycle'] == END_CYCLE, 'QDischarge'].values[0]
        ir_100 = cell_data.loc[cell_data['cycle'] == END_CYCLE, 'IR'].values[0]
        
        # B. CALCULATE ADVANCED FEATURES
        
        # Feature 1: Log of Capacity Fade (Best predictor for small data)
        # We take Abs() because Q decreases, so diff is negative.
        # Log scaling helps linearize the relationship with Life.
        delta_q = q_100 - q_10
        feat_log_delta = np.log10(abs(delta_q))

        # Feature 2: Curvature (Is the fade accelerating?)
        # Compare actual Q at cycle 55 vs. the "expected" linear value
        try:
            q_55_real = cell_data.loc[cell_data['cycle'] == MID_CYCLE, 'QDischarge'].values[0]
            q_55_linear = (q_10 + q_100) / 2
            feat_curvature = q_55_real - q_55_linear
        except IndexError:
            # Fallback if cycle 55 is missing in data stream
            feat_curvature = 0

        # Feature 3: Internal Resistance at cycle 100
        feat_ir = ir_100

        # C. DEFINE TARGET
        # We predict Log10(Life) because life varies exponentially
        target_log_life = np.log10(max_life)

        features.append([feat_log_delta, feat_curvature, feat_ir])
        targets.append(target_log_life)
        cell_ids.append(cell_id)

    except (IndexError, ValueError) as e:
        # Skip cells that have missing data at key cycles
        continue

# Create Arrays
feature_names = ['Log(Delta_Q)', 'Curvature', 'IR_100']
X = pd.DataFrame(features, columns=feature_names)
y = np.array(targets)

print(f"Features ready for {len(X)} cells.")

# 3. SPLIT & SCALE (CRITICAL STEP)
# Random State 42 guarantees reproducible splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler transforms data to Mean=0, Std=1
# This fixes the "Zero Weights" issue
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAIN MODEL (ElasticNetCV)
# Automatically finds best Alpha/L1 Ratio
print("Training Model...")
model = ElasticNetCV(cv=5, random_state=42, max_iter=2000)
model.fit(X_train_scaled, y_train)

# 5. PREDICT & EVALUATE
preds_log = model.predict(X_test_scaled)

# Convert Log predictions back to Cycles
preds_cycles = 10**preds_log
y_test_cycles = 10**y_test

rmse = np.sqrt(mean_squared_error(y_test_cycles, preds_cycles))
r2 = r2_score(y_test, preds_log) # Measure R2 on the Log scale (standard)

print("\n==============================")
print("       FINAL RESULTS")
print("==============================")
print(f"Test Set Size: {len(y_test)} cells")
print(f"RMSE (Error):  {rmse:.2f} cycles")
print(f"R^2 Score:     {r2:.4f}")
print("------------------------------")
print("Feature Coefficients (Importance):")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name:15s}: {coef:.5f}")

# 6. SAVE PLOT
plt.figure(figsize=(9, 7))
plt.scatter(y_test_cycles, preds_cycles, color='#2c3e50', alpha=0.8, s=100, label='Test Data')

# Perfect Line
min_val = min(y_test_cycles.min(), preds_cycles.min()) * 0.9
max_val = max(y_test_cycles.max(), preds_cycles.max()) * 1.1
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')

plt.title(f"Battery EOL Prediction (Cycles 10-100)\nModel: ElasticNet | R²: {r2:.3f} | RMSE: {int(rmse)}", fontsize=14)
plt.xlabel("Actual Cycle Life", fontsize=12)
plt.ylabel("Predicted Cycle Life", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

output_img = "final_eol_prediction.png"
plt.savefig(output_img)
print(f"\n[Done] Visualization saved to '{output_img}'")