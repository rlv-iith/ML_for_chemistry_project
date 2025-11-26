import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Save plots to file
import matplotlib.pyplot as plt
import seaborn as sns

filename = "battery_data_summary.csv"

# 1. Load Data
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print("CSV not found.")
    exit()

print(f"Analyzing {df['cell_id'].nunique()} cells in Batch 1...")

# 2. Extract Multiple Features
stats = []
start = 10
end = 100

for cell_id, cell in df.groupby("cell_id"):
    cell = cell.sort_values("cycle")
    max_life = cell['cycle'].max()
    
    if max_life < 110: continue

    try:
        # Get slice 10-100
        sub = cell[(cell['cycle'] >= start) & (cell['cycle'] <= end)]
        
        # Raw Values
        q_start = cell.loc[cell['cycle'] == start, 'QDischarge'].values[0]
        q_end = cell.loc[cell['cycle'] == end, 'QDischarge'].values[0]
        ir_end = cell.loc[cell['cycle'] == end, 'IR'].values[0]
        
        # Feature 1: Log Delta Q (The standard)
        f1_log_delta = np.log10(abs(q_start - q_end))
        
        # Feature 2: Slope (Polyfit) - Smoothes out noise
        if len(sub) > 5:
            slope = np.polyfit(sub['cycle'], sub['QDischarge'], 1)[0]
            f2_log_slope = np.log10(abs(slope))
        else:
            f2_log_slope = np.nan
            
        # Feature 3: Internal Resistance (Often cleaner than capacity in Batch 1)
        f3_ir = ir_end
        
        # Feature 4: Initial Capacity (Sometimes simply starting lower = dying faster)
        f4_init_q = q_start
        
        # Target
        target_life = np.log10(max_life)
        
        stats.append({
            'Log_Delta_Q': f1_log_delta,
            'Log_Slope': f2_log_slope,
            'IR_100': f3_ir,
            'Initial_Q': f4_init_q,
            'Log_Life': target_life
        })
        
    except (IndexError, ValueError):
        continue

# 3. Correlation Matrix
corr_df = pd.DataFrame(stats)
corr_df = corr_df.dropna() # Remove any failed calcs

# Calculate correlation with Life
correlations = corr_df.corr()['Log_Life'].sort_values()

print("\n--- WHICH FEATURE WORKS BEST FOR THIS BATCH? ---")
print(correlations.drop('Log_Life')) # Don't show correlation with itself
print("------------------------------------------------")

# 4. Interpret
best_feature = correlations.drop('Log_Life').abs().idxmax()
best_score = correlations[best_feature]

print(f"\nWINNER: '{best_feature}' with score {best_score:.4f}")
print("Hint: If score is negative, that's GOOD (Inverse relationship).")
print("Target is closer to -1.0 or +1.0")

# 5. Plot the Winner
plt.figure(figsize=(8,6))
sns.regplot(data=corr_df, x=best_feature, y='Log_Life', color='teal')
plt.title(f"Best Feature for Batch 1: {best_feature}\nR={best_score:.3f}")
plt.grid(True, alpha=0.3)
plt.savefig("feature_hunt_result.png")
print("\nPlot saved to 'feature_hunt_result.png'")