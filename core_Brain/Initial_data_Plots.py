import matplotlib
matplotlib.use('Agg') # Save to file (prevents crash on Windows)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Handling the Colormap deprecation warning safely
try:
    from matplotlib import colormaps
    get_cmap = colormaps.get_cmap
except ImportError:
    import matplotlib.cm as cm
    get_cmap = cm.get_cmap

# Load Data
filename = "battery_data_combined.csv"
try:
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print("CRITICAL: Please run process_all_batches.py first!")
    exit()

# ==========================================================
# PART 1: REPLICATING FIGURE 1a (Discharge Capacity Curves)
# ==========================================================
print("Generating Figure 1a (Capacity Fade Curves)...")

plt.figure(figsize=(10, 7))

# 1. Setup Colors (Red = Short Life, Blue = Long Life)
grouped = df.groupby('cell_id')
max_lives = grouped['cycle'].max()
min_life = max_lives.min()
max_life = max_lives.max()

# Normalization & Colormap
norm = mcolors.Normalize(vmin=min_life, vmax=max_life)
cmap = get_cmap('coolwarm_r') # _r reverses it so Red=Short, Blue=Long

# 2. Plot every cell
count = 0
for cell_id, cell_data in grouped:
    cell_data = cell_data.sort_values("cycle")
    life = cell_data['cycle'].max()
    
    # Apply smoothing (window=15) for professional "smooth" lines like the paper
    y_smooth = cell_data['QDischarge'].rolling(window=15, center=True).mean()
    
    color = cmap(norm(life))
    plt.plot(cell_data['cycle'], y_smooth, color=color, linewidth=1.5, alpha=0.8)
    count += 1

print(f"Plotted traces for {count} cells.")

# 3. Aesthetics
plt.title("Figure 1a Replica: Discharge Capacity vs Cycle Number", fontsize=14, fontweight='bold')
plt.xlabel("Cycle Number", fontsize=12)
plt.ylabel("Discharge Capacity (Ah)", fontsize=12)
plt.xlim(0, 2000)
plt.ylim(0.85, 1.15)
plt.grid(False) # Clean background style

# FIX 1: Explicitly pass the current axes (ax) to colorbar to prevent the ValueError
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca()) # <--- FIX IS HERE
cbar.set_label('Total Cycle Life (Count)', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig("Nature_Fig1a_Replica.png", dpi=300)
print("Saved 'Nature_Fig1a_Replica.png'")


# ==========================================================
# PART 2: REPLICATING FIGURE 1c/d (Log-Log Correlation)
# ==========================================================
print("Generating Figure 1c Replica (Correlation Scatter)...")

features = []

for cell_id, cell_data in grouped:
    cell_data = cell_data.sort_values("cycle")
    life = cell_data['cycle'].max()
    if life < 105: continue # Must survive past prediction point (Cycle 100)

    try:
        # We use a slight rolling window to remove noise for the point calculation
        cell_data['Q_Smooth'] = cell_data['QDischarge'].rolling(window=9, center=True).mean()
        
        # Get smoothed values at cycle 10 and 100
        q10 = cell_data.loc[cell_data['cycle'] == 10, 'Q_Smooth'].values[0]
        q100 = cell_data.loc[cell_data['cycle'] == 100, 'Q_Smooth'].values[0]
        
        # Calculate Log Delta Q
        # We take abs() and log10
        delta_q_val = abs(q100 - q10)
        if delta_q_val == 0: continue
            
        log_delta_q = np.log10(delta_q_val)
        
        features.append({
            'log_delta_q': log_delta_q,
            'life': life,
            'log_life': np.log10(life)
        })
    except IndexError:
        continue

feat_df = pd.DataFrame(features)

plt.figure(figsize=(7, 7))

# Scatter plot with same coloring
plt.scatter(feat_df['log_delta_q'], feat_df['log_life'], 
            c=feat_df['life'], cmap='coolwarm_r', 
            norm=norm, s=60, alpha=0.9, edgecolors='grey')

# FIX 2: Used r"..." (Raw Strings) to handle mathematical symbols correctly
plt.title(r"Figure 1c/d Replica: Cycle Life vs $\Delta Q_{100-10}$", fontsize=14)
plt.xlabel(r"Log($\Delta$Discharge Capacity)", fontsize=12)
plt.ylabel("Log(Cycle Life)", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.3)

# Calculate Correlation
r_val = feat_df['log_delta_q'].corr(feat_df['log_life'])

# Dynamically place the text label to avoid overlap
x_pos = feat_df['log_delta_q'].min()
y_pos = feat_df['log_life'].min() + 0.1

plt.text(x_pos, y_pos, 
         fr"Correlation $\rho = {r_val:.2f}$", fontsize=14, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.savefig("Nature_Fig1c_Replica.png", dpi=300)
print("Saved 'Nature_Fig1c_Replica.png'")