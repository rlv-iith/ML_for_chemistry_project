import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Save to file
import matplotlib.pyplot as plt
import seaborn as sns

# Load your existing CSV
df = pd.read_csv("battery_data_summary.csv")

data_points = []
start_cycle = 10
end_cycle = 100

for cell_id, cell_data in df.groupby("cell_id"):
    cell_data = cell_data.sort_values("cycle")
    max_life = cell_data['cycle'].max()
    
    # Must live past 100 to be useful
    if max_life < 110: 
        continue

    try:
        q_10 = cell_data.loc[cell_data['cycle'] == start_cycle, 'QDischarge'].values[0]
        q_100 = cell_data.loc[cell_data['cycle'] == end_cycle, 'QDischarge'].values[0]
        
        # KEY FEATURE: Log of Capacity Loss
        delta_q = np.log10(abs(q_100 - q_10))
        log_life = np.log10(max_life)
        
        data_points.append({'cell_id': cell_id, 
                            'Log_Delta_Q': delta_q, 
                            'Log_Cycle_Life': log_life,
                            'Actual_Cycles': max_life})
    except:
        continue

corr_df = pd.DataFrame(data_points)

# Calculate Pearson Correlation (Standard metric)
correlation = corr_df['Log_Delta_Q'].corr(corr_df['Log_Cycle_Life'])
print(f"=======================================")
print(f"PHYISCS CHECK (Correlation Analysis)")
print(f"Number of valid cells: {len(corr_df)}")
print(f"Correlation (Feature vs Life): {correlation:.4f}")
print(f"=======================================")
if correlation < -0.5:
    print("GOOD NEWS: Strong negative correlation found.")
    print("This means the feature WORKS, you just need more data.")
else:
    print("WARNING: Correlation is weak. Data might be noisy.")

# PLOT
plt.figure(figsize=(8,6))
sns.regplot(data=corr_df, x='Log_Delta_Q', y='Log_Cycle_Life', color='purple')
plt.title(f"Does Capacity Fade Predict Life?\nCorrelation: {correlation:.3f}", fontsize=14)
plt.xlabel("Log10(Capacity Loss Q100-Q10)")
plt.ylabel("Log10(Total Cycle Life)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("physics_check.png")
print("Plot saved to 'physics_check.png'")