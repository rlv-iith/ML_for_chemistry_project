import h5py
import pandas as pd
import numpy as np
import os

# The 3 famous batches from Severson et al. (Nature Energy, 2019)
filenames = [
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat", # Batch 1
    "2017-06-30_batchdata_updated_struct_errorcorrect.mat", # Batch 2
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat"  # Batch 3
]

output_csv = "battery_data_combined.csv"
all_data = []

print("--- Starting Bulk Processing ---")

for batch_idx, fname in enumerate(filenames):
    if not os.path.exists(fname):
        print(f"SKIPPING {fname} (File not found in folder)")
        continue
    
    print(f"Processing Batch {batch_idx + 1}: {fname}...")
    
    try:
        with h5py.File(fname, "r") as f:
            batch_obj = f['batch']
            
            # Access 'summary' based on file structure (handling variations)
            if 'summary' in batch_obj:
                summary_refs = batch_obj['summary']
            elif 'summary' in f:
                summary_refs = f['summary']
            else:
                # Some versions might store it directly under batch keys
                print("  Warning: Complex structure, trying direct keys...")
                # Fallback logic would go here, but standard Toyota files align with above
                continue

            # Handle shape dimensions (N, 1) vs (1, N)
            dim0, dim1 = summary_refs.shape
            n_cells = max(dim0, dim1)
            
            print(f"  > Found {n_cells} cells.")

            for i in range(n_cells):
                try:
                    # Get the reference for this specific cell
                    if dim0 > dim1: ref = summary_refs[i, 0]
                    else:           ref = summary_refs[0, i]

                    cell_group = f[ref]
                    
                    # EXTRACT DATA
                    data_dict = {}
                    
                    # Create Unique ID: "b1_cell_0", "b2_cell_1", etc.
                    # This prevents ID collisions between batches
                    unique_id = f"b{batch_idx+1}_c{i}"
                    data_dict['cell_id'] = unique_id
                    data_dict['batch'] = batch_idx + 1

                    # Grab 'cycle', 'QDischarge', 'IR', 'Tavg'
                    target_keys = ['cycle', 'QDischarge', 'IR', 'Tavg']
                    
                    valid_cell = False
                    for k in target_keys:
                        if k in cell_group:
                            val = cell_group[k][:]
                            val = val.flatten()
                            data_dict[k] = val
                            valid_cell = True
                    
                    if valid_cell:
                        df_cell = pd.DataFrame(data_dict)
                        # Filter: Removing super short/bad tests (under 50 cycles)
                        if df_cell['cycle'].max() > 50:
                            all_data.append(df_cell)
                            
                except Exception as e:
                    # Occasional empty ref in file
                    continue

    except Exception as e:
        print(f"CRITICAL ERROR reading {fname}: {e}")

# Save Final
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns
    cols = ['cell_id', 'batch', 'cycle'] + [c for c in final_df.columns if c not in ['cell_id', 'batch', 'cycle']]
    final_df = final_df[cols]
    
    final_df.to_csv(output_csv, index=False)
    print("------------------------------------------")
    print(f"SUCCESS! Combined data saved to '{output_csv}'")
    print(f"Total rows: {len(final_df)}")
    print(f"Total unique batteries: {final_df['cell_id'].nunique()}")
else:
    print("No data extracted. Did you download the .mat files?")