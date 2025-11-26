import h5py
import numpy as np
import pandas as pd

filename = "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
output_csv = "battery_data_summary.csv"

all_data = []

print("Opening file...")

with h5py.File(filename, "r") as f:
    # 1. Access the 'batch' entry
    batch_obj = f['batch']
    
    # Check if 'batch' is a Group (folder) or Dataset
    if isinstance(batch_obj, h5py.Group):
        # The data is usually inside the 'summary' key within the batch group
        print(f"Batch keys detected: {list(batch_obj.keys())}")
        if 'summary' in batch_obj:
            # This is the array of pointers to every cell's summary
            summary_refs = batch_obj['summary'] 
        else:
            raise KeyError("Could not find 'summary' inside 'batch' group.")
    else:
        # Fallback: sometimes batch itself is the dataset
        summary_refs = batch_obj

    # 2. Get dimensions
    # MATLAB arrays are often (N, 1) or (1, N). We generally want N.
    n_cells = summary_refs.shape[0] if summary_refs.shape[0] > 1 else summary_refs.shape[1]
    print(f"Processing {n_cells} cells...")

    # 3. Iterate through every cell
    for i in range(n_cells):
        try:
            # Handle shape variations (index into the correct axis)
            if summary_refs.shape[0] > summary_refs.shape[1]:
                ref = summary_refs[i, 0]
            else:
                ref = summary_refs[0, i]

            # Follow the reference to get the actual cell data
            # HDF5 references need to be dereferenced using f[ref]
            cell_summary_data = f[ref]
            
            # Prepare a temporary dict to hold this cell's data
            cell_id = f"cell_{i}"
            data_dict = {'cell_id': cell_id}
            
            # Keys to extract (Case sensitive check required mostly)
            # This dataset typically uses: 'cycle', 'QDischarge', 'QCharge', 'IR', 'Tavg'
            keys_to_extract = ['cycle', 'QDischarge', 'QCharge', 'IR', 'Tmax', 'Tavg', 'chargetime']

            has_data = False
            
            for k in keys_to_extract:
                if k in cell_summary_data:
                    # Extract the values, flatten them to 1D array
                    val = cell_summary_data[k][:]
                    data_dict[k] = val.flatten()
                    has_data = True
            
            if has_data:
                # Create DataFrame for this cell
                df_cell = pd.DataFrame(data_dict)
                all_data.append(df_cell)
                if i % 10 == 0:
                    print(f"Processed {cell_id}...")
            
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")

# 4. Save to CSV
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Organize columns nicely
    first_cols = ['cell_id', 'cycle']
    existing_cols = [c for c in final_df.columns if c in first_cols]
    other_cols = [c for c in final_df.columns if c not in first_cols]
    final_df = final_df[existing_cols + other_cols]

    final_df.to_csv(output_csv, index=False)
    print("------------------------------------------")
    print(f"Done! Saved {len(final_df)} rows to '{output_csv}'")
else:
    print("No data extracted. Check file structure.")