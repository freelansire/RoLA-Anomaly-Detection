"""
RoLA (Real-time Online Lightweight Anomaly Detection) - With Ground Truth Evaluation
Author: Samuel Moses Orokpo
Date: 20th March, 2025

Preprocessing steps

"""

import pandas as pd

#Merging raw datasets and inspecting

# Define file paths for the uploaded CSVs
file_paths = {
    "C3_Temperature": "/mnt/data/C3_Temperature_10-28_10-30.csv",
    "C3_Turbidity": "/mnt/data/C3_Turbidity_10-28_10-30.csv",
    "Flow_Flow": "/mnt/data/Flow_Flow_10-28_10-30.csv",
    "Flow_Temperature": "/mnt/data/Flow_Temperature_10-28_10-30.csv",
    "Optode_Concentration": "/mnt/data/Optode_Concentration10-28_10-30.csv",
    "Optode_Saturation": "/mnt/data/Optode_Saturation_10-28_10-30.csv",
    "Optode_Temperature": "/mnt/data/Optode_Temperature_10-28_10-30.csv",
    "SEB45_Conductivity": "/mnt/data/SEB45_Conductivity_10-28_10-30.csv",
    "SEB45_Salinity": "/mnt/data/SEB45_Salinity_10-28_10-30.csv",
}

# Load each dataset into a dictionary
datasets = {}
for name, path in file_paths.items():
    try:
        datasets[name] = pd.read_csv(path)
        print(f"Loaded {name}: {datasets[name].shape} rows")
    except Exception as e:
        print(f"Failed to load {name}: {e}")

# Display first few rows of each dataset
for name, df in datasets.items():
    print(f"\n Preview of {name}:")
    print(df.head())

"""
Findings from the Dataset Inspection

-Each CSV file contains time series data for different environmental parameters (e.g., temperature, turbidity, salinity, flow, etc.).
-Timestamps appear to be the common identifier across all datasets.
-Each dataset contains four columns, but only one of them is useful for numerical processing (sensor readings).
-Latitude and longitude information is present, but will exclude  for now.
-Dropped non-numeric columns (e.g., metadata, text-based data).
"""

# Extract and merge datasets based on timestamps and preprocessing
merged_df = None

# Ensure all numerical columns are properly converted before interpolation

# Drop potential non-numeric columns except Timestamp
for col in merged_df.columns:
    if col != "Timestamp":
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')  # Convert to float, set errors to NaN

# Reapply missing value handling
merged_df.ffill(inplace=True)  # Forward-fill missing values
merged_df.bfill(inplace=True)  # Backward-fill missing values

# Apply interpolation only to numeric columns
numeric_cols = merged_df.select_dtypes(include=['number']).columns
merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method='linear')

"""
Generating Ground truth Labels
"""

from scipy.stats import zscore


file_path = "/dataset/Final_Processed_Dataset_for_RoLA.csv"
df = pd.read_csv(file_path)

# Convert Timestamp to datetime and set as index
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.set_index("Timestamp", inplace=True)

# Compute Z-scores for each numerical column
z_scores = df.apply(zscore)

# Mark anomalies where Z-score is greater than 3 (3σ rule)
df["Anomaly_Label"] = (z_scores.abs() > 3).any(axis=1).astype(int)


