import pandas as pd
import numpy as np
import os

df = pd.read_csv(f"81822-EnvDataSci_DashboardAssignment_Dataset.csv", low_memory=False)
df["SampleDateTime"] = pd.to_datetime(df["SampleDateTime"], format="mixed", errors="coerce")

def clean_value(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip()
    if v.startswith("<"):
        try:
            return float(v[1:]) / 2
        except ValueError:
            return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan

df["MeasurementValueNum"] = df["MeasurementValue"].apply(clean_value)
df.to_parquet("water_quality_data.parquet", compression="snappy", index=False)

print(f"Done. Parquet size: {os.path.getsize('water_quality_data.parquet') / 1024**2:.2f} MB")