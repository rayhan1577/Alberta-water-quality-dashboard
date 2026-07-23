# Alberta Surface Water Quality Dashboard

Interactive dashboard for reviewing unvalidated surface water quality data from Alberta's monitoring network (2020–2023).  

## What this dashboard does

The dashboard helps :

1. **Visualize** measurements with interactive filters (date range, basin, parameter, station)
2. **Compare** values against CCME water quality guidelines (with custom limit overrides)
3. **Assess data quality** — flag suspect values, holding-time issues, physically impossible measurements
4. **Identify top stations of concern** through a composite Data Quality Score
5. **Map spatial risk** — stations are color-coded by data quality on an interactive map
6. **Analyze trends** — multi-year time-series and seasonal patterns by basin

## Project structure

```
.
├── dashboard.py            # Main dashboard application
├── convert_to_parquet.py   # Compress CSV file to parquet
├── water_quality_data.parquet  # Cleaned dataset (Parquet, ~1.5 MB)
├── requirements.txt            # Python dependencies
├── README.md                   # This file

```

## Run locally

**Requirements:** Python 3.9+

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
python -m streamlit run dashboard.py

# The dashboard will open in the browser at http://localhost:8501
```

## Hosted version

The dashboard is hosted on Streamlit Community Cloud (free, read-only access):

**URL:** https://alberta-water-quality-dashboard-by-rayhan.streamlit.app/

No login required — just open the link.

## Data preparation notes

The raw CSV (≈70 MB, 247k rows) is converted to Parquet (≈1.5 MB) for fast loading.
The conversion logic also:
- Parses dates from mixed formats
- Converts string `"<0.05"`-style detection-limit entries to numeric values (using half-detection-limit substitution)
- Maps basin codes to readable names

To rebuild the parquet from the original CSV:

```python
import pandas as pd, numpy as np

df = pd.read_csv("water_quality_data.csv", low_memory=False)
df["SampleDateTime"] = pd.to_datetime(df["SampleDateTime"], format="mixed", errors="coerce")

def clean_value(v):
    if pd.isna(v): return np.nan
    v = str(v).strip()
    if v.startswith("<"):
        try: return float(v[1:]) / 2
        except: return np.nan
    try: return float(v)
    except: return np.nan

df["MeasurementValueNum"] = df["MeasurementValue"].apply(clean_value)
df.to_parquet("water_quality_data.parquet", compression="snappy", index=False)
```

## Methodology — Data Quality Score

Each station receives a composite score (higher = more concerns):

```
Score = (QualifierIssue%  × 40)
      + (ImpossibleValue% × 30)
      + (Exceedance%      × 20)
      + (AboveDetection%  × 10)
```

Components:
- **Qualifier issues** — records flagged as SUSPECT, HOLDING TIME EXCEEDED, STANDARD PROCEDURE NOT FOLLOWED, etc.
- **Impossible values** — pH outside 0–14, negative dissolved oxygen, water temperature outside −2 to 40 °C
- **Guideline exceedances** — values outside CCME limits for pH, P, NH₃, Cl, NO₃, TKN, turbidity, temperature
- **Above-detection over-range** — values flagged 'G' (above upper detection limit)

## CCME guidelines used

| Parameter | Lower limit | Upper limit | Unit |
|---|---|---|---|
| pH | 6.5 | 9.0 | pH units |
| Dissolved Oxygen | 6.5 | — | mg/L |
| Total Phosphorus | — | 0.05 | mg/L |
| Total Ammonia | — | 1.5 | mg/L |
| Chloride Dissolved | — | 120 | mg/L |
| Nitrate | — | 13 | mg/L |
| Total Kjeldahl Nitrogen | — | 1.0 | mg/L |
| Turbidity | — | 8 | NTU |

These are configurable in the dashboard via the "custom limits" checkbox.

---

## References

1. **Canadian Council of Ministers of the Environment (CCME).** *Canadian Water Quality Guidelines for the Protection of Aquatic Life.* https://ccme.ca/en/resources/water-aquatic-life

2. **Alberta Environment.** (1999). *Surface Water Quality Guidelines for Use in Alberta.* Edmonton, AB. Available at: https://open.alberta.ca/dataset/e0074537-88f6-4cc4-af5c-808161fa9af8

3. **Apfelbaum, S.I., Heimerl, S., & Waller, D.M.** (2021). Shifts in precipitation and agricultural intensity increase phosphorus concentrations and loads in an agricultural watershed. *Journal of Environmental Management*, 290, 112651. https://www.sciencedirect.com/science/article/abs/pii/S0301479721000815


