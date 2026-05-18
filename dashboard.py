"""
Alberta Surface Water Quality - Interactive Dashboard
Environmental Data Scientist Dashboard Assignment

Author: Rayhan Kabir
Purpose: Interactive review of  surface water quality data (2020-2023)
         for stakeholders to identify data issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Alberta Water Quality Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2e75b6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f6fb;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e75b6;
    }
    [data-testid="stMetric"] {
        background-color: #f0f6fb;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e75b6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #f0f6fb;
        border-radius: 5px 5px 0 0;
        padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e75b6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CCME WATER QUALITY GUIDELINES (Aquatic Life Protection)
# Source: Canadian Council of Ministers of the Environment (CCME)
# ============================================================================
CCME_GUIDELINES = {
    "PH (LAB)": {"min": 6.5, "max": 9.0, "unit": "pH units",
                 "rationale": "CCME guideline for protection of freshwater aquatic life."},
    "PH (FIELD)": {"min": 6.5, "max": 9.0, "unit": "pH units",
                   "rationale": "CCME guideline for protection of freshwater aquatic life."},
    "OXYGEN DISSOLVED (FIELD METER)": {"min": 6.5, "max": None, "unit": "mg/L",
                                        "rationale": "CCME guideline (cold-water early life stages)."},
    "PHOSPHORUS TOTAL (P)": {"min": None, "max": 0.05, "unit": "mg/L",
                              "rationale": "CCME guidance trigger range; Alberta surface water benchmark."},
    "AMMONIA TOTAL": {"min": None, "max": 1.5, "unit": "mg/L",
                      "rationale": "CCME unionized ammonia threshold (approx. as total NH3 at typical pH/temp)."},
    "CHLORIDE DISSOLVED": {"min": None, "max": 120, "unit": "mg/L",
                            "rationale": "CCME long-term chronic guideline for freshwater life."},
    "NITRATE": {"min": None, "max": 13, "unit": "mg/L",
                "rationale": "CCME chronic exposure guideline for freshwater life."},
    "NITROGEN TOTAL KJELDAHL (TKN)": {"min": None, "max": 1.0, "unit": "mg/L",
                                       "rationale": "Provincial trigger value used for screening."},
    "TURBIDITY": {"min": None, "max": 8, "unit": "NTU",
                  "rationale": "CCME short-term increase above background (clear-flow systems)."},
    "TEMPERATURE WATER": {"min": None, "max": 25, "unit": "deg C",
                          "rationale": "Site-specific aquatic life thermal tolerance threshold."},
}

# ============================================================================
# RIVER BASIN NAMES
# ============================================================================
BASIN_NAMES = {
    "ATH": "Athabasca",
    "NSA": "North Saskatchewan",
    "BAT": "Battle",
    "PEA": "Peace",
    "RED": "Red Deer",
    "OLD": "Oldman",
    "BEA": "Beaver",
    "BOW": "Bow",
    "SSA": "South Saskatchewan",
    "MIL": "Milk",
}

# ============================================================================
# DATA LOADING (cached for performance)
# ============================================================================
@st.cache_data(show_spinner="Loading water quality data...")
def load_data():
    """Load and pre-clean the water quality dataset."""
    df = pd.read_parquet('water_quality_data.parquet')
    df["Year"] = df["SampleDateTime"].dt.year
    df["Month"] = df["SampleDateTime"].dt.month
    df["BasinName"] = df["RiverBasinCode"].map(BASIN_NAMES).fillna(df["RiverBasinCode"])
    return df

@st.cache_data
def compute_station_quality_scores(df):
    """
    Compute a data quality / risk score for each station.
    Higher score = more data concerns (worse).
    Score components:
      - % of records flagged with quality qualifiers
      - % below detection limit
      - % above detection / over-range
      - Outliers (impossible values, e.g. pH > 14 or DO < 0)
      - Guideline exceedances
    """
    records = []
    for station, sub in df.groupby("Station"):
        total = len(sub)
        if total == 0:
            continue

        qualifier_issues = sub["MeasurementQualifierDescription"].notna().sum()
        below_dl = (sub["MeasurementFlag"] == "L").sum()
        above_dl = (sub["MeasurementFlag"] == "G").sum()

        # Impossible values
        impossible = 0
        ph = sub[sub["VariableName"].isin(["PH (LAB)", "PH (FIELD)"])]["MeasurementValueNum"]
        impossible += ((ph < 0) | (ph > 14)).sum()
        do = sub[sub["VariableName"] == "OXYGEN DISSOLVED (FIELD METER)"]["MeasurementValueNum"]
        impossible += (do < 0).sum()

        # Guideline exceedances
        exceedances = 0
        for var, lim in CCME_GUIDELINES.items():
            v = sub[sub["VariableName"] == var]["MeasurementValueNum"].dropna()
            if lim["max"] is not None:
                exceedances += (v > lim["max"]).sum()
            if lim["min"] is not None:
                exceedances += (v < lim["min"]).sum()

        quality_score = (
            (qualifier_issues / total) * 40 +
            (impossible / total) * 30 +
            (exceedances / total) * 20 +
            (above_dl / total) * 10
        ) * 100

        records.append({
            "Station": station,
            "TotalMeasurements": total,
            "QualifierIssues": qualifier_issues,
            "QualifierIssuePct": qualifier_issues / total * 100,
            "BelowDetectionLimit": below_dl,
            "AboveDetectionLimit": above_dl,
            "ImpossibleValues": impossible,
            "GuidelineExceedances": exceedances,
            "DataQualityScore": quality_score,
            "Latitude": sub["LatitudeDecimalDegrees"].iloc[0],
            "Longitude": sub["LongitudeDecimalDegrees"].iloc[0],
            "BasinName": sub["BasinName"].iloc[0],
            "Basin": sub["RiverBasinCode"].iloc[0],
        })
    return pd.DataFrame(records).sort_values("DataQualityScore", ascending=False)


# ============================================================================
# LOAD DATA
# ============================================================================
df =load_data()
station_scores = compute_station_quality_scores(df)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="main-header">💧 Alberta Surface Water Quality Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    "**Interactive review of unvalidated surface water quality data (2020–2023)** — "
    "for scientists, data stewards, and field staff to identify data issues, exceedances, "
    "and patterns of concern across Alberta's monitoring network."
)

# ============================================================================
# SIDEBAR — GLOBAL FILTERS
# ============================================================================
st.sidebar.markdown("### 🔍 Global Filters")
st.sidebar.markdown("*Applied across all tabs*")

# Date range
date_min = df["SampleDateTime"].min().date()
date_max = df["SampleDateTime"].max().date()
date_range = st.sidebar.date_input(
    "📅 Sample Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

# Basin selection
all_basins = sorted(df["BasinName"].unique())
selected_basins = st.sidebar.multiselect(
    "🏞️ River Basin(s)",
    options=all_basins,
    default=all_basins,
)

# Apply filters
mask = (
    (df["SampleDateTime"].dt.date >= start_date) &
    (df["SampleDateTime"].dt.date <= end_date) &
    (df["BasinName"].isin(selected_basins))
)
fdf = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered records:** {len(fdf):,} of {len(df):,}")
st.sidebar.markdown(f"**Stations in view:** {fdf['Station'].nunique()}")
st.sidebar.markdown(f"**Parameters in view:** {fdf['VariableName'].nunique()}")

with st.sidebar.expander("ℹ️ About this dashboard"):
    st.markdown("""
    Built for an Environmental Data Scientist assignment using the
    Alberta surface water quality dataset (2020–2023).

    **Features:**
    - Interactive filtering by date, basin, station, parameter
    - Configurable guideline limits (CCME defaults + custom)
    - Data quality scoring & top stations of concern
    - Spatial risk map with quality-based color coding
    - Time-series & trend analysis

    """)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔬 Parameter Explorer",
    "🛡️ Data Quality",
    "⚠️ Guideline Exceedances",
    "🗺️ Spatial Risk Map",
    "📈 Trends & Stations of Concern",
])

# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with tab1:
    st.subheader("Dataset at a Glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Measurements", f"{len(fdf):,}")
    c2.metric("Unique Stations", f"{fdf['Station'].nunique()}")
    c3.metric("Parameters Tracked", f"{fdf['VariableName'].nunique()}")
    c4.metric("River Basins", f"{fdf['RiverBasinCode'].nunique()}")

    c1, c2, c3, c4 = st.columns(4)
    qual_issue_pct = fdf["MeasurementQualifierDescription"].notna().mean() * 100
    below_dl_pct = (fdf["MeasurementFlag"] == "L").mean() * 100
    c1.metric("Records with QA flags", f"{qual_issue_pct:.2f}%")
    c2.metric("Below Detection Limit", f"{below_dl_pct:.1f}%")
    suspect_pct = fdf["MeasurementQualifierDescription"].str.contains("SUSPECT", na=False).mean() * 100
    c3.metric("Flagged 'Suspect'", f"{suspect_pct:.2f}%")
    ht_pct = fdf["MeasurementQualifierDescription"].str.contains("HOLDING TIME EXCEEDED", na=False).mean() * 100
    c4.metric("Holding Time Exceeded", f"{ht_pct:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Measurements by River Basin")
        basin_counts = fdf.groupby("BasinName").size().reset_index(name="Count").sort_values("Count", ascending=True)
        fig = px.bar(basin_counts, x="Count", y="BasinName", orientation="h",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(height=400, showlegend=False, yaxis_title="", xaxis_title="Measurements")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Measurements Over Time (Monthly)")
        ts = fdf.set_index("SampleDateTime").resample("MS").size().reset_index(name="Count")
        ts.columns = ["Month", "Count"]
        fig = px.area(ts, x="Month", y="Count", color_discrete_sequence=["#2e75b6"])
        fig.update_layout(height=400, xaxis_title="", yaxis_title="Measurements")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 15 Most-Measured Parameters")
    top_vars = fdf["VariableName"].value_counts().head(15).reset_index()
    top_vars.columns = ["Parameter", "Count"]
    fig = px.bar(top_vars, x="Count", y="Parameter", orientation="h",
                 color="Count", color_continuous_scale="Teal")
    fig.update_layout(height=500, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 2 — PARAMETER EXPLORER
# ============================================================================
with tab2:
    st.subheader("Parameter Explorer")
    st.caption("Pick a parameter, optionally focus on specific stations, and explore time-series, distribution, and summary statistics.")

    c1, c2 = st.columns([1, 2])
    with c1:
        # Sort parameters by frequency
        param_options = fdf["VariableName"].value_counts().index.tolist()
        selected_param = st.selectbox("Parameter", options=param_options, index=0 if param_options else None)
    with c2:
        stations_available = sorted(fdf[fdf["VariableName"] == selected_param]["Station"].unique())
        selected_stations = st.multiselect(
            f"Stations ({len(stations_available)} available — leave empty for all)",
            options=stations_available,
            default=[],
        )

    pdata = fdf[fdf["VariableName"] == selected_param].copy()
    if selected_stations:
        pdata = pdata[pdata["Station"].isin(selected_stations)]

    if len(pdata) == 0:
        st.warning("No data for the current filter selection.")
    else:
        # Stats
        unit = pdata["UnitCode"].iloc[0] if len(pdata) > 0 else ""
        vals = pdata["MeasurementValueNum"].dropna()

        st.markdown(f"##### {selected_param} ({unit})")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Records", f"{len(pdata):,}")
        c2.metric("Min", f"{vals.min():.3f}")
        c3.metric("Median", f"{vals.median():.3f}")
        c4.metric("Mean", f"{vals.mean():.3f}")
        c5.metric("Max", f"{vals.max():.3f}")

        # Time series
        st.markdown("#### Time Series")
        if selected_stations and len(selected_stations) <= 10:
            fig = px.scatter(pdata, x="SampleDateTime", y="MeasurementValueNum",
                             color="Station", opacity=0.7,
                             labels={"MeasurementValueNum": f"Value ({unit})", "SampleDateTime": "Date"})
        else:
            fig = px.scatter(pdata, x="SampleDateTime", y="MeasurementValueNum",
                             opacity=0.4, color_discrete_sequence=["#2e75b6"],
                             labels={"MeasurementValueNum": f"Value ({unit})", "SampleDateTime": "Date"})

        # Guideline overlay
        if selected_param in CCME_GUIDELINES:
            g = CCME_GUIDELINES[selected_param]
            if g["max"] is not None:
                fig.add_hline(y=g["max"], line_dash="dash", line_color="red",
                              annotation_text=f"Upper limit: {g['max']} {g['unit']}",
                              annotation_position="top right")
            if g["min"] is not None:
                fig.add_hline(y=g["min"], line_dash="dash", line_color="orange",
                              annotation_text=f"Lower limit: {g['min']} {g['unit']}",
                              annotation_position="bottom right")

        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Distribution + Box plot
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribution")
            fig = px.histogram(pdata, x="MeasurementValueNum", nbins=50,
                               color_discrete_sequence=["#2e75b6"],
                               labels={"MeasurementValueNum": f"Value ({unit})"})
            fig.update_layout(height=400, yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Box Plot by Basin")
            fig = px.box(pdata, x="BasinName", y="MeasurementValueNum",
                         color="BasinName",
                         labels={"MeasurementValueNum": f"Value ({unit})", "BasinName": "Basin"})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 3 — DATA QUALITY ASSESSMENT
# ============================================================================
with tab3:
    st.subheader("Data Quality Assessment")
    st.caption("Identify QA flags, suspect values, holding-time issues, and physically impossible measurements.")

    # Impossible value detection
    st.markdown("#### 🚨 Physically Impossible Values Detected")
    st.caption("Values that violate physical/chemical constraints — likely entry or instrument errors.")

    impossible_records = []
    # pH must be 0–14
    ph_bad = fdf[fdf["VariableName"].isin(["PH (LAB)", "PH (FIELD)"]) &
                 ((fdf["MeasurementValueNum"] < 0) | (fdf["MeasurementValueNum"] > 14))]
    if len(ph_bad) > 0:
        impossible_records.append(("pH outside 0–14", len(ph_bad), ph_bad))

    do_bad = fdf[(fdf["VariableName"] == "OXYGEN DISSOLVED (FIELD METER)") &
                 (fdf["MeasurementValueNum"] < 0)]
    if len(do_bad) > 0:
        impossible_records.append(("Negative dissolved oxygen", len(do_bad), do_bad))

    temp_bad = fdf[(fdf["VariableName"] == "TEMPERATURE WATER") &
                   ((fdf["MeasurementValueNum"] < -2) | (fdf["MeasurementValueNum"] > 40))]
    if len(temp_bad) > 0:
        impossible_records.append(("Water temp outside -2 to 40 °C", len(temp_bad), temp_bad))

    if impossible_records:
        for label, n, data in impossible_records:
            with st.expander(f"⚠️ {label} — {n} record(s)"):
                display = data.sort_values("MeasurementValueNum", ascending=False)[
                    ["Station", "SampleDateTime", "VariableName",
                    "MeasurementValue", "UnitCode"]
                ]
                st.dataframe(display, use_container_width=True, height=250)
    else:
        st.success("✅ No physically impossible values found in the current filter.")

    st.markdown("---")

    # Qualifier breakdown
    st.markdown("#### Measurement Qualifier Breakdown")
    qual = fdf["MeasurementQualifierDescription"].value_counts().head(15).reset_index()
    qual.columns = ["Qualifier", "Count"]
    fig = px.bar(qual, x="Count", y="Qualifier", orientation="h",
                 color="Count", color_continuous_scale="Reds")
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Below detection limit by parameter
    st.markdown("#### Top Parameters by % Below Detection Limit")
    below_dl_pct = (
        fdf.assign(below=fdf["MeasurementFlag"] == "L")
        .groupby("VariableName")
        .agg(total=("below", "count"), below=("below", "sum"))
        .assign(pct=lambda x: x["below"] / x["total"] * 100)
        .query("total >= 100")
        .sort_values("pct", ascending=False)
        .head(15)
        .reset_index()
    )
    fig = px.bar(below_dl_pct, x="pct", y="VariableName", orientation="h",
                 color="pct", color_continuous_scale="Oranges",
                 labels={"pct": "% Below Detection Limit", "VariableName": ""})
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 4 — GUIDELINE EXCEEDANCES
# ============================================================================
with tab4:
    st.subheader("Guideline Exceedances")
    st.caption("Compare measurements against CCME water quality guidelines or set custom limits.")

    # Parameter for exceedance check
    guideline_params = list(CCME_GUIDELINES.keys())
    available_guideline_params = [p for p in guideline_params if p in fdf["VariableName"].values]

    if not available_guideline_params:
        st.warning("No parameters with guidelines in the current filter.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            sel_param = st.selectbox("Parameter to evaluate", options=available_guideline_params)
        with c2:
            st.markdown(
                            """
                            <div style="
                                background-color: #e7f3ff;
                                border: 2px solid #2e75b6;
                                padding: 6px 12px;
                                border-radius: 6px;
                                margin-top: 28px;
                                font-weight: 700;
                                color: #1f4e79;
                                display: inline-block;
                            ">
                                ⚙️ CUSTOM LIMITS
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            use_custom = st.checkbox(" Check this box to override CCME default thresholds", value=False)

        default_g = CCME_GUIDELINES[sel_param]
        if use_custom:
            c1, c2 = st.columns(2)
            with c1:
                min_limit = st.number_input("Lower limit (leave blank for none)",
                                            value=default_g["min"] if default_g["min"] is not None else 0.0,
                                            format="%.4f")
                use_min = st.checkbox("Apply lower limit", value=default_g["min"] is not None)
            with c2:
                max_limit = st.number_input("Upper limit (leave blank for none)",
                                            value=default_g["max"] if default_g["max"] is not None else 0.0,
                                            format="%.4f")
                use_max = st.checkbox("Apply upper limit", value=default_g["max"] is not None)
            min_limit = min_limit if use_min else None
            max_limit = max_limit if use_max else None
        else:
            min_limit = default_g["min"]
            max_limit = default_g["max"]
            st.info(f"**CCME Guideline:** {default_g['rationale']}")

        pdata = fdf[fdf["VariableName"] == sel_param].copy()
        pdata = pdata.dropna(subset=["MeasurementValueNum"])

        exceeds = pd.Series([False] * len(pdata), index=pdata.index)
        if max_limit is not None:
            exceeds |= pdata["MeasurementValueNum"] > max_limit
        if min_limit is not None:
            exceeds |= pdata["MeasurementValueNum"] < min_limit
        pdata["Exceeds"] = exceeds

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Measurements", f"{len(pdata):,}")
        c2.metric("Exceedances", f"{exceeds.sum():,}")
        c3.metric("Exceedance Rate", f"{exceeds.mean() * 100:.2f}%")

        # Scatter with limits
        unit = pdata["UnitCode"].iloc[0] if len(pdata) > 0 else ""
        fig = px.scatter(pdata, x="SampleDateTime", y="MeasurementValueNum",
                         color="Exceeds",
                         color_discrete_map={True: "#d62728", False: "#2e75b6"},
                         labels={"MeasurementValueNum": f"Value ({unit})", "SampleDateTime": "Date"},
                         opacity=0.6)
        if max_limit is not None:
            fig.add_hline(y=max_limit, line_dash="dash", line_color="red",
                          annotation_text=f"Upper limit: {max_limit}")
        if min_limit is not None:
            fig.add_hline(y=min_limit, line_dash="dash", line_color="orange",
                          annotation_text=f"Lower limit: {min_limit}")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Top stations with exceedances
        st.markdown("#### Top Stations by Exceedance Rate")
        station_exc = (
            pdata.groupby("Station")
            .agg(Measurements=("Exceeds", "count"), Exceedances=("Exceeds", "sum"))
            .assign(ExceedanceRate=lambda x: x["Exceedances"] / x["Measurements"] * 100)
            .query("Measurements >= 5")
            .sort_values("ExceedanceRate", ascending=False)
            .head(15)
            .reset_index()
        )
        st.dataframe(station_exc, use_container_width=True, height=400)


# ============================================================================
# TAB 5 — SPATIAL RISK MAP
# ============================================================================
with tab5:
    st.subheader("Spatial Risk Map")
    st.caption("Stations colored by data quality / risk score. Red circles = higher concern. Hover for station details.")

    # Use filtered data to compute scores
    fdf_scores = compute_station_quality_scores(fdf) if len(fdf) > 0 else pd.DataFrame()

    if len(fdf_scores) == 0:
        st.warning("No data for the current filter.")
    else:
        c1, c2 = st.columns([3, 1])
        with c2:
            map_style = st.radio("Map style", ["Light", "Dark", "Streets"], index=0)
        with c1:
            # Color mapping based on score
            def score_color(score, max_score):
                """Return color tuple (R,G,B,A) — green (good) → red (bad)."""
                if max_score == 0:
                    pct = 0
                else:
                    pct = min(score / max_score, 1.0)
                r = int(255 * pct)
                g = int(255 * (1 - pct))
                return [r, g, 50, 200]

            max_score = fdf_scores["DataQualityScore"].max()
            fdf_scores["color"] = fdf_scores["DataQualityScore"].apply(
                lambda s: score_color(s, max_score)
            )
             # Size also reflects data quality score (redundant encoding with color)
            score_range = fdf_scores["DataQualityScore"].max() - fdf_scores["DataQualityScore"].min() + 1e-9
            score_normalized = (fdf_scores["DataQualityScore"] - fdf_scores["DataQualityScore"].min()) / score_range
            fdf_scores["radius"] = (4 + score_normalized * 14) #* 1000  # 4–18 unit range

            style_map = {
                "Light": "mapbox://styles/mapbox/light-v9",
                "Dark": "mapbox://styles/mapbox/dark-v9",
                "Streets": "mapbox://styles/mapbox/streets-v11",
            }
            # Fallback to free Carto styles (no mapbox token needed)
            free_styles = {
                "Light": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                "Dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                "Streets": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            }

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=fdf_scores,
                get_position=["Longitude", "Latitude"],
                get_fill_color="color",
                get_radius="radius_pixels",
                radius_units="pixels",          # 👈 NEW — size in screen pixels, not meters
                radius_min_pixels=4,             # 👈 NEW — never smaller than 4px at any zoom
                radius_max_pixels=20,            # 👈 NEW — never bigger than 20px at any zoom
                pickable=True,
                opacity=0.8,
                stroked=True,
                get_line_color=[0, 0, 0, 150],
                line_width_min_pixels=1,
            )
            view = pdk.ViewState(
                latitude=53.5,
                longitude=-115.0,
                zoom=4.8,
                pitch=0,
            )
            deck = pdk.Deck(
                map_style=free_styles[map_style],
                initial_view_state=view,
                layers=[layer],
                tooltip={
                    "html": "<b>{Station}</b><br/>"
                            "Basin: {BasinName}<br/>"
                            "Measurements: {TotalMeasurements}<br/>"
                            "Quality Score: {DataQualityScore}<br/>"
                            "QA flags: {QualifierIssues}<br/>"
                            "Impossible values: {ImpossibleValues}<br/>"
                            "Guideline exceedances: {GuidelineExceedances}",
                    "style": {"backgroundColor": "white", "color": "black",
                              "fontSize": "12px", "padding": "8px"}
                },
            )
            st.pydeck_chart(deck)

        #st.caption("🟢 Green = lower risk · 🔴 Red = higher data-quality concerns")
        # Color gradient legend (matches the map colors)
        # Color gradient legend (matches the map colors)
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:12px; margin-top:8px; font-size:0.9em;">
                <span><b>Risk scale:</b></span>
                <span style="color:#555;">Lower</span>
                <div style="
                    flex: 0 0 240px;
                    height: 14px;
                    border-radius: 7px;
                    background: linear-gradient(to right,
                        rgb(0,255,50),
                        rgb(128,191,50),
                        rgb(191,128,50),
                        rgb(255,0,50));
                    border: 1px solid #ccc;
                "></div>
                <span style="color:#555;">Higher</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Legend & detail table
        st.markdown("#### Station Risk Ranking")
        st.dataframe(
            fdf_scores[[
                "Station", "BasinName", "TotalMeasurements", "QualifierIssues",
                "ImpossibleValues", "GuidelineExceedances", "DataQualityScore"
            ]].head(20).style.background_gradient(
                subset=["DataQualityScore"], cmap="RdYlGn_r"
            ).format({"DataQualityScore": "{:.2f}"}),
            use_container_width=True,
            height=400,
        )


# ============================================================================
# TAB 6 — TRENDS & STATIONS OF CONCERN
# ============================================================================
with tab6:
    st.subheader("Trend Analysis & Top Stations of Concern")
    st.caption("Identify temporal patterns and prioritize stations requiring attention.")

    # Top stations of concern
    st.markdown("#### 🎯 Top 10 Stations of Concern")
    fdf_scores = compute_station_quality_scores(fdf) if len(fdf) > 0 else pd.DataFrame()
    if len(fdf_scores) > 0:
        top10 = fdf_scores.head(10)

        fig = go.Figure(go.Bar(
            x=top10["DataQualityScore"],
            y=top10["Station"],
            orientation="h",
            marker=dict(color=top10["DataQualityScore"], colorscale="Reds",
                        showscale=True, colorbar=dict(title="Quality Score")),
            text=[f"{v:.1f}" for v in top10["DataQualityScore"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Data Quality Score (higher = more concerns)",
            margin=dict(l=300),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Trend analysis for selected parameter
    st.markdown("#### 📈 Multi-Year Trend Analysis")
    c1, c2 = st.columns([1, 1])
    with c1:
        trend_params = fdf["VariableName"].value_counts().index.tolist()
        trend_param = st.selectbox("Parameter for trend analysis", options=trend_params,
                                    index=trend_params.index("PHOSPHORUS TOTAL (P)") if "PHOSPHORUS TOTAL (P)" in trend_params else 0)
    with c2:
        agg_method = st.selectbox("Aggregation", ["Median", "Mean", "Max"], index=0)

    trend_data = fdf[fdf["VariableName"] == trend_param].copy()
    trend_data = trend_data.dropna(subset=["MeasurementValueNum"])

    if len(trend_data) > 0:
        agg_func = {"Median": "median", "Mean": "mean", "Max": "max"}[agg_method]
        monthly = (
            trend_data
            .set_index("SampleDateTime")
            .groupby([pd.Grouper(freq="MS"), "BasinName"])["MeasurementValueNum"]
            .agg(agg_func)
            .reset_index()
        )
        unit = trend_data["UnitCode"].iloc[0]
        fig = px.line(monthly, x="SampleDateTime", y="MeasurementValueNum",
                      color="BasinName",
                      labels={"MeasurementValueNum": f"{agg_method} ({unit})",
                              "SampleDateTime": "", "BasinName": "Basin"})

        if trend_param in CCME_GUIDELINES:
            g = CCME_GUIDELINES[trend_param]
            if g["max"] is not None:
                fig.add_hline(y=g["max"], line_dash="dash", line_color="red",
                              annotation_text=f"CCME limit: {g['max']}")

        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Seasonal pattern
        st.markdown("#### 🌱 Seasonal Pattern")
        trend_data["MonthName"] = trend_data["SampleDateTime"].dt.month_name()
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        fig = px.box(trend_data, x="MonthName", y="MeasurementValueNum",
                     category_orders={"MonthName": month_order},
                     color_discrete_sequence=["#2e75b6"],
                     labels={"MeasurementValueNum": f"Value ({unit})", "MonthName": "Month"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(
    "**Prepared by Rayhan Kabir**  "
    
    f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
)
