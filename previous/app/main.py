import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import matplotlib.pyplot as plt
import shap

from src.spatial_etl import build_spatial_feature_store
from src.model import TurismRadarRegressor

# Page configuration
st.set_page_config(
    page_title="TurismRadar | Spatial Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛰️ TurismRadar")
st.caption("De-Biasing Swedish Municipal Tourism Metrics via Spatial Anomaly Detection")

# --- Sidebar Controls ---
st.sidebar.header("Model Configuration")
sigma_val = st.sidebar.slider("Distance Decay Sigma (km)", min_value=5.0, max_value=100.0, value=25.0, step=5.0)
huber_alpha = st.sidebar.slider("Huber Loss Alpha", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# --- Data Pipeline Execution (Cached) ---
@st.cache_data
def load_and_process_data(sigma: float, alpha: float):
    # Mock data loading step — Replace with PostGIS / SCB PxWeb fetchers in production
    # Load geometries
    kommuner = gpd.read_file("data/kommuner.geojson")
    borders = gpd.read_file("data/borders.geojson")
    airports = gpd.read_file("data/airports.geojson")
    
    # Spatial feature extraction
    df_spatial = build_spatial_feature_store(kommuner, borders, airports, sigma=sigma)
    
    # Model inference & residual analysis
    model_engine = TurismRadarRegressor(alpha=alpha)
    df_results = model_engine.fit_predict(df_spatial)
    
    return df_results, model_engine

# Load processed dataset
try:
    df_data, engine = load_and_process_data(sigma_val, huber_alpha)
except Exception:
    st.info("💡 Place GeoJSON layer files in `data/` to run dynamic spatial pipeline. Displaying core dashboard layout structure.")
    st.stop()

# --- Top Key Metrics Summary ---
st.subheader("Regional Overview")
col1, col2, col3, col4 = st.columns(4)

total_raw_spend = df_data["S_total_spend"].sum() / 1e6
total_adj_spend = df_data["Y_predicted_leisure"].sum() / 1e6
total_spillover = df_data["residual_spillover"].clip(lower=0).sum() / 1e6

col1.metric("Total Raw Spend", f"{total_raw_spend:,.1f} M SEK")
col2.metric("Baseline Leisure Spend", f"{total_adj_spend:,.1f} M SEK")
col3.metric("Detected Noise Spillover", f"{total_spillover:,.1f} M SEK", delta=f"-{(total_spillover/total_raw_spend)*100:.1f}%")
col4.metric("Municipalities Analyzed", f"{len(df_data)}")

st.divider()

# --- Spatial PyDeck Map & Ranking Table ---
col_map, col_table = st.columns([3, 2])

with col_map:
    st.subheader("Spatial Residual & Spillover Map")
    
    # 1. Ensure CRS is WGS84 (lat/lon)
    df_map = df_data.to_crs(epsg=4326).copy()
    
    # 2. Assign dynamic fill color RGBA lists
    def get_color(residual):
        if residual > 100e6:     # High spillover noise (Red)
            return [220, 50, 50, 180]
        elif residual < -100e6:  # Genuine destination baseline (Green)
            return [50, 180, 50, 180]
        return [100, 140, 200, 180] # Neutral (Blue)

    df_map["fill_color"] = df_map["residual_spillover"].apply(get_color)
    
    # 3. Explicitly serialize to pure Python dict via GeoJSON string
    geojson_dict = json.loads(df_map.to_json())
    
    # 4. Define PyDeck GeoJsonLayer
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_dict,
        opacity=0.8,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        pickable=True
    )

    # Center view over Sweden
    view_state = pdk.ViewState(
        latitude=60.0,
        longitude=15.0,
        zoom=4.2,
        pitch=0
    )

    # Render Deck with open CartoDB dark style (no Mapbox API key required)
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            tooltip={"text": "{kommun_name}\nRaw Spend: {raw_per_capita:.0f} SEK/cap\nShift: {rank_shift}"}
        )
    )
    
with col_table:
    st.subheader("Top Rank Shift Discrepancies")
    st.dataframe(
        df_data[["kommun_name", "raw_rank", "adjusted_rank", "rank_shift", "residual_spillover"]]
        .sort_values(by="rank_shift", ascending=False)
        .head(10),
        use_container_width=True
    )

st.divider()

# --- Municipal Deep Dive & SHAP Waterfall ---
st.subheader("Municipal Anomaly Deep-Dive")
selected_muni = st.selectbox("Select Municipality", df_data["kommun_name"].unique())

muni_row = df_data[df_data["kommun_name"] == selected_muni].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Raw Rank", f"#{muni_row['raw_rank']}")
m2.metric("Adjusted Rank", f"#{muni_row['adjusted_rank']}", delta=int(muni_row['rank_shift']))
m3.metric("Raw Spend / Capita", f"{muni_row['raw_per_capita']:,.0f} SEK")
m4.metric("Adjusted Spend / Capita", f"{muni_row['adjusted_per_capita']:,.0f} SEK")

# SHAP Waterfall Plot
st.markdown("#### Feature Attribution (SHAP Explanation)")
shap_exp = engine.get_shap_explanation(df_data[df_data["kommun_name"] == selected_muni])

fig, ax = plt.subplots(figsize=(8, 3.5))
shap.plots.waterfall(shap_exp[0], show=False)
st.pyplot(fig)