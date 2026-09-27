import numpy as np
import pandas as pd
from data.scb_client import fetch_scb_population, fetch_scb_tourism_stats
from data.spatial_client import load_municipal_geometries, compute_spatial_distances

def build_turismradar_dataset(sigma_km: float = 25.0, year: str = "2023") -> pd.DataFrame:
    print("1. Fetching real population statistics from SCB PxWeb API...")
    df_pop = fetch_scb_population(year=year)
    print(f"   Loaded {len(df_pop)} municipalities from SCB.")
    
    print("2. Mapping tourism metrics and transaction totals...")
    df_tourism = fetch_scb_tourism_stats(df_pop)
    df_stats = pd.merge(df_pop, df_tourism, on="knkod")
    
    print("3. Fetching Eurostat LAU geospatial boundaries for Sweden...")
    gdf_spatial = load_municipal_geometries()
    
    print("4. Computing geodesic distances to borders & airports (EPSG:3006)...")
    gdf_spatial = compute_spatial_distances(gdf_spatial)
    
    # Join on 4-digit municipality code
    gdf_spatial["knkod"] = gdf_spatial["LAU_ID"].str.zfill(4)
    df_merged = gdf_spatial.merge(df_stats, on="knkod")
    
    # Exponential Distance-Decay Features
    df_merged["F_border"] = np.exp(-df_merged["d_border_km"] / sigma_km)
    df_merged["F_airport"] = np.exp(-df_merged["d_airport_km"] / sigma_km)
    
    df_merged["raw_spend_per_capita"] = df_merged["raw_spend"] / df_merged["P_population"]
    
    # Map column names required by TurismRadarRegressor and app.py
    df_merged["S_total_spend"] = df_merged["raw_spend"]
    df_merged["raw_spend_per_capita"] = df_merged["S_total_spend"] / df_merged["P_population"]
    
    return df_merged

if __name__ == "__main__":
    df = build_turismradar_dataset(sigma_km=25.0)
    print("\n--- Pipeline Execution Success ---")
    print(f"Total Rows: {len(df)}")
    print(df[["knkod", "LAU_NAME", "P_population", "G_overnight_stays", "F_border", "F_airport"]].head(10))
