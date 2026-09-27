import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# 1. Generate Airport Points (Arlanda, Landvetter)
airports = gpd.GeoDataFrame([
    {"name": "Arlanda Airport", "geometry": Point(17.9186, 59.6498)},
    {"name": "Landvetter Airport", "geometry": Point(12.2928, 57.6628)}
], crs="EPSG:4326")
airports.to_file("data/airports.geojson", driver="GeoJSON")

# 2. Generate Border Crossing Points (Svinesund, Charlottenberg)
borders = gpd.GeoDataFrame([
    {"name": "Svinesund Border", "geometry": Point(11.2681, 59.0983)},
    {"name": "Charlottenberg Border", "geometry": Point(12.2961, 59.8839)}
], crs="EPSG:4326")
borders.to_file("data/borders.geojson", driver="GeoJSON")

# 3. Generate Sample Municipal Polygons & Synthesized Tourism Metrics
def make_square(lon, lat, delta=0.1):
    return Polygon([
        (lon - delta, lat - delta),
        (lon + delta, lat - delta),
        (lon + delta, lat + delta),
        (lon - delta, lat + delta)
    ])

mock_kommuner = [
    # Normal Tourism Destinations
    {"kommun_code": "0180", "kommun_name": "Stockholm", "lat": 59.3293, "lon": 18.0686, "P": 980000, "G": 4500000, "C": 25000, "S": 12000000000},
    {"kommun_code": "0980", "kommun_name": "Gotland", "lat": 57.5000, "lon": 18.5000, "P": 61000, "G": 1200000, "C": 8000, "S": 2200000000},
    {"kommun_code": "2321", "kommun_name": "Åre", "lat": 63.3990, "lon": 13.0815, "P": 12000, "G": 950000, "C": 12000, "S": 1800000000},
    
    # High Anomaly / Noise Destinations
    {"kommun_code": "1486", "kommun_name": "Strömstad", "lat": 58.9383, "lon": 11.1736, "P": 13000, "G": 300000, "C": 3000, "S": 4500000000}, # High border spend
    {"kommun_code": "0191", "kommun_name": "Sigtuna", "lat": 59.6173, "lon": 17.7233, "P": 50000, "G": 400000, "C": 4000, "S": 3800000000},  # High airport transit spend
]

features = []
for k in mock_kommuner:
    poly = make_square(k["lon"], k["lat"])
    features.append({
        "kommun_code": k["kommun_code"],
        "kommun_name": k["kommun_name"],
        "P_population": k["P"],
        "G_overnight_stays": k["G"],
        "C_bed_capacity": k["C"],
        "S_total_spend": k["S"],
        "geometry": poly
    })

kommuner_gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
kommuner_gdf.to_file("data/kommuner.geojson", driver="GeoJSON")

print("✅ Synthetic datasets generated successfully in `data/`!")