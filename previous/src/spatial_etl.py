import numpy as np
import geopandas as gpd

def compute_distance_decay(
    gdf_kommuner: gpd.GeoDataFrame,
    gdf_points: gpd.GeoDataFrame,
    sigma: float = 25.0,
    target_crs: str = "EPSG:3006" # SWEREF99 TM
) -> gpd.GeoSeries:
    """
    Computes exponential distance decay factor F = exp(-d / sigma)
    from municipal centroids to nearest spatial point features (border/airports).
    """
    # Ensure metric projection
    kommuner_proj = gdf_kommuner.to_crs(target_crs)
    points_proj = gdf_points.to_crs(target_crs)
    
    # Extract centroids
    centroids = kommuner_proj.geometry.centroid
    
    # Calculate shortest distance in km to any target feature point
    min_distances_km = centroids.apply(
        lambda c: points_proj.distance(c).min() / 1000.0
    )
    
    # Calculate exponential distance decay factor
    decay_factor = np.exp(-min_distances_km / sigma)
    return min_distances_km, decay_factor


def build_spatial_feature_store(
    kommuner_gdf: gpd.GeoDataFrame,
    borders_gdf: gpd.GeoDataFrame,
    airports_gdf: gpd.GeoDataFrame,
    sigma: float = 25.0
) -> gpd.GeoDataFrame:
    """Builds spatial decay features for border and airport proximity."""
    df = kommuner_gdf.copy()
    
    df["d_border_km"], df["F_border"] = compute_distance_decay(df, borders_gdf, sigma)
    df["d_airport_km"], df["F_airport"] = compute_distance_decay(df, airports_gdf, sigma)
    
    return df