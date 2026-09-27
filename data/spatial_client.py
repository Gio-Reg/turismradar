import geopandas as gpd
from shapely.geometry import Point, MultiPoint

# EPSG:3006 (SWEREF99 TM) uses meters as base units
TARGET_CRS = "EPSG:3006"

# Primary Swedish border crossing points (NO/FI borders) & Major Transit Airports
AIRPORTS_WGS84 = {
    "Arlanda": (17.9186, 59.6498),
    "Landvetter": (12.2798, 57.6683),
    "Skavsta": (16.9122, 58.7886),
    "Malmö_Sturup": (13.3713, 55.5355),
}

BORDER_POINTS_WGS84 = [
    (11.2642, 59.0964), # Svinesund / Strömstad
    (12.2858, 59.8118), # Charlottenberg / Eda
    (12.2536, 61.1578), # Stöllet / Trysil corridor
    (24.1436, 65.8428), # Haparanda / Tornio
]

def load_municipal_geometries() -> gpd.GeoDataFrame:
    """
    Loads official Swedish municipal boundaries from SCB open geodata or Eurostat LAU.
    """
    url = "https://ec.europa.eu/eurostat/cache/GISCO/distribution/v2/lau/geojson/LAU_RG_01M_2023_4326.geojson"
    gdf = gpd.read_file(url)
    
    # Filter for Swedish municipalities (ISO country code SE)
    gdf_se = gdf[gdf["CNTR_CODE"] == "SE"].copy()
    gdf_se = gdf_se.to_crs(TARGET_CRS)
    
    # Calculate municipal centroids for spatial calculations
    gdf_se["centroid"] = gdf_se.geometry.centroid
    return gdf_se

def compute_spatial_distances(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Computes minimum distance in km from municipal centroid to nearest border and airport."""
    # Convert reference points to SWEREF99 TM (EPSG:3006)
    airports_gdf = gpd.GeoSeries(
        [Point(coords) for coords in AIRPORTS_WGS84.values()], 
        crs="EPSG:4326"
    ).to_crs(TARGET_CRS)
    
    borders_gdf = gpd.GeoSeries(
        [Point(coords) for coords in BORDER_POINTS_WGS84], 
        crs="EPSG:4326"
    ).to_crs(TARGET_CRS)
    
    airport_union = airports_gdf.union_all()
    border_union = borders_gdf.union_all()

    # Calculate shortest distance in kilometers (m / 1000)
    gdf["d_airport_km"] = gdf["centroid"].apply(lambda c: c.distance(airport_union) / 1000.0)
    gdf["d_border_km"] = gdf["centroid"].apply(lambda c: c.distance(border_union) / 1000.0)
    
    gdf = gdf.drop(columns=["centroid"])
    return gdf
