import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial.distance import cdist
from python_tsp.heuristics import solve_tsp_local_search
import warnings
import random
import networkx as nx
import networkit as nk
import os
# For colormaps
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import re 
import unicodedata


# Suppress warnings for cleaner output, especially from geopandas/shapely
warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
warnings.filterwarnings('ignore', 'The `geoseries.isna()` method is deprecated', UserWarning)
warnings.filterwarnings('ignore', 'The `geoseries.notna()` method is deprecated', UserWarning)
# Suppress MatplotlibDeprecationWarning for cm.get_cmap
warnings.filterwarnings('ignore', 'The get_cmap function was deprecated in Matplotlib', UserWarning)
# Suppress Geometry is in a geographic CRS warning for area calculation
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS. Results from \'area\' are likely incorrect.', UserWarning)
# Suppress Geometry is in a geographic CRS warning for centroid calculation
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS. Results from \'centroid\' are likely incorrect.', UserWarning)
# Suppress specific pyogrio/GDAL warning about SHX for robustness
os.environ['SHAPE_RESTORE_SHX'] = 'YES'


# --- GLOBAL CONFIGURATION (Update These Paths and Names) ---
WARD_NAME = "Alperton" # The specific ward you want to analyze
FIXED_CRIME_RATE_THRESHOLD = 0.40 # LSOAs with crime_rate >= this value will be considered high crime
MAX_TOTAL_WAYPOINTS_FOR_TSP = 20 # Maximum total waypoints to feed into the TSP solver
POINTS_PER_LSOA_MIN = 1 # Minimum points to generate for any high-crime LSOA
POINTS_PER_LSOA_MAX_CAP = 5 # Maximum points to generate for a single LSOA, regardless of size
LSOA_AREA_TO_POINT_FACTOR = 50000 # 1 point per this many square meters (e.g., 1 point per 0.05 sq km)
LSOA_MAJORITY_AREA_THRESHOLD = 0.1 # Percentage of LSOA area that must be within the ward (e.g., 0.1 for 10%)
OUTPUT_MAP_FILE = f"{WARD_NAME.lower().replace(' ', '_')}_patrol_route.html" # Dynamic output file name

# --- IMPORTANT FILE PATHS ---
PREDICTIONS_FILE_PATH = r'..\model\sample_predictions.parquet'
LSOA_WARD_LOOKUP_FILE_PATH = r'..\processed_data\LSOA_to_Ward_LAD_lookup.csv'
WARD_SHAPEFILE_PATH = r"..\data\London_Ward.shp"
# THIS IS CRUCIAL: Path to a shapefile containing ALL LSOA geometries
LSOA_BOUNDARIES_SHAPEFILE_PATH = r"..\data\Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4.gpkg"

# --- COLUMN NAMES FROM YOUR FILES ---
LSOA_COLUMN_NAME_IN_PREDICTIONS = 'index' # LSOA column name in sample_predictions.parquet
LSOA_COLUMN_NAME_FOR_MERGE = 'LSOA21CD' # Common LSOA column name (matches lookup CSV and LSOA boundaries shapefile)
CRIME_COUNT_COLUMN_NAME = 'median' # Column name for predicted crime count in predictions file ('median' as specified)
WARD_COLUMN_NAME_IN_LOOKUP = 'WD24NM' # Ward name column in LSOA_to_Ward_LAD_lookup.csv
WARD_COLUMN_NAME_IN_SHAPEFILE = 'NAME' # Ward name column in your London_Ward.shp shapefile
LSOA_ID_COLUMN_IN_LSOA_SHAPEFILE = "LSOA21CD" # LSOA ID column name in your LSOA_BOUNDARIES_SHAPEFILE_PATH


# --- 1. Initial Data Loading and Merging ---

print(f"--- Step 1: Loading and Merging DataFrames ---")

# Load predictions DataFrame
try:
    predictions_df = pd.read_parquet(PREDICTIONS_FILE_PATH)
    # Rename the LSOA column from its name in predictions_df to the common merge name
    if LSOA_COLUMN_NAME_IN_PREDICTIONS in predictions_df.columns:
        predictions_df = predictions_df.rename(columns={LSOA_COLUMN_NAME_IN_PREDICTIONS: LSOA_COLUMN_NAME_FOR_MERGE})
    elif predictions_df.index.name is None:
        predictions_df = predictions_df.reset_index()
        predictions_df = predictions_df.rename(columns={'index': LSOA_COLUMN_NAME_FOR_MERGE})
    elif predictions_df.index.name == LSOA_COLUMN_NAME_IN_PREDICTIONS:
        predictions_df = predictions_df.reset_index(names=[LSOA_COLUMN_NAME_FOR_MERGE])
    else:
        print(f"ERROR: LSOA column '{LSOA_COLUMN_NAME_IN_PREDICTIONS}' not found as a regular column or index in '{PREDICTIONS_FILE_PATH}'.")
        print("Please verify your 'sample_predictions.parquet' structure or adjust 'LSOA_COLUMN_NAME_IN_PREDICTIONS'.")
        exit()

    print(f"'{PREDICTIONS_FILE_PATH}' loaded and LSOA column transformed. Head:")
    print(predictions_df.head())
    print(f"Columns: {predictions_df.columns.tolist()}\n")

except ImportError:
    print("ERROR: 'pyarrow' or 'fastparquet' library not found. Please install one: `pip install pyarrow` or `pip install fastparquet`.")
    exit()
except FileNotFoundError:
    print(f"ERROR: '{PREDICTIONS_FILE_PATH}' not found. Please ensure the file exists at this path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading '{PREDICTIONS_FILE_PATH}': {e}")
    exit()

# Load LSOA to Ward lookup DataFrame
try:
    lsoa_ward_lookup_df = pd.read_csv(LSOA_WARD_LOOKUP_FILE_PATH)
    print(f"'{LSOA_WARD_LOOKUP_FILE_PATH}' loaded successfully. Head:")
    print(lsoa_ward_lookup_df.head())
    print(f"Columns: {lsoa_ward_lookup_df.columns.tolist()}\n")
except FileNotFoundError:
    print(f"ERROR: '{LSOA_WARD_LOOKUP_FILE_PATH}' not found. Please ensure the file exists at this path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading '{LSOA_WARD_LOOKUP_FILE_PATH}': {e}")
    exit()

# Validate essential columns exist before merging
if CRIME_COUNT_COLUMN_NAME not in predictions_df.columns:
    print(f"ERROR: Crime count column '{CRIME_COUNT_COLUMN_NAME}' not found in '{PREDICTIONS_FILE_PATH}'.")
    print(f"Available columns in predictions_df: {predictions_df.columns.tolist()}")
    exit()
if LSOA_COLUMN_NAME_FOR_MERGE not in lsoa_ward_lookup_df.columns:
    print(f"ERROR: LSOA column '{LSOA_COLUMN_NAME_FOR_MERGE}' not found in '{LSOA_WARD_LOOKUP_FILE_PATH}'.")
    print(f"Available columns in lsoa_ward_lookup_df: {lsoa_ward_lookup_df.columns.tolist()}")
    exit()
if WARD_COLUMN_NAME_IN_LOOKUP not in lsoa_ward_lookup_df.columns:
    print(f"ERROR: Ward column '{WARD_COLUMN_NAME_IN_LOOKUP}' not found in '{LSOA_WARD_LOOKUP_FILE_PATH}'.")
    print(f"Available columns in lsoa_ward_lookup_df: {lsoa_ward_lookup_df.columns.tolist()}")
    exit()

# Merge the predictions DataFrame with the LSOA to Ward lookup DataFrame
# This creates a DataFrame with LSOA codes, their median crime counts, and their associated ward names.
lsoa_crime_ward_data = pd.merge(
    predictions_df,
    lsoa_ward_lookup_df[[LSOA_COLUMN_NAME_FOR_MERGE, WARD_COLUMN_NAME_IN_LOOKUP]],
    on=LSOA_COLUMN_NAME_FOR_MERGE,
    how='inner'
)
print("LSOA crime and ward data merged successfully. Head:")
print(lsoa_crime_ward_data.head())
print(f"Columns: {lsoa_crime_ward_data.columns.tolist()}\n")


# --- 2. Geospatial Data Acquisition and Preprocessing ---

def download_area_boundary(area_name):
    """
    Downloads the boundary of the specified London area (ward) using OSMnx.
    NOTE: This uses OSMnx which might not be perfectly aligned with your local ward shapefile.
          It's kept for consistency with original code, but direct loading from WARD_SHAPEFILE_PATH
          is performed later for filtering LSOAs.
    Args:
        area_name (str): The name of the area to download.
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the area boundary.
    """
    print(f"--- Step 2.1: Downloading area boundary for: {area_name} from OpenStreetMap ---")
    try:
        area_gdf = ox.geocode_to_gdf(area_name)
        print(f"   Area boundary downloaded. CRS: {area_gdf.crs}")

        area_gdf_projected = area_gdf.to_crs(epsg=27700) # British National Grid for accurate area calculation
        # Print area in square meters for better precision for small wards
        print(f"   Approximate Area: {area_gdf_projected.geometry.area.sum():.0f} m² ({area_gdf_projected.geometry.area.sum() / 1e6:.2f} km²) (in projected CRS)")

        return area_gdf
    except Exception as e:
        print(f"Error downloading area boundary from OSMnx: {e}")
        print("This might happen if the area name is not well-defined in OpenStreetMap or there's no network connection.")
        return None

def load_and_filter_lsoa_boundaries(all_lsoas_with_crime_and_geo_gdf, area_gdf_osm, crime_value_column, majority_threshold=LSOA_MAJORITY_AREA_THRESHOLD):
    """
    Filters a pre-loaded GeoDataFrame of LSOA boundaries with crime data to the specified area (ward),
    retaining only LSOAs where a significant portion of their area is within the target area.
    Args:
        all_lsoas_with_crime_and_geo_gdf (geopandas.GeoDataFrame): GeoDataFrame of ALL LSOAs with 'geometry',
                                                                     and the crime_value_column (e.g., 'median'),
                                                                     and ward lookup info (WD24NM).
        area_gdf_osm (geopandas.GeoDataFrame): GeoDataFrame of the target ward boundary (from OSMnx).
        crime_value_column (str): The name of the column in all_lsoas_with_crime_and_geo_gdf
                                  that contains the crime values (e.g., 'median').
        majority_threshold (float): Percentage (0-1) of LSOA area that must be within the ward.
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame of LSOAs within the area with their actual crime rates.
    """
    area_name_for_log = area_gdf_osm['name'].iloc[0] if 'name' in area_gdf_osm.columns else "the specified area"
    print(f"--- Step 2.2: Filtering LSOA boundaries to {area_name_for_log} using pre-loaded crime data ---")

    try:
        # Ensure CRSs are consistent for initial spatial join with OSMnx boundary
        if all_lsoas_with_crime_and_geo_gdf.crs != area_gdf_osm.crs:
            all_lsoas_with_crime_and_geo_gdf = all_lsoas_with_crime_and_geo_gdf.to_crs(area_gdf_osm.crs)
            print(f"   Re-projected LSOAs to {area_gdf_osm.crs} for spatial operations.")

        # Initial spatial join to get LSOAs that intersect with the target area (ward from OSMnx)
        lsoas_intersecting_area_osm = gpd.sjoin(all_lsoas_with_crime_and_geo_gdf, area_gdf_osm, how="inner", predicate="intersects")
        lsoas_intersecting_area_osm = lsoas_intersecting_area_osm.reset_index(drop=True)
        lsoas_intersecting_area_osm = lsoas_intersecting_area_osm.drop(columns=[col for col in lsoas_intersecting_area_osm.columns if '_right' in col or 'index_' in col], errors='ignore')

        print(f"   Initial filter resulted in {len(lsoas_intersecting_area_osm)} LSOAs intersecting the OSMnx area.")

        if lsoas_intersecting_area_osm.empty:
            print("   No LSOAs found intersecting the specified OSMnx area after initial spatial join.")
            return gpd.GeoDataFrame(columns=[LSOA_COLUMN_NAME_FOR_MERGE, 'LSOA21NM', 'geometry', crime_value_column, 'crime_rate'], crs="EPSG:4326")

        # Further filter LSOAs where a majority of their area is within the target ward (OSMnx boundary)
        print(f"   Filtering LSOAs where less than {majority_threshold*100:.0f}% of area is within the OSMnx ward boundary...")

        area_polygon_projected = area_gdf_osm.to_crs(epsg=27700).geometry.iloc[0]
        lsoas_intersecting_area_projected = lsoas_intersecting_area_osm.to_crs(epsg=27700)

        filtered_lsoas_list = []
        for idx, row in lsoas_intersecting_area_projected.iterrows():
            lsoa_geom = row.geometry
            lsoa_original_area = lsoa_geom.area
            intersection_geom = lsoa_geom.intersection(area_polygon_projected)
            if intersection_geom.is_valid and not intersection_geom.is_empty:
                intersection_area = intersection_geom.area
                threshold = majority_threshold
                if len(lsoas_intersecting_area_projected) == 1 and intersection_area > 0:
                    threshold = 0.0
                if lsoa_original_area > 1e-9 and (intersection_area / lsoa_original_area) >= threshold:
                    filtered_lsoas_list.append(lsoas_intersecting_area_osm.iloc[row.name])

        if not filtered_lsoas_list:
            print(f"   After majority-area filtering (threshold: {majority_threshold*100:.0f}%), no LSOAs remain. Using all LSOAs that intersect the area.")
            lsoas_within_area_gdf = lsoas_intersecting_area_osm.to_crs(epsg=4326)
        else:
            lsoas_within_area_gdf = gpd.GeoDataFrame(filtered_lsoas_list, crs=lsoas_intersecting_area_osm.crs).to_crs(epsg=4326)
            print(f"   Filtered to {len(lsoas_within_area_gdf)} LSOAs after majority-area check.")

        lsoas_within_area_gdf['crime_rate'] = lsoas_within_area_gdf[crime_value_column].fillna(0).astype(float)
        print(f"   Assigned crime rates (from '{crime_value_column}' column) to {len(lsoas_within_area_gdf)} LSOAs.")

        return lsoas_within_area_gdf
    except Exception as e:
        print(f"Error loading or filtering LSOA boundaries: {e}")
        return gpd.GeoDataFrame(columns=[LSOA_COLUMN_NAME_FOR_MERGE, 'LSOA21NM', 'geometry', crime_value_column, 'crime_rate'], crs="EPSG:4326")

def extract_cycle_network(area_polygon):
    """
    Extracts the cycle network within the area boundary using OSMnx,
    prioritizing residential roads by heavily penalizing non-residential ones.
    Args:
        area_polygon (shapely.geometry.Polygon): The polygon of the area.
    Returns:
        networkx.MultiDiGraph: The cycle network graph.
    """
    print("--- Step 2.3: Extracting cycle network within the area, prioritizing residential roads ---")
    try:
        G = ox.graph_from_polygon(area_polygon, network_type='drive')
        G = ox.distance.add_edge_lengths(G)

        # Get the largest strongly connected component to ensure all nodes are reachable
        if len(G.nodes) > 0:
            components = list(nx.strongly_connected_components(G))
            if components:
                largest_component_nodes = max(components, key=len)
                G = G.subgraph(largest_component_nodes).copy()
                print(f"   Filtered to largest strongly connected component. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
            else:
                print("   Warning: No strongly connected components found in the graph after initial extraction.")
                return None
        else:
            print("   Graph has no nodes after initial extraction.")
            return None

        # --- MODIFIED LOGIC FOR PRIORITIZING RESIDENTIAL ROADS ---
        # Define types of roads considered residential or otherwise suitable for quieter cycle routes
        residential_like_highway_types = ['residential', 'living_street', 'unclassified', 'service', 'track', 'cycleway', 'path']
        # 'unclassified' and 'service' are often quieter local roads. 'cycleway' and 'path' are explicitly for bikes.
        
        # Define a high penalty factor for non-residential roads
        # A higher value (e.g., 50.0 or 100.0) will more strongly discourage using these roads.
        penalty_factor_non_residential = 30.0 

        for u, v, k, data in G.edges(keys=True, data=True):
            is_residential_like = False
            if 'highway' in data:
                # Handle cases where 'highway' tag might be a list (multiple types)
                highway_types = data['highway'] if isinstance(data['highway'], list) else [data['highway']]
                for highway_type in highway_types:
                    if highway_type in residential_like_highway_types:
                        is_residential_like = True
                        break # Found a residential-like type, no need to penalize
            
            # If the road is NOT residential-like, apply the heavy penalty
            if not is_residential_like:
                if 'length' in data: # Ensure edge has a length attribute
                    data['length'] *= penalty_factor_non_residential
                # else: If an edge has no length, it might be problematic.
                # The 'add_edge_lengths' should ideally cover most cases.

        print(f"   Cycle network extracted and prioritized for residential roads. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
        return G
    except Exception as e:
        print(f"Error extracting cycle network: {e}")
        return None
    
# Helper function to generate random points within a polygon
def generate_random_points_in_polygon(polygon, num_points):
    """
    Generates random points within a given shapely Polygon.
    Args:
        polygon (shapely.geometry.Polygon): The polygon to generate points within.
        num_points (int): The number of points to generate.
    Returns:
        list: A list of shapely Point objects.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []
    attempts = 0
    max_attempts_per_point = 100
    while len(points) < num_points and attempts < num_points * max_attempts_per_point:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            points.append(random_point)
        attempts += 1
    if len(points) < num_points:
        print(f"   Warning: Could not generate {num_points} points within polygon. Generated {len(points)}.")
    return points

def calculate_points_based_on_size(lsoa_area_sqm, min_points=POINTS_PER_LSOA_MIN, max_points_per_lsoa_cap=POINTS_PER_LSOA_MAX_CAP, area_scaling_factor=LSOA_AREA_TO_POINT_FACTOR):
    """
    Determines the number of points to generate for an LSOA based on its area.
    Args:
        lsoa_area_sqm (float): Area of the LSOA in square meters (projected CRS).
        min_points (int): Minimum number of points to generate.
        max_points_per_lsoa_cap (int): Maximum number of points to generate for a single LSOA.
        area_scaling_factor (float): Area (in sqm) per point.
    Returns:
        int: Number of points to generate for this LSOA.
    """
    points = int(lsoa_area_sqm / area_scaling_factor)
    points = max(min_points, points)
    points = min(max_points_per_lsoa_cap, points)
    return points

# --- Main function to simulate crime and select waypoints ---
def simulate_crime_and_select_waypoints(lsoas_gdf, G, max_total_waypoints_for_tsp, fixed_crime_rate_threshold):
    """
    Identifies high-crime LSOAs (based on median crime rate),
    and selects representative network nodes as patrol waypoints, ensuring
    a minimum of one point per LSOA, then adding more based on size/crime priority.
    Args:
        lsoas_gdf (geopandas.GeoDataFrame): GeoDataFrame of LSOAs with 'crime_rate' (which is 'median').
        G (networkx.MultiDiGraph): The cycle network graph.
        max_total_waypoints_for_tsp (int): Maximum total waypoints to select for TSP.
        fixed_crime_rate_threshold (float): LSOAs with crime_rate >= this value will be considered high crime.
    Returns:
        pandas.DataFrame: DataFrame of selected waypoints with their details.
        geopandas.GeoDataFrame: LSOAs GeoDataFrame with crime rates and high-crime flag.
    """
    print("--- Step 4: Selecting Patrol Waypoints Based on Median Crime and Area ---")
    
    # Identify high-crime LSOAs based on the fixed crime rate threshold for display/labeling purposes only.
    lsoas_gdf['is_high_crime'] = lsoas_gdf['crime_rate'] >= fixed_crime_rate_threshold
    
    # Sort by crime rate (descending) so that higher crime LSOAs are processed earlier for potential extra points
    lsoas_to_process = lsoas_gdf.sort_values(by='crime_rate', ascending=False)

    if lsoas_to_process.empty:
        print("    No LSOAs found in the ward to select waypoints from. Exiting waypoint selection.")
        return pd.DataFrame(), lsoas_gdf

    guaranteed_waypoints = []
    potential_additional_waypoints = []
    
    print(f"    Generating points for all {len(lsoas_to_process)} LSOAs in the ward...")
    
    for idx, lsoa_row in lsoas_to_process.iterrows():
        # Get LSOA geometry in projected CRS for accurate area calculation
        lsoa_polygon_projected = gpd.GeoSeries([lsoa_row.geometry], crs=lsoas_gdf.crs).to_crs(epsg=27700).iloc[0]
        lsoa_area_sqm = lsoa_polygon_projected.area
        
        # Ensure the LSOA geometry is valid before generating points (use original CRS for random point generation)
        lsoa_polygon_geographic = lsoa_row.geometry
        if not lsoa_polygon_geographic.is_valid:
            lsoa_polygon_geographic = lsoa_polygon_geographic.buffer(0) # Attempt to fix invalid geometry

        if lsoa_polygon_geographic.is_empty:
            print(f"    Warning: LSOA ({lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]}) has an empty or invalid geometry. Skipping point generation.")
            continue

        # --- 1. GUARANTEE AT LEAST ONE POINT PER LSOA ---
        # Generate 1 point for this LSOA
        guaranteed_random_points = generate_random_points_in_polygon(lsoa_polygon_geographic, POINTS_PER_LSOA_MIN)
        
        if guaranteed_random_points: # Check if at least one point was successfully generated
            point = guaranteed_random_points[0] # Take the first generated point
            nearest_node = ox.distance.nearest_nodes(G, point.x, point.y)
            if nearest_node is not None and nearest_node in G.nodes:
                guaranteed_waypoints.append({
                    'LSOA21CD': lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE],
                    'LSOA21NM': lsoa_row['LSOA21NM'] + ' (Guaranteed Pt)' if 'LSOA21NM' in lsoa_row else f"LSOA {lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]} (Guaranteed Pt)",
                    'simulated_crime_rate': lsoa_row['crime_rate'],
                    'waypoint_lat': G.nodes[nearest_node]['y'],
                    'waypoint_lon': G.nodes[nearest_node]['x'],
                    'nearest_node_id': nearest_node
                })
            else:
                print(f"    Warning: Could not find valid nearest network node for guaranteed point in LSOA {lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]}. Skipping guaranteed point for this LSOA.")
        else:
            print(f"    Warning: Could not generate a guaranteed point within polygon for LSOA {lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]}. Skipping guaranteed point for this LSOA.")

        # --- 2. CALCULATE AND GENERATE ADDITIONAL POINTS BASED ON SIZE ---
        # Calculate total points this LSOA should get based on size, respecting the max cap
        num_points_by_size = calculate_points_based_on_size(lsoa_area_sqm)
        
        # Determine how many *additional* points are needed, beyond the guaranteed one(s)
        num_additional_points = num_points_by_size - POINTS_PER_LSOA_MIN 
        
        if num_additional_points > 0:
            additional_random_points = generate_random_points_in_polygon(lsoa_polygon_geographic, num_additional_points)
            for i, point in enumerate(additional_random_points):
                nearest_node = ox.distance.nearest_nodes(G, point.x, point.y)
                if nearest_node is not None and nearest_node in G.nodes:
                    potential_additional_waypoints.append({
                        'LSOA21CD': lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE],
                        'LSOA21NM': lsoa_row['LSOA21NM'] + f' (Addl Pt {i+1})' if 'LSOA21NM' in lsoa_row else f"LSOA {lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]} (Addl Pt {i+1})",
                        'simulated_crime_rate': lsoa_row['crime_rate'],
                        'waypoint_lat': G.nodes[nearest_node]['y'],
                        'waypoint_lon': G.nodes[nearest_node]['x'],
                        'nearest_node_id': nearest_node
                    })
                else:
                    print(f"    Warning: Could not find valid nearest network node for additional point in LSOA {lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]}. Skipping this additional waypoint.")

    # --- 3. COMBINE AND APPLY FINAL CAP ---
    final_patrol_waypoints = pd.DataFrame(guaranteed_waypoints)
    
    # Calculate how many more waypoints we can add up to the total max budget
    remaining_capacity = max_total_waypoints_for_tsp - len(final_patrol_waypoints)
    
    if remaining_capacity > 0 and potential_additional_waypoints:
        potential_additional_df = pd.DataFrame(potential_additional_waypoints)
        # Prioritize additional points by crime rate before adding them
        potential_additional_df = potential_additional_df.sort_values(by='simulated_crime_rate', ascending=False)
        
        # Take only as many additional points as the remaining capacity allows
        selected_additional_df = potential_additional_df.head(remaining_capacity)
        
        final_patrol_waypoints = pd.concat([final_patrol_waypoints, selected_additional_df], ignore_index=True)
        print(f"    Added {len(selected_additional_df)} additional waypoints based on crime priority. Remaining capacity: {remaining_capacity}.")
    elif remaining_capacity <= 0:
        print(f"    No capacity for additional waypoints. Reached MAX_TOTAL_WAYPOINTS_FOR_TSP ({max_total_waypoints_for_tsp}) with guaranteed points.")
    else:
        print("    No additional waypoints generated.")


    print(f"    Selected a total of {len(final_patrol_waypoints)} patrol waypoints for TSP (including guaranteed points).")
    return final_patrol_waypoints, lsoas_gdf

# --- 3. Designing the Optimal Cycle Patrol Route ---

def calculate_optimal_route(G, waypoints_df):
    """
    Calculates the optimal cycle patrol route using TSP.
    Args:
        G (networkx.MultiDiGraph): The cycle network graph.
        waypoints_df (pandas.DataFrame): DataFrame of selected waypoints.
    Returns:
        list: Ordered list of node IDs forming the optimal TSP tour.
        float: Total distance of the optimal tour.
        list: List of lists of node IDs for each segment of the route.
    """
    print("5. Calculating optimal cycle patrol route using TSP...")
    
    node_ids = waypoints_df['nearest_node_id'].tolist()
    num_nodes = len(node_ids)

    if num_nodes < 2:
        print("   Not enough waypoints to create a route.")
        return [], 0, []

    # NetworKit for fast CPU parallel shortest paths
    print("   Using NetworKit for fast parallel shortest path calculation (CPU).")
    # Map node ids to indices
    node_id_to_idx = {nid: i for i, nid in enumerate(G.nodes())}
    idx_to_node_id = {i: nid for nid, i in node_id_to_idx.items()}
    nkG = nk.Graph(len(G.nodes()), weighted=True, directed=False)
    for u, v, data in G.edges(data=True):
        w = float(data.get('length', 1.0))
        nkG.addEdge(node_id_to_idx[u], node_id_to_idx[v], w)
    # Compute all-pairs shortest paths for waypoints
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i, src in enumerate(node_ids):
        apsp = nk.distance.Dijkstra(nkG, node_id_to_idx[src], storePaths=False)
        apsp.run()
        for j, dst in enumerate(node_ids):
            dist = apsp.distance(node_id_to_idx[dst])
            distance_matrix[i, j] = dist if dist < float('inf') else np.inf

    # Handle unreachable waypoints by making their distance very high but not infinite for TSP solver
    if np.isinf(distance_matrix).any():
        print("   Warning: Some waypoints are not reachable from each other. TSP might fail or give suboptimal results.")
        # Replace inf with a large number based on existing max distance
        max_finite_dist = distance_matrix[~np.isinf(distance_matrix)].max()
        # Ensure max_finite_dist is not 0 if all distances are 0 or inf
        replacement_val = max_finite_dist * 2 if max_finite_dist > 0 else 1e9
        distance_matrix[np.isinf(distance_matrix)] = replacement_val
    
    print("   Distance matrix calculated.")

    permutation, distance = solve_tsp_local_search(distance_matrix)

    
    # The TSP solver returns an order of indices; convert back to original node IDs
    optimal_node_sequence_waypoints = [node_ids[i] for i in permutation]
    # Add the start waypoint at the end to complete the loop
    optimal_node_sequence_waypoints.append(optimal_node_sequence_waypoints[0])

    route_segments_detailed = [] # Store detailed node sequences for each segment
    total_route_distance = 0
    
    # Reconstruct the full detailed path for visualization and total distance
    for i in range(len(optimal_node_sequence_waypoints) - 1):
        try:
            # Get the full list of nodes for the shortest path between current and next waypoint
            path_nodes = ox.shortest_path(G, optimal_node_sequence_waypoints[i], optimal_node_sequence_waypoints[i+1], weight='length')
            if path_nodes:
                route_segments_detailed.append(path_nodes)
                segment_length = 0
                for k in range(len(path_nodes) - 1):
                    u, v = path_nodes[k], path_nodes[k+1]
                    if 0 in G[u][v] and 'length' in G[u][v][0]:
                        segment_length += G[u][v][0]['length']
                    else:
                        # Fallback for missing length during reconstruction, same as distance matrix
                        try:
                            segment_length += ox.distance.great_circle_vec(
                                G.nodes[u]['y'], G.nodes[u]['x'], G.nodes[v]['y'], G.nodes[v]['x']
                            )
                        except KeyError:
                            segment_length += 0 # Or some error handling
                total_route_distance += segment_length
            else:
                # If path_nodes is None, it means no path was found even after connected component filtering.
                # This should ideally not happen if waypoints are selected from the main component.
                # If it does, it indicates a problem with the graph or waypoint selection.
                print(f"   Error: No path found between {optimal_node_sequence_waypoints[i]} and {optimal_node_sequence_waypoints[i+1]} during route reconstruction. This segment will appear as a straight line.")
                # To avoid a straight line, we could add a direct line segment as a last resort,
                # but the user explicitly wants to avoid this. The best fix is to ensure connectivity.
                # For now, we'll just log and continue, which might lead to a visual gap or straight line.
                # A more robust solution would be to re-evaluate waypoint selection or graph connectivity.
        except Exception as e:
            print(f"   Error reconstructing path segment between {optimal_node_sequence_waypoints[i]} and {optimal_node_sequence_waypoints[i+1]}: {e}")
            
    print(f"   Optimal route calculated. Total distance: {total_route_distance:.2f} meters.")
    return optimal_node_sequence_waypoints, total_route_distance, route_segments_detailed

# --- 4. Interactive Visualization of the Patrol Route ---
def visualize_route(area_gdf, lsoas_gdf, waypoints_df, G, optimal_node_sequence, route_segments, output_file_raw_name, location_name):
    """
    Creates an interactive Folium map to visualize the area, LSOAs, waypoints, and patrol route.
    Args:
        area_gdf (geopandas.GeoDataFrame): Area boundary (the target ward).
        lsoas_gdf (geopandas.GeoDataFrame): LSOAs within the ward with crime rates (median).
        waypoints_df (pandas.DataFrame): Selected waypoints.
        G (networkx.MultiDiGraph): Cycle network graph.
        optimal_node_sequence (list): Ordered list of node IDs for the route (just waypoints).
        route_segments (list): List of lists of node IDs representing each detailed segment of the route.
        output_file_raw_name (str): The raw ward name or desired base filename (will be sanitized).
    """
    print("--- Step 6: Visualizing the patrol route on an interactive map ---")
    
    # Sanitize the output file name before saving
    # 1. Convert to lowercase
    sanitized_name = output_file_raw_name.lower()
    # 2. Replace spaces with underscores
    sanitized_name = sanitized_name.replace(' ', '_')
    # 3. Remove any characters that are NOT lowercase letters, numbers, or underscores
    # This will effectively remove apostrophes and other special symbols.
    sanitized_name = re.sub(r'[^a-z0-9_]', '', sanitized_name) 
    
    # Construct the final sanitized filename
    final_output_filename = f"{sanitized_name}_patrol_route.html"

    area_gdf_projected = area_gdf.to_crs(epsg=27700) # British National Grid
    area_centroid_projected = area_gdf_projected.geometry.centroid.iloc[0]
    area_centroid_geographic = gpd.GeoSeries([area_centroid_projected], crs="EPSG:27700").to_crs(epsg=4326).iloc[0]
    m = folium.Map(location=[area_centroid_geographic.y, area_centroid_geographic.x], zoom_start=14, tiles='cartodbpositron')

    # Determine ward name for display on the map (handles global WARD_NAME or fallbacks)
    try:
        # Assumes WARD_NAME is a global constant defined elsewhere in your script
        ward_name_for_display = WARD_NAME 
    except NameError:
        # Fallback if WARD_NAME is not globally defined, reconstruct from sanitized name
        ward_name_for_display = sanitized_name.replace('_', ' ').title() 

    # Add Area Boundary (the specific ward)
    folium.GeoJson(
        area_gdf.to_json(),
        name=f'{ward_name_for_display} Boundary',
        style_function=lambda x: {
            'fillColor': '#cccccc',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.1
        }
    ).add_to(m)

    # Add LSOA Choropleth based on 'crime_rate' (which is now the median crime count)
    lsoas_gdf_wgs84 = lsoas_gdf.to_crs(epsg=4326)

    min_crime = lsoas_gdf_wgs84['crime_rate'].min()
    max_crime = lsoas_gdf_wgs84['crime_rate'].max()
    
    if min_crime == max_crime:
        # If all crime rates are the same, use a default range for normalization
        norm = matplotlib.colors.Normalize(vmin=min_crime - 0.1, vmax=max_crime + 0.1) if min_crime != 0 else matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    else:
        norm = matplotlib.colors.Normalize(vmin=min_crime, vmax=max_crime)

    # Update cmap call to use the modern Matplotlib API
    cmap = matplotlib.colormaps.get_cmap('YlOrRd')

    def style_function(feature):
        crime_rate = feature['properties']['crime_rate']
        return {
            'fillColor': matplotlib.colors.rgb2hex(cmap(norm(crime_rate))),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        }

    # CRIME_COUNT_COLUMN_NAME is assumed to be a global constant
    folium.GeoJson(
        lsoas_gdf_wgs84.to_json(),
        name=f'LSOA Median Crime Rates for {ward_name_for_display}',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['LSOA21CD', 'LSOA21NM', 'crime_rate'], aliases=['LSOA Code:', 'LSOA Name:', f'Median Crime Rate (from "{CRIME_COUNT_COLUMN_NAME}"):'])
    ).add_to(m)

    # Add Patrol Waypoints
    waypoint_group = folium.FeatureGroup(name='Patrol Waypoints').add_to(m)
    for idx, row in waypoints_df.iterrows():
        is_start_point = (row['nearest_node_id'] == optimal_node_sequence[0])
        if is_start_point:
            icon = folium.Icon(color='green', icon='play', prefix='fa')
        else:
            icon = folium.Icon(color='blue', icon='circle', prefix='fa')

        folium.Marker(
            location=[row['waypoint_lat'], row['waypoint_lon']],
            popup=f"<b>LSOA:</b> {row['LSOA21NM']}<br><b>Median Crime:</b> {row['simulated_crime_rate']:.2f}",
            icon=icon
        ).add_to(waypoint_group)

    # Add Optimized Cycle Route - ensuring it follows the network paths
    full_route_latlons = []
    if route_segments:
        # Concatenate all route segments into a single list of node IDs
        if route_segments[0]: # Ensure the first segment exists
            for node_id in route_segments[0]:
                if node_id in G.nodes:
                    full_route_latlons.append([G.nodes[node_id]['y'], G.nodes[node_id]['x']])
                else:
                    print(f"Warning: Node ID {node_id} not found in graph G during final route coordinates extraction (first segment).")

        for i in range(1, len(route_segments)):
            if route_segments[i]: # Ensure subsequent segments exist
                # [1:] to skip the duplicate start node of the segment (which is the end of the previous)
                for node_id in route_segments[i][1:]: 
                    if node_id in G.nodes:
                        full_route_latlons.append([G.nodes[node_id]['y'], G.nodes[node_id]['x']])
                    else:
                        print(f"Warning: Node ID {node_id} not found in graph G during final route coordinates extraction (segment {i}).")
            else:
                print(f"Warning: Route segment {i} is empty or None. This might cause a visual gap or straight line.")

    if full_route_latlons:
        folium.PolyLine(
            full_route_latlons,
            color='red',
            weight=5,
            opacity=0.8,
            name='Optimized Cycle Patrol Route'
        ).add_to(m)

    # Add Layer Control
    folium.LayerControl().add_to(m)

    WARD_ROUTE_DIR = "ward_routes"

    # Save the map with the sanitized filename
    m.save(os.path.join(WARD_ROUTE_DIR, f"{location_name.replace(' ', '_').lower()}.html"))
    print(f"   Interactive map saved to {final_output_filename}")


def normalize_ward_name(name):
    """
    Normalize ward names for robust matching:
    - Lowercase
    - Remove punctuation (including apostrophes, periods)
    - Remove extra spaces
    """
    if not isinstance(name, str):
        return ""
    # Remove accents
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    # Lowercase
    name = name.lower()
    # Remove punctuation (keep only alphanumerics and spaces)
    name = re.sub(r'[^a-z0-9 ]', '', name)
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    # Strip leading/trailing spaces
    name = name.strip()
    return name

def load_ward_boundary_from_shapefile(ward_name):
    """
    Loads the ward boundary from a local shapefile.
    Args:
        ward_name (str): The name of the ward to load.
    Returns:
        geopandas.GeoDataFrame: The GeoDataFrame containing the ward boundary.
    """
    print(f"--- Step 1: Loading Ward Boundary for '{ward_name}' from local shapefile ---")
    try:
        all_wards_gdf = gpd.read_file(WARD_SHAPEFILE_PATH)
        # Normalize all ward names in the shapefile
        all_wards_gdf['_normalized_name'] = all_wards_gdf[WARD_COLUMN_NAME_IN_SHAPEFILE].apply(normalize_ward_name)
        # Normalize the input ward name
        normalized_input = normalize_ward_name(ward_name)
        # Filter using normalized names
        selected_ward_gdf = all_wards_gdf[all_wards_gdf['_normalized_name'] == normalized_input].copy()

        if selected_ward_gdf.empty:
            print(f"ERROR: Ward '{ward_name}' not found in '{WARD_SHAPEFILE_PATH}' under column '{WARD_COLUMN_NAME_IN_SHAPEFILE}'.")
            print(f"Available ward names (first 10): {all_wards_gdf[WARD_COLUMN_NAME_IN_SHAPEFILE].unique()[:10]}")
            # Print normalized names for debugging
            print("Normalized ward names (first 10):", all_wards_gdf['_normalized_name'].unique()[:10])
            return None
        
        # Ensure single geometry for area_polygon and correct CRS
        if not selected_ward_gdf.crs: # If CRS is not set, assume WGS84
            selected_ward_gdf.crs = "EPSG:4326"
        selected_ward_gdf = selected_ward_gdf.to_crs(epsg=4326) # Ensure WGS84 for OSMnx/Folium consistency

        # For OSMnx, we often use a single polygon. If the ward is multi-part, dissolve it.
        area_polygon_for_osmnx = selected_ward_gdf.geometry.unary_union
        # If unary_union results in a GeometryCollection, pick the largest polygon if possible
        if area_polygon_for_osmnx.geom_type == 'GeometryCollection':
            polygons = [geom for geom in area_polygon_for_osmnx.geoms if geom.geom_type == 'Polygon']
            if polygons:
                area_polygon_for_osmnx = max(polygons, key=lambda p: p.area)
            else:
                print("Warning: Ward geometry is a GeometryCollection with no polygons. Cannot extract network.")
                return None
        elif area_polygon_for_osmnx.geom_type == 'MultiPolygon':
            area_polygon_for_osmnx = area_polygon_for_osmnx.convex_hull # Or just pick the largest part

        if area_polygon_for_osmnx is None:
            print("Failed to get a valid polygon for the ward. Exiting.")
            return None
        print(f"Ward '{ward_name}' boundary loaded from local shapefile.")
        return selected_ward_gdf
    except Exception as e:
        print(f"ERROR loading ward boundary from shapefile: {e}")
        return None

def load_lsoa_crime_data():
    """
    Loads LSOA boundaries and merges them with crime data.
    Returns:
        geopandas.GeoDataFrame: LSOAs with crime data merged.
    """
    print("--- Step 2.1: Loading LSOA boundaries and merging with crime data ---")
    try:
        # Load LSOA boundaries from the local shapefile
        all_lsoas_gdf = gpd.read_file(LSOA_BOUNDARIES_SHAPEFILE_PATH)
        all_lsoas_with_crime_and_geo_gdf = all_lsoas_gdf.merge(
            lsoa_crime_ward_data, 
            left_on=LSOA_COLUMN_NAME_FOR_MERGE, 
            right_on=LSOA_COLUMN_NAME_FOR_MERGE, 
            how='left'
        )

        if all_lsoas_with_crime_and_geo_gdf.empty:
            print("ERROR: No LSOAs found in the shapefile or no matching crime data. Exiting.")
            return None

        # Ensure the GeoDataFrame is in WGS84 for consistency with OSMnx/Folium
        all_lsoas_with_crime_and_geo_gdf = all_lsoas_with_crime_and_geo_gdf.to_crs(epsg=4326)
        
        # Calculate median crime rate for each LSOA
        all_lsoas_with_crime_and_geo_gdf['crime_rate'] = all_lsoas_with_crime_and_geo_gdf[CRIME_COUNT_COLUMN_NAME].fillna(0).astype(float)

        print(f"   Loaded {len(all_lsoas_with_crime_and_geo_gdf)} LSOAs with crime data.")
        return all_lsoas_with_crime_and_geo_gdf
    except Exception as e:
        print(f"Error loading or merging LSOA boundaries with crime data: {e}")
        return None

def process_location(ward_name):
    """
    Main function to process the specified ward, extract cycle network,
    filter LSOAs, simulate crime, select waypoints, calculate optimal route,
    and visualize the results.
    Args:
        ward_name (str): The name of the ward to process.
    """
    global WARD_NAME
    WARD_NAME = ward_name

    # Load the ward boundary from local shapefile
    selected_ward_gdf = load_ward_boundary_from_shapefile(WARD_NAME)

    if selected_ward_gdf is None:
        print(f"ERROR: Could not load ward boundary for '{WARD_NAME}'. Exiting.")
        return

    # Load LSOA boundaries and merge with crime data
    lsoa_crime_ward_data = load_lsoa_crime_data()

    if lsoa_crime_ward_data is None or lsoa_crime_ward_data.empty:
        print("ERROR: No LSOA crime data available. Exiting.")
        return

    # Filter LSOAs within the selected ward
    lsoas_in_ward_gdf = load_and_filter_lsoa_boundaries(lsoa_crime_ward_data, selected_ward_gdf, CRIME_COUNT_COLUMN_NAME)

    if lsoas_in_ward_gdf.empty:
        print("No LSOAs found within the specified ward with relevant crime data. Exiting.")
        return

    # Extract cycle network for the ward using OSMnx
    area_polygon_for_osmnx = selected_ward_gdf.geometry.unary_union
    G = extract_cycle_network(area_polygon_for_osmnx)

    if G is None:
        print("Failed to extract cycle network. Exiting.")
        return

    # Simulate crime and select waypoints based on median crime rates
    patrol_waypoints_df, lsoas_with_crime_rates = simulate_crime_and_select_waypoints(
        lsoas_in_ward_gdf, G, MAX_TOTAL_WAYPOINTS_FOR_TSP, FIXED_CRIME_RATE_THRESHOLD
    )

    if patrol_waypoints_df.empty:
        print("No patrol waypoints generated. Exiting.")
        return

    # Calculate the optimal route using TSP
    optimal_node_sequence, total_distance, route_segments = calculate_optimal_route(G, patrol_waypoints_df)

    if not optimal_node_sequence:
        print("No valid route could be calculated. Exiting.")
        return

    # Visualize the route on an interactive map
    visualize_route(
        selected_ward_gdf,
        lsoas_with_crime_rates,
        patrol_waypoints_df,
        G,
        optimal_node_sequence,
        route_segments,
        WARD_NAME,
        WARD_NAME.replace(' ', '_').lower()  # Use ward name for the output file
    )

# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Starting Patrol Route Generation Script ---")

    # Step 1: Load the specific ward boundary from your local shapefile
    print(f"\n--- Step 1: Loading Ward Boundary for '{WARD_NAME}' ---")
    try:
        all_wards_gdf = gpd.read_file(WARD_SHAPEFILE_PATH)
        selected_ward_gdf = all_wards_gdf[all_wards_gdf[WARD_COLUMN_NAME_IN_SHAPEFILE] == WARD_NAME].copy()

        if selected_ward_gdf.empty:
            print(f"ERROR: Ward '{WARD_NAME}' not found in '{WARD_SHAPEFILE_PATH}' under column '{WARD_COLUMN_NAME_IN_SHAPEFILE}'.")
            print(f"Available ward names (first 10): {all_wards_gdf[WARD_COLUMN_NAME_IN_SHAPEFILE].unique()[:10]}")

            # print the sum of unique values in the all_wards_gdf for debugging
            print(all_wards_gdf[WARD_COLUMN_NAME_IN_SHAPEFILE].nunique())
            exit()
        
        # Ensure single geometry for area_polygon and correct CRS
        if not selected_ward_gdf.crs: # If CRS is not set, assume WGS84
            selected_ward_gdf.crs = "EPSG:4326"
        selected_ward_gdf = selected_ward_gdf.to_crs(epsg=4326) # Ensure WGS84 for OSMnx/Folium consistency

        # For OSMnx, we often use a single polygon. If the ward is multi-part, dissolve it.
        area_polygon_for_osmnx = selected_ward_gdf.geometry.unary_union
        # If unary_union results in a GeometryCollection, pick the largest polygon if possible
        if area_polygon_for_osmnx.geom_type == 'GeometryCollection':
            polygons = [geom for geom in area_polygon_for_osmnx.geoms if geom.geom_type == 'Polygon']
            if polygons:
                area_polygon_for_osmnx = max(polygons, key=lambda p: p.area)
            else:
                print("Warning: Ward geometry is a GeometryCollection with no polygons. Cannot extract network.")
                area_polygon_for_osmnx = None
        elif area_polygon_for_osmnx.geom_type == 'MultiPolygon':
            area_polygon_for_osmnx = area_polygon_for_osmnx.convex_hull # Or just pick the largest part
            # For simplicity, taking convex_hull. For precise network extraction, consider iterating parts.

        if area_polygon_for_osmnx is None:
             print("Failed to get a valid polygon for the ward. Exiting.")
             exit()

        print(f"Ward '{WARD_NAME}' boundary loaded from local shapefile.")
        
    except Exception as e:
        print(f"ERROR loading ward shapefile or processing its geometry: {e}")
        exit()

    # Load all LSOA boundaries and merge with crime and ward data
    print(f"\n--- Step 2: Loading All LSOA Geometries and Merging with Crime Data ---")
    try:
        all_lsoa_geo_df = gpd.read_file(LSOA_BOUNDARIES_SHAPEFILE_PATH)
        if LSOA_ID_COLUMN_IN_LSOA_SHAPEFILE != LSOA_COLUMN_NAME_FOR_MERGE:
            all_lsoa_geo_df = all_lsoa_geo_df.rename(columns={LSOA_ID_COLUMN_IN_LSOA_SHAPEFILE: LSOA_COLUMN_NAME_FOR_MERGE})
        print(f"All LSOA geometries loaded from {LSOA_BOUNDARIES_SHAPEFILE_PATH}.")

        # Merge LSOA geometries with their median crime data and ward information
        lsoas_with_crime_and_geo_gdf = pd.merge(
            all_lsoa_geo_df,
            lsoa_crime_ward_data,
            on=LSOA_COLUMN_NAME_FOR_MERGE,
            how='inner'
        )
        print("All LSOA geometries merged with crime and ward data. Head:")
        print(lsoas_with_crime_and_geo_gdf.head())
        print(f"Columns: {lsoas_with_crime_and_geo_gdf.columns.tolist()}\n")

    except Exception as e:
        print(f"ERROR loading or merging LSOA boundaries and crime data: {e}")
        exit()


    # Pass the pre-merged GeoDataFrame to the filtering function
    # This will filter LSOAs specific to the target ward and prepare 'crime_rate'
    lsoas_in_ward_gdf = load_and_filter_lsoa_boundaries(
        lsoas_with_crime_and_geo_gdf, # Pass the enriched GeoDataFrame
        selected_ward_gdf,            # Pass the selected ward's GeoDataFrame (from local file)
        CRIME_COUNT_COLUMN_NAME,      # Specify the column holding median crime (now 'median')
        LSOA_MAJORITY_AREA_THRESHOLD
    )
    
    if lsoas_in_ward_gdf.empty:
        print("No LSOAs found within the specified ward with relevant crime data. Exiting.")
    else:
        # Extract cycle network using the OSMnx polygon for the ward
        # Note: Using OSMnx's polygon for network extraction, which might differ slightly from your local ward shapefile.
        # This is a common practice for street networks.
        G = extract_cycle_network(area_polygon_for_osmnx)
        if G is None:
            print("Failed to extract cycle network. Exiting.")
        else:
            patrol_waypoints_df, lsoas_with_crime_gdf_final = simulate_crime_and_select_waypoints(
                lsoas_in_ward_gdf, G, MAX_TOTAL_WAYPOINTS_FOR_TSP, FIXED_CRIME_RATE_THRESHOLD
            )
            
            if patrol_waypoints_df.empty:
                print("No patrol waypoints selected. Exiting.")
            else:
                optimal_node_sequence, total_route_distance, route_segments = calculate_optimal_route(G, patrol_waypoints_df)

                if not optimal_node_sequence:
                    print("No optimal route could be calculated. Exiting.")
                else:
                    # Use the OSMnx area_gdf for visualization as it was used for network extraction
                    visualize_route(selected_ward_gdf, lsoas_with_crime_gdf_final, patrol_waypoints_df, G, optimal_node_sequence, route_segments, OUTPUT_MAP_FILE, WARD_NAME)

    print("\n--- Script Finished ---")