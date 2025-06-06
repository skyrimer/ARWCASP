import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial.distance import cdist
from python_tsp.exact import solve_tsp_dynamic_programming
import warnings
import random
import networkx as nx # Added for graph component analysis
import os
# For colormaps
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import re 


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
WARD_NAME = "Alexandra" # The specific ward you want to analyze
FIXED_CRIME_RATE_THRESHOLD = 0.40 # LSOAs with crime_rate >= this value will be considered high crime
MAX_TOTAL_WAYPOINTS_FOR_TSP = 20 # Maximum total waypoints to feed into the TSP solver
POINTS_PER_LSOA_MIN = 1 # Minimum points to generate for any high-crime LSOA
POINTS_PER_LSOA_MAX_CAP = 5 # Maximum points to generate for a single LSOA, regardless of size
LSOA_AREA_TO_POINT_FACTOR = 50000 # 1 point per this many square meters (e.g., 1 point per 0.05 sq km)
LSOA_MAJORITY_AREA_THRESHOLD = 0.1 # Percentage of LSOA area that must be within the ward (e.g., 0.1 for 10%)
OUTPUT_MAP_FILE = f"{WARD_NAME.lower().replace(' ', '_')}_patrol_route.html" # Dynamic output file name

# --- IMPORTANT FILE PATHS ---
PREDICTIONS_FILE_PATH = r'C:\Users\20231086\ARWCASP-4\model\sample_predictions.parquet'
LSOA_WARD_LOOKUP_FILE_PATH = r'C:\Users\20231086\ARWCASP-4\processed_data\LSOA_to_Ward_LAD_lookup.csv'
WARD_SHAPEFILE_PATH = r"C:\Users\20231086\ARWCASP-4\data\London_Ward.shp"
# THIS IS CRUCIAL: Path to a shapefile containing ALL LSOA geometries
LSOA_BOUNDARIES_SHAPEFILE_PATH = r"C:\Users\20231086\ARWCASP-4\data\Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4.gpkg"

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
        
        # Drop duplicate columns from the join
        lsoas_intersecting_area_osm = lsoas_intersecting_area_osm.drop(columns=[col for col in lsoas_intersecting_area_osm.columns if '_right' in col or 'index_' in col], errors='ignore')

        print(f"   Initial filter resulted in {len(lsoas_intersecting_area_osm)} LSOAs intersecting the OSMnx area.")

        if lsoas_intersecting_area_osm.empty:
            print("   No LSOAs found intersecting the specified OSMnx area after initial spatial join.")
            # Ensure the returned GeoDataFrame has the expected columns for downstream processes
            return gpd.GeoDataFrame(columns=[LSOA_COLUMN_NAME_FOR_MERGE, 'LSOA21NM', 'geometry', crime_value_column, 'crime_rate'], crs="EPSG:4326")

        # Further filter LSOAs where a majority of their area is within the target ward (OSMnx boundary)
        print(f"   Filtering LSOAs where less than {majority_threshold*100:.0f}% of area is within the OSMnx ward boundary...")
        
        # Project both for accurate area and intersection calculations (using British National Grid)
        area_polygon_projected = area_gdf_osm.to_crs(epsg=27700).geometry.iloc[0]
        lsoas_intersecting_area_projected = lsoas_intersecting_area_osm.to_crs(epsg=27700)

        filtered_lsoas_list = []
        for idx, row in lsoas_intersecting_area_projected.iterrows():
            lsoa_geom = row.geometry
            lsoa_original_area = lsoa_geom.area

            intersection_geom = lsoa_geom.intersection(area_polygon_projected)
            
            if intersection_geom.is_valid and not intersection_geom.is_empty:
                intersection_area = intersection_geom.area
                if lsoa_original_area > 1e-9 and (intersection_area / lsoa_original_area) > majority_threshold:
                    # Append the row from the initial geographic CRS dataframe
                    filtered_lsoas_list.append(lsoas_intersecting_area_osm.loc[idx])

        if not filtered_lsoas_list:
            print(f"   After majority-area filtering (threshold: {majority_threshold*100:.0f}%), no LSOAs remain. This might indicate an issue with the area or LSOA data, or the threshold is too strict. Consider adjusting LSOA_MAJORITY_AREA_THRESHOLD.")
            return gpd.GeoDataFrame(columns=[LSOA_COLUMN_NAME_FOR_MERGE, 'LSOA21NM', 'geometry', crime_value_column, 'crime_rate'], crs="EPSG:4326")

        # Create the final GeoDataFrame, ensuring it's back in WGS84 for Folium
        lsoas_within_area_gdf = gpd.GeoDataFrame(filtered_lsoas_list, crs=lsoas_intersecting_area_osm.crs).to_crs(epsg=4326)
        print(f"   Filtered to {len(lsoas_within_area_gdf)} LSOAs after majority-area check.")

        # Assign the actual crime rates (median) to a 'crime_rate' column for consistency with existing functions
        # Ensure it's a numeric type, fill NaN with 0 if any LSOA somehow lost its crime data
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
        # Using network_type='drive' to get a comprehensive network that includes residential roads.
        # 'bike' network type can sometimes be too sparse for large areas/complex routes.
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

def simulate_crime_and_select_waypoints(lsoas_gdf, G, max_total_waypoints_for_tsp, fixed_crime_rate_threshold):
    """
    Identifies high-crime LSOAs (based on median crime rate),
    and selects representative network nodes as patrol waypoints, generating additional points
    based on LSOA size for high-crime LSOAs.
    Args:
        lsoas_gdf (geopandas.GeoDataFrame): GeoDataFrame of LSOAs with 'crime_rate' (which is 'median').
        G (networkx.MultiDiGraph): The cycle network graph.
        max_total_waypoints_for_tsp (int): Maximum total waypoints to select for TSP.
        fixed_crime_rate_threshold (float): LSOAs with crime_rate >= this value will be considered high crime.
    Returns:
        pandas.DataFrame: DataFrame of selected waypoints with their details.
        geopandas.GeoDataFrame: LSOAs GeoDataFrame with crime rates and high-crime flag.
    """
    print("--- Step 4: Selecting Patrol Waypoints Based on Median Crime ---")
    
    # Identify high-crime LSOAs based on the fixed crime rate threshold for display/labeling purposes only.
    lsoas_gdf['is_high_crime'] = lsoas_gdf['crime_rate'] >= fixed_crime_rate_threshold
    
    # To ensure every LSOA gets at least one point, we will iterate over ALL LSOAs in the ward.
    # We still sort by crime rate (descending) so that higher crime LSOAs are processed earlier,
    # which can be relevant if MAX_TOTAL_WAYPOINTS_FOR_TSP is hit and points need to be capped.
    lsoas_to_process = lsoas_gdf.sort_values(by='crime_rate', ascending=False)

    if lsoas_to_process.empty:
        print("   No LSOAs found in the ward to select waypoints from. Exiting waypoint selection.")
        return pd.DataFrame(), lsoas_gdf

    all_generated_waypoints = []
    
    print(f"   Generating points for all {len(lsoas_to_process)} LSOAs in the ward...")
    
    for idx, lsoa_row in lsoas_to_process.iterrows():
        # Get LSOA geometry in projected CRS for accurate area calculation
        lsoa_polygon_projected = gpd.GeoSeries([lsoa_row.geometry], crs=lsoas_gdf.crs).to_crs(epsg=27700).iloc[0]
        lsoa_area_sqm = lsoa_polygon_projected.area

        # Calculate points based on size, respecting POINTS_PER_LSOA_MIN (which is 1)
        num_points_for_this_lsoa = calculate_points_based_on_size(lsoa_area_sqm)
        
        print(f"   Generating {num_points_for_this_lsoa} points in LSOA: {lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]} (Area: {lsoa_area_sqm/1e6:.2f} km², Median Crime: {lsoa_row['crime_rate']:.2f})")
        
        # Ensure the LSOA geometry is valid before generating points (use original CRS for random point generation)
        lsoa_polygon_geographic = lsoa_row.geometry
        if not lsoa_polygon_geographic.is_valid:
            lsoa_polygon_geographic = lsoa_polygon_geographic.buffer(0) # Attempt to fix invalid geometry

        if lsoa_polygon_geographic.is_empty:
            print(f"   Warning: LSOA ({lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]}) has an empty or invalid geometry. Skipping point generation.")
            continue

        random_points = generate_random_points_in_polygon(lsoa_polygon_geographic, num_points_for_this_lsoa)
        
        for i, point in enumerate(random_points):
            nearest_node = ox.nearest_nodes(G, point.x, point.y)
            # Ensure the nearest node is actually in the graph G (the largest connected component)
            if nearest_node in G.nodes:
                all_generated_waypoints.append({
                    'LSOA21CD': lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE], # Store actual LSOA code
                    'LSOA21NM': lsoa_row['LSOA21NM'] + f' (Pt {i+1})' if 'LSOA21NM' in lsoa_row else f"LSOA {lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]} (Pt {i+1})",
                    'simulated_crime_rate': lsoa_row['crime_rate'], # This is actually the median crime
                    'waypoint_lat': G.nodes[nearest_node]['y'],
                    'waypoint_lon': G.nodes[nearest_node]['x'],
                    'nearest_node_id': nearest_node
                })
            else:
                print(f"   Warning: Nearest node {nearest_node} for point in LSOA {lsoa_row[LSOA_COLUMN_NAME_FOR_MERGE]} is not in the main connected component. Skipping this waypoint.")

    patrol_waypoints_df = pd.DataFrame(all_generated_waypoints)

    # If more waypoints were generated than the maximum allowed for TSP, select the top ones.
    # This caps the total number of waypoints used in the TSP, prioritizing higher crime LSOAs.
    if len(patrol_waypoints_df) > max_total_waypoints_for_tsp:
        print(f"   Total waypoints generated ({len(patrol_waypoints_df)}) exceeds MAX_TOTAL_WAYPOINTS_FOR_TSP ({max_total_waypoints_for_tsp}). Selecting top {max_total_waypoints_for_tsp} by crime rate (median).")
        patrol_waypoints_df = patrol_waypoints_df.sort_values(by='simulated_crime_rate', ascending=False).head(max_total_waypoints_for_tsp)
    else:
        print(f"   All {len(patrol_waypoints_df)} generated waypoints will be used.")
    
    print(f"   Selected a total of {len(patrol_waypoints_df)} patrol waypoints for TSP.")
    return patrol_waypoints_df, lsoas_gdf

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
    print("--- Step 5: Calculating Optimal Cycle Patrol Route using TSP ---")
    
    node_ids = waypoints_df['nearest_node_id'].tolist()
    num_nodes = len(node_ids)

    if num_nodes < 2:
        print("   Not enough waypoints to create a route.")
        return [], 0, []
    
    # Create a distance matrix for TSP
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                try:
                    path_nodes = ox.shortest_path(G, node_ids[i], node_ids[j], weight='length')
                    if path_nodes:
                        segment_length = 0
                        for k in range(len(path_nodes) - 1):
                            u, v = path_nodes[k], path_nodes[k+1]
                            if 0 in G[u][v] and 'length' in G[u][v][0]:
                                segment_length += G[u][v][0]['length']
                            else:
                                try:
                                    segment_length += ox.distance.great_circle_vec(
                                        G.nodes[u]['y'], G.nodes[u]['x'], G.nodes[v]['y'], G.nodes[v]['x']
                                    )
                                except KeyError:
                                    segment_length += np.inf
                        distance_matrix[i, j] = segment_length
                    else:
                        distance_matrix[i, j] = np.inf
                except Exception as e:
                    print(f"   Warning: Could not find path between {node_ids[i]} and {node_ids[j]}: {e}")
                    distance_matrix[i, j] = np.inf

    # Handle unreachable waypoints by making their distance very high but not infinite for TSP solver
    if np.isinf(distance_matrix).any():
        print("   Warning: Some waypoints are not reachable from each other. TSP might fail or give suboptimal results.")
        max_finite_dist = distance_matrix[~np.isinf(distance_matrix)].max()
        replacement_val = max_finite_dist * 2 if max_finite_dist > 0 else 1e9
        distance_matrix[np.isinf(distance_matrix)] = replacement_val

    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    
    # The TSP solver returns an order of indices; convert back to original node IDs
    optimal_node_sequence_waypoints = [node_ids[i] for i in permutation]
    # Add the start waypoint at the end to complete the loop
    optimal_node_sequence_waypoints.append(optimal_node_sequence_waypoints[0])

    route_segments_detailed = [] # Store detailed node sequences for each segment
    total_route_distance = 0
    
    # Reconstruct the full detailed path for visualization and total distance
    for i in range(len(optimal_node_sequence_waypoints) - 1):
        try:
            path_nodes = ox.shortest_path(G, optimal_node_sequence_waypoints[i], optimal_node_sequence_waypoints[i+1], weight='length')
            if path_nodes:
                route_segments_detailed.append(path_nodes)
                segment_length = 0
                for k in range(len(path_nodes) - 1):
                    u, v = path_nodes[k], path_nodes[k+1]
                    if 0 in G[u][v] and 'length' in G[u][v][0]:
                        segment_length += G[u][v][0]['length']
                    else:
                        try:
                            segment_length += ox.distance.great_circle_vec(
                                G.nodes[u]['y'], G.nodes[u]['x'], G.nodes[v]['y'], G.nodes[v]['x']
                            )
                        except KeyError:
                            segment_length += 0
                total_route_distance += segment_length
            else:
                print(f"   Error: No path found between {optimal_node_sequence_waypoints[i]} and {optimal_node_sequence_waypoints[i+1]} during route reconstruction. This segment will appear as a straight line.")
        except Exception as e:
            print(f"   Error reconstructing path segment between {optimal_node_sequence_waypoints[i]} and {optimal_node_sequence_waypoints[i+1]}: {e}")
            
    print(f"   Optimal route calculated. Total distance: {total_route_distance:.2f} meters.")
    return optimal_node_sequence_waypoints, total_route_distance, route_segments_detailed

# --- 4. Interactive Visualization of the Patrol Route ---
def visualize_route(area_gdf, lsoas_gdf, waypoints_df, G, optimal_node_sequence, route_segments, output_file_raw_name):
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

    # Save the map with the sanitized filename
    m.save(final_output_filename)
    print(f"   Interactive map saved to {final_output_filename}")

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
                    visualize_route(selected_ward_gdf, lsoas_with_crime_gdf_final, patrol_waypoints_df, G, optimal_node_sequence, route_segments, OUTPUT_MAP_FILE)

    print("\n--- Script Finished ---")