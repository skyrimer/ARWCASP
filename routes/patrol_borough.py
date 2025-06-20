import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from shapely.geometry import Polygon, Point, LineString
from python_tsp.heuristics import solve_tsp_local_search
import warnings
import random
import networkx as nx
import networkit as nk
import os

# For colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors


# Suppress warnings for cleaner output, especially from geopandas/shapely
warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
warnings.filterwarnings('ignore', 'The `geoseries.isna()` method is deprecated', UserWarning)
warnings.filterwarnings('ignore', 'The `geoseries.notna()` method is deprecated', UserWarning)
# Suppress MatplotlibDeprecationWarning for cm.get_cmap
warnings.filterwarnings('ignore', 'The get_cmap function was deprecated in Matplotlib', UserWarning)
# Suppress Geometry is in a geographic CRS warning for area calculation
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS. Results from \'area\' are likely incorrect.', UserWarning)


# --- Configuration ---
POINTS_PER_LSOA_MIN = 0 # Minimum points to generate for any high-crime LSOA
POINTS_PER_LSOA_MAX_CAP = 25 # Maximum points to generate for a single LSOA, regardless of size
LSOA_MAJORITY_AREA_THRESHOLD = 0.3 # Percentage of LSOA area that must be within the ward (e.g., 0.3 for 30%)
AREA_TO_POINTS_SCALING_FACTOR = 150000 # Area (in sqm) per point for base calculation

# --- IMPORTANT: Configure your LSOA Shapefile Path here ---
LSOA_SHAPEFILE_PATH = "../data/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4.gpkg"
BOROUGH_SHAPEFILE_PATH = "../data/London_Boroughs.gpkg" 

# --- 1. Geospatial Data Acquisition and Preprocessing ---

def download_area_boundary(area_name):
    """
    Downloads the boundary of the specified London area using OSMnx or shapefile for boroughs.
    Args:
        area_name (str): The name of the area to download.
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the area boundary.
    """
    boroughs = [
        "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "Croydon", "Ealing", "Enfield",
        "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", "Hillingdon",
        "Hounslow", "Islington", "Royal Borough of Kensington and Chelsea", "Kingston upon Thames", "Lambeth", "Lewisham",
        "Merton", "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
        "Waltham Forest", "Wandsworth", "City of Westminster", "City of London"
    ]
    for borough in boroughs:
        if area_name.lower().strip() == borough.lower():
            # Use shapefile for boroughs instead of OSMnx
            print(f"1. Loading borough boundary from shapefile for: {borough} ...")
            try:
                boroughs_gdf = gpd.read_file(BOROUGH_SHAPEFILE_PATH)
                # Try to match by name (case-insensitive)
                match = boroughs_gdf[boroughs_gdf['name'].str.lower() == borough.lower()]
                if match.empty:
                    # Try alternative column names if needed
                    possible_cols = [col for col in boroughs_gdf.columns if 'name' in col.lower()]
                    for col in possible_cols:
                        match = boroughs_gdf[boroughs_gdf[col].str.lower() == borough.lower()]
                        if not match.empty:
                            break
                if not match.empty:
                    area_gdf = match.copy()
                    area_gdf = area_gdf.to_crs(epsg=4326)
                    area_gdf_projected = area_gdf.to_crs(epsg=27700)
                    area_km2 = area_gdf_projected.geometry.area.sum() / 1e6
                    print(f"   Area boundary loaded from shapefile. CRS: {area_gdf.crs}")
                    print(f"   Approximate Area: {area_km2:.2f} km² (in projected CRS)")
                    return area_gdf
                else:
                    print(f"   Could not find borough '{borough}' in shapefile. Falling back to OSMnx.")
            except Exception as e:
                print(f"   Error loading borough shapefile: {e}. Falling back to OSMnx.")
            # If not found in shapefile, fallback to OSMnx below

    # For wards or other names, fallback to previous logic
    if "london" not in area_name.lower():
        area_name = area_name + ", London, UK"
    print(f"1. Downloading boundary for: {area_name}...")
    try:
        area_gdf = ox.geocode_to_gdf(area_name)
        area_gdf_projected = area_gdf.to_crs(epsg=27700)
        area_km2 = area_gdf_projected.geometry.area.sum() / 1e6
        print(f"   Area boundary downloaded. CRS: {area_gdf.crs}")
        print(f"   Approximate Area: {area_km2:.2f} km² (in projected CRS)")
        return area_gdf
    except Exception as e:
        print(f"Error downloading area boundary: {e}")
        return None

def _standardize_lsoa_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardizes LSOA code and name columns to 'LSOA21CD' and 'LSOA21NM'.
    Args:
        gdf (geopandas.GeoDataFrame): The LSOA GeoDataFrame to standardize.
    Returns:
        geopandas.GeoDataFrame: The GeoDataFrame with standardized column names.
    """
    # Standardize LSOA Code column
    lsoa_code_cols = ['LSOA21CD', 'LSOA11CD', 'lsoa_code', 'code']
    found_code_col = None
    for col in lsoa_code_cols:
        if col in gdf.columns:
            found_code_col = col
            break
    if found_code_col and found_code_col != 'LSOA21CD':
        gdf = gdf.rename(columns={found_code_col: 'LSOA21CD'})
        print(f"   Renamed LSOA code column '{found_code_col}' to 'LSOA21CD'.")
    elif not found_code_col:
        print("   Warning: No standard LSOA code column found. 'LSOA21CD' may be missing.")

    # Standardize LSOA Name column
    lsoa_name_cols = ['LSOA21NM', 'LSOA11NM', 'lsoa_name', 'NAME', 'name']
    found_name_col = None
    for col in lsoa_name_cols:
        if col in gdf.columns:
            found_name_col = col
            break
    if found_name_col and found_name_col != 'LSOA21NM':
        gdf = gdf.rename(columns={found_name_col: 'LSOA21NM'})
        print(f"   Renamed LSOA name column '{found_name_col}' to 'LSOA21NM'.")
    elif not found_name_col:
        print("   Warning: No standard LSOA name column found. 'LSOA21NM' may be missing.")
        # As a fallback, if LSOA21NM is absolutely necessary and not found,
        # you might consider creating a dummy column or handling the absence gracefully.
        # For now, we'll let the KeyError propagate if it's still used downstream without being found.
        if 'LSOA21NM' not in gdf.columns:
             gdf['LSOA21NM'] = gdf['LSOA21CD'] # Fallback: use code as name if name missing


    return gdf


def load_and_filter_lsoa_boundaries(lsoa_file_path, area_gdf, majority_threshold=LSOA_MAJORITY_AREA_THRESHOLD):
    """
    Loads LSOA boundaries from a geopackage file or per-borough shapefile and filters them to the specified area,
    retaining only LSOAs where a majority of their area is within the target area.
    Assigns random crime rates to the filtered LSOAs.
    Args:
        lsoa_file_path (str): Path to the LSOA geopackage file.
        area_gdf (geopandas.GeoDataFrame): GeoDataFrame of the target area boundary.
        majority_threshold (float): Percentage (0-1) of LSOA area that must be within the ward.
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame of LSOAs within the area with random crime rates.
    """
    # Try to use per-borough LSOA shapefile if available
    area_name_for_log = area_gdf['name'].iloc[0] if 'name' in area_gdf.columns else "the specified area"
    borough_shapefile_path = None
    if 'name' in area_gdf.columns:
        borough_name = area_gdf['name'].iloc[0]
        borough_shapefile_path = f"../data/lsoashape/{borough_name}.shp"
        if not os.path.exists(borough_shapefile_path):
            # Try with underscores (for e.g. "Kingston upon Thames" -> "Kingston_upon_Thames.shp")
            borough_shapefile_path = f"../data/lsoashape/{borough_name.replace(' ', '_')}.shp"
        if not os.path.exists(borough_shapefile_path):
            borough_shapefile_path = None

    if borough_shapefile_path and os.path.exists(borough_shapefile_path):
        print(f"2. Loading LSOA boundaries from per-borough shapefile: {borough_shapefile_path}")
        try:
            lsoas_gdf = gpd.read_file(borough_shapefile_path)
            lsoas_gdf = lsoas_gdf.to_crs(epsg=4326)
            lsoas_gdf = _standardize_lsoa_columns(lsoas_gdf) # Standardize columns
            print(f"   Loaded {len(lsoas_gdf)} LSOAs from {borough_shapefile_path}.")
            # Add dummy crime_rate column if not present (will be merged later)
            if 'crime_rate' not in lsoas_gdf.columns:
                lsoas_gdf['crime_rate'] = np.nan
            return lsoas_gdf
        except Exception as e:
            print(f"   Error loading per-borough LSOA shapefile: {e}. Falling back to default LSOA file.")

    # Improved logging for area name
    area_name_for_log = area_gdf['name'].iloc[0] if 'name' in area_gdf.columns else "the specified area"
    print(f"2. Loading LSOA boundaries from: {lsoa_file_path} and filtering to {area_name_for_log}...")
    try:
        # Load the full LSOA geopackage
        all_lsoas_gdf = gpd.read_file(lsoa_file_path)
        print(f"   Loaded {len(all_lsoas_gdf)} LSOAs from file. CRS: {all_lsoas_gdf.crs}")

        all_lsoas_gdf = _standardize_lsoa_columns(all_lsoas_gdf) # Standardize columns

        # Ensure both GeoDataFrames have the same CRS for initial spatial join
        if all_lsoas_gdf.crs != area_gdf.crs:
            all_lsoas_gdf = all_lsoas_gdf.to_crs(area_gdf.crs)
            print(f"   Re-projected LSOAs to {area_gdf.crs} for spatial join.")

        # Initial spatial join to get LSOAs that intersect with the target area
        lsoas_within_area_gdf_initial = gpd.sjoin(all_lsoas_gdf, area_gdf, how="inner", predicate="intersects")
        
        # Drop duplicate columns from the join (e.g., 'index_right', 'name_right')
        lsoas_within_area_gdf_initial = lsoas_within_area_gdf_initial.drop(columns=[col for col in lsoas_within_area_gdf_initial.columns if '_right' in col or 'index_' in col], errors='ignore')

        print(f"   Initial filter resulted in {len(lsoas_within_area_gdf_initial)} LSOAs intersecting the area.")

        # Debugging information
        print(f"   Area boundary geometry type: {area_gdf.geometry.iloc[0].geom_type}")
        print(f"   Area boundary bounds: {area_gdf.geometry.iloc[0].bounds}")
        print(f"   LSOA CRS: {all_lsoas_gdf.crs}, Area CRS: {area_gdf.crs}")

        # After spatial join
        print(f"   Number of LSOAs after spatial join: {len(lsoas_within_area_gdf_initial)}")
        if lsoas_within_area_gdf_initial.empty:
            print("   No LSOAs found intersecting the specified area. Try using a more specific area name, e.g., 'London Borough of Barnet, London, UK'.")
            return gpd.GeoDataFrame(columns=['LSOA21CD', 'LSOA21NM', 'geometry', 'crime_rate'], crs="EPSG:4326")

        # Further filter LSOAs where a majority (e.g., >50%) of their area is within the target area
        print(f"   Filtering LSOAs where less than {majority_threshold*100:.0f}% of area is not in the ward...")
        
        # Project both for accurate area and intersection calculations
        area_polygon_projected = area_gdf.to_crs(epsg=27700).geometry.iloc[0]
        lsoas_within_area_gdf_projected = lsoas_within_area_gdf_initial.to_crs(epsg=27700)

        filtered_lsoas_list = []
        for idx, row in lsoas_within_area_gdf_projected.iterrows():
            lsoa_geom = row.geometry
            lsoa_original_area = lsoa_geom.area # Area in projected CRS

            # Calculate intersection with the main area polygon
            intersection_geom = lsoa_geom.intersection(area_polygon_projected)
            
            if intersection_geom.is_valid and not intersection_geom.is_empty:
                intersection_area = intersection_geom.area
                # Check if the LSOA has a non-zero original area to avoid division by zero
                if lsoa_original_area > 1e-9 and (intersection_area / lsoa_original_area) > majority_threshold:
                    filtered_lsoas_list.append(row)

        if not filtered_lsoas_list:
            print(f"   After majority-area filtering (threshold: {majority_threshold*100:.0f}%), no LSOAs remain. This might indicate an issue with the area or LSOA data, or the threshold is too strict. Consider adjusting LSOA_MAJORITY_AREA_THRESHOLD.")
            return gpd.GeoDataFrame(columns=['LSOA21CD', 'LSOA21NM', 'geometry', 'crime_rate'], crs="EPSG:4326")

        lsoas_within_area_gdf = gpd.GeoDataFrame(filtered_lsoas_list, crs=lsoas_within_area_gdf_projected.crs).to_crs(epsg=4326) # Convert back to WGS84
        print(f"   Filtered to {len(lsoas_within_area_gdf)} LSOAs after majority-area check.")

        # Assign random crime rates to the actual LSOAs
        #lsoas_within_area_gdf['crime_rate'] = np.random.uniform(0.1, 1.0, len(lsoas_within_area_gdf))
        #print(f"   Assigned crime rates to {len(lsoas_within_area_gdf)} LSOAs.")
        return lsoas_within_area_gdf
    
    except FileNotFoundError:
        print(f"Error: LSOA file not found at {lsoa_file_path}. Please check the path.")
        return gpd.GeoDataFrame(columns=['LSOA21CD', 'LSOA21NM', 'geometry', 'crime_rate'], crs="EPSG:4326")
    except Exception as e:
        print(f"Error loading or filtering LSOA boundaries: {e}")
        return gpd.GeoDataFrame(columns=['LSOA21CD', 'LSOA21NM', 'geometry', 'crime_rate'], crs="EPSG:4326")

def extract_cycle_network(area_polygon):
    """
    Extracts the drive network within the area boundary using OSMnx.
    Args:
        area_polygon (shapely.geometry.Polygon): The polygon of the area.
    Returns:
        networkx.MultiDiGraph: The graph network graph.
    """
    print("3. Extracting graph network within the area...")
    try:
        G = ox.graph_from_polygon(area_polygon, network_type='drive')
        G = ox.distance.add_edge_lengths(G)

        edges = ox.graph_to_gdfs(G, nodes=False)
        residential_edges = edges[edges['highway'].isin(['residential', 'living_street'])]

        # Get the largest strongly connected component to ensure all nodes are reachable
        # This is crucial for TSP and continuous paths
        if len(G.nodes) > 0:
            components = list(nx.strongly_connected_components(G))
            if components:
                largest_component_nodes = max(components, key=len)
                G = G.subgraph(largest_component_nodes).copy()
                print(f"   Filtered to largest strongly connected component. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
            else:
                print("   Warning: No strongly connected components found in the graph after initial extraction.")
                return None # Return None if no connected component is found
        else:
            print("   Graph has no nodes after initial extraction.")
            return None

        # Penalize main roads to encourage avoiding them
        main_roads_to_avoid = ['primary', 'secondary', 'trunk', 'motorway']
        penalty_factor = 3.0 # Multiply length by this factor for main roads

        for u, v, k, data in G.edges(keys=True, data=True):
            if 'highway' in data:
                if isinstance(data['highway'], list):
                    for highway_type in data['highway']:
                        if highway_type in main_roads_to_avoid:
                            if 'length' in data:
                                data['length'] *= penalty_factor
                            break
                elif isinstance(data['highway'], str) and data['highway'] in main_roads_to_avoid:
                    if 'length' in data:
                        data['length'] *= penalty_factor

        print(f"   Cycle network extracted. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
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

def calculate_points(lsoa_area_sqm: float, area_scaling_factor: float, expected_burglaries: float, min_points: int=POINTS_PER_LSOA_MIN, max_points_per_lsoa_cap: int=POINTS_PER_LSOA_MAX_CAP) -> int:
    """
    Determines the number of points to generate for an LSOA based on its area and expected burglaries.
    Args:
        lsoa_area_sqm (float): Area of the LSOA in square meters (projected CRS).
        area_scaling_factor (float): Area (in sqm) per point (e.g., 150,000 sqm per point).
        expected_burglaries (float): The predicted number of burglaries for this LSOA.
        min_points (int): Minimum number of points to generate.
        max_points_per_lsoa_cap (int): Maximum number of points to generate for a single LSOA.
    Returns:
        int: Number of points to generate for this LSOA.
    """
    # Calculate a base number of points proportional to area
    base_points = lsoa_area_sqm / area_scaling_factor

    # Apply a square root to reduce the impact of very large areas
    # Add 1 to ensure at least 1 point is considered for very small or zero base_points
    scaled_points = 1 + np.sqrt(base_points)

    # Only adjust points if there is predicted crime, and make the impact smaller
    # This factor (0.5 here) can be tuned to control the influence of crime rate
    if expected_burglaries > 0:
        # Increase the number of points based on expected burglaries.
        # A small multiplier (e.g., 0.5) ensures crime rate adds to the density without dominating area.
        points_from_crime = scaled_points * (1 + (expected_burglaries * 0.5))
        points = points_from_crime
    else:
        points = scaled_points # If no crime, just use the area-based scaled points

    # Ensure points are integers and within defined min/max caps
    points = int(np.floor(points))
    points = max(min_points, points)
    points = min(max_points_per_lsoa_cap, points)
    return points

def select_waypoints_from_lsoas(lsoas_gdf, G):
    """
    Selects waypoints from LSOAs based on their size and crime rate.
    Args:
        lsoas_gdf (geopandas.GeoDataFrame): GeoDataFrame of LSOAs with crime rates.
        G (networkx.MultiDiGraph): The cycle network graph.
        max_total_waypoints_for_tsp (int): Maximum total waypoints to select for TSP.
    Returns:
        pandas.DataFrame: DataFrame of selected waypoints with their details.
    """
    print("4. Selecting waypoints from LSOAs...")
    
    all_generated_waypoints = []

    # The LSOA21CD and LSOA21NM columns should now be standardized by load_and_filter_lsoa_boundaries
    # No need for the separate renaming logic here anymore.

    # append the 'crime_rate' column to the LSOAs GeoDataFrame based on the number of burglaries predicted
    # open the expected burglaries parquet file
    expected_burglaries_file = "../model/sample_predictions.parquet"
    if os.path.exists(expected_burglaries_file):
        expected_burglaries_df = pd.read_parquet(expected_burglaries_file)
        # Merge the expected burglaries with the LSOAs GeoDataFrame

        expected_burglaries_df = expected_burglaries_df.reset_index()
        expected_burglaries_df.rename(columns={'index': 'LSOA21CD'}, inplace=True)     

        lsoas_gdf = pd.merge(lsoas_gdf, expected_burglaries_df[['LSOA21CD', 'median']], on='LSOA21CD', how='left')
        
        # Remove any duplicate 'crime_rate' columns before renaming
        if 'crime_rate' in lsoas_gdf.columns:
            lsoas_gdf = lsoas_gdf.drop(columns=['crime_rate'])
        # Rename the column to 'crime_rate' for consistency
        lsoas_gdf.rename(columns={'median': 'crime_rate'}, inplace=True)

        print(f"   Merged expected burglaries data. Total LSOAs: {len(lsoas_gdf)}")
        print("   crime_rate stats after merge:")
        print(lsoas_gdf['crime_rate'].describe())
        print(f"   Number of LSOAs with NaN crime_rate: {lsoas_gdf['crime_rate'].isna().sum()}")
    else:
        print(f"   Warning: Expected burglaries file not found at {expected_burglaries_file}. Using random crime rates instead.")
        # Assign random crime rates if the expected burglaries file is not found
        lsoas_gdf['crime_rate'] = np.random.uniform(0.1, 1.0, len(lsoas_gdf))
        print(f"   Assigned random crime rates to {len(lsoas_gdf)} LSOAs.")
        

    # Iterate through all LSOAs to generate points based on size
    for idx, lsoa_row in lsoas_gdf.iterrows():
        # Ensure crime_rate is a scalar, not a Series
        crime_rate = lsoa_row['crime_rate']
        if isinstance(crime_rate, pd.Series):
            crime_rate = crime_rate.iloc[0]

        # Skip LSOAs where no burglary is predicted (crime_rate is zero or NaN)
        if pd.isna(crime_rate) or crime_rate == 0:
            continue

        # Get LSOA geometry in projected CRS for accurate area calculation
        lsoa_polygon_projected = gpd.GeoSeries([lsoa_row.geometry], crs=lsoas_gdf.crs).to_crs(epsg=27700).iloc[0]
        lsoa_area_sqm = lsoa_polygon_projected.area

        num_points_for_this_lsoa = calculate_points(lsoa_area_sqm, AREA_TO_POINTS_SCALING_FACTOR, crime_rate)
        
        # print(f"   Generating {num_points_for_this_lsoa} points in LSOA: {lsoa_row['LSOA21NM']} (Area: {lsoa_area_sqm/1e6:.2f} km², Crime Rate: {lsoa_row['crime_rate']:.2f})")
        
        # Ensure the LSOA geometry is valid before generating points (use original CRS for random point generation)
        lsoas_polygon_geographic = lsoa_row.geometry
        if not lsoas_polygon_geographic.is_valid:
            lsoas_polygon_geographic = lsoas_polygon_geographic.buffer(0) # Attempt to fix invalid geometry

        if lsoas_polygon_geographic.is_empty:
            print(f"   Warning: LSOA ({lsoa_row['LSOA21NM']}) has an empty or invalid geometry. Skipping point generation.")
            continue

        random_points = generate_random_points_in_polygon(lsoas_polygon_geographic, num_points_for_this_lsoa)
        
        for i, point in enumerate(random_points):
            nearest_node = ox.nearest_nodes(G, point.x, point.y)
            # Ensure the nearest node is actually in the graph G (the largest connected component)
            if nearest_node in G.nodes:
                all_generated_waypoints.append({
                    'LSOA21CD': lsoa_row['LSOA21CD'],
                    'LSOA21NM': lsoa_row['LSOA21NM'] + f' (Pt {i+1})', # Differentiate points within the same LSOA
                    'expected burglaries': lsoa_row['crime_rate'],
                    'waypoint_lat': G.nodes[nearest_node]['y'],
                    'waypoint_lon': G.nodes[nearest_node]['x'],
                    'nearest_node_id': nearest_node
                })
            else:
                print(f"   Warning: Nearest node {nearest_node} for point in LSOA {lsoa_row['LSOA21NM']} is not in the main connected component. Skipping this waypoint.")
    
    patrol_waypoints_df = pd.DataFrame(all_generated_waypoints)

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
                # If path_nodes is None, it means no path was found.
                print(f"   Warning: No path found between {optimal_node_sequence_waypoints[i]} and {optimal_node_sequence_waypoints[i+1]}. This segment will be skipped in visualization and distance calculation.")
        except Exception as e:
            print(f"   Error reconstructing path segment between {optimal_node_sequence_waypoints[i]} and {optimal_node_sequence_waypoints[i+1]}: {e}. This segment will be skipped.")
            
    print(f"   Optimal route calculated. Total distance: {total_route_distance:.2f} meters.")
    return optimal_node_sequence_waypoints, total_route_distance, route_segments_detailed

# --- 4. Interactive Visualization of the Patrol Route ---

def visualize_route(area_gdf, lsoas_gdf, waypoints_df, G, optimal_node_sequence, route_segments, output_file, name):
    """
    Creates an interactive Folium map to visualize the area, LSOAs, waypoints, and patrol route.
    Args:
        area_gdf (geopandas.GeoDataFrame): Area boundary.
        lsoas_gdf (geopandas.GeoDataFrame): LSOAs with crime rates.
        waypoints_df (pandas.DataFrame): Selected waypoints.
        G (networkx.MultiDiGraph): Cycle network graph.
        optimal_node_sequence (list): Ordered list of node IDs for the route (just waypoints).
        route_segments (list): List of lists of node IDs representing each detailed segment of the route.
        output_file (str): Name of the HTML file to save the map.
    """
    print("6. Visualizing the patrol route on an interactive map...")
    
    area_gdf_projected = area_gdf.to_crs(epsg=27700) # British National Grid
    area_centroid_projected = area_gdf_projected.geometry.centroid.iloc[0]
    area_centroid_geographic = gpd.GeoSeries([area_centroid_projected], crs="EPSG:27700").to_crs(epsg=4326).iloc[0]
    m = folium.Map(location=[area_centroid_geographic.y, area_centroid_geographic.x], zoom_start=14, tiles='cartodbpositron')


    # Add Area Boundary
    folium.GeoJson(
        area_gdf.to_json(),
        name=f'{name} Boundary',
        style_function=lambda x: {
            'fillColor': '#cccccc',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.1
        }
    ).add_to(m)

    # Add LSOA Choropleth based on simulated crime rates
    lsoas_gdf_wgs84 = lsoas_gdf.to_crs(epsg=4326)

    # Filter out rows with missing or non-numeric crime_rate before calculating min/max
    lsoas_gdf_wgs84 = lsoas_gdf_wgs84[pd.to_numeric(lsoas_gdf_wgs84['crime_rate'], errors='coerce').notnull()]
    lsoas_gdf_wgs84['crime_rate'] = pd.to_numeric(lsoas_gdf_wgs84['crime_rate'])


    if lsoas_gdf_wgs84.empty:
        print("Warning: No LSOAs with valid crime_rate for visualization. Using default color scale.")
        min_crime = 0
        max_crime = 1
    else:
        min_crime = lsoas_gdf_wgs84['crime_rate'].min()
        max_crime = lsoas_gdf_wgs84['crime_rate'].max()
        # Handle case where min_crime equals max_crime (e.g., all crime rates are the same)
        if min_crime == max_crime:
            if min_crime == 0: # If all are zero
                max_crime = 1 # Set a range to avoid division by zero in normalization
            else: # If all are same non-zero value
                min_crime = 0
                max_crime = min_crime * 2
                if max_crime == 0: max_crime = 1


    # Use matplotlib.colormaps for modern compatibility
    cmap = cm.get_cmap('YlOrRd')  # Yellow to Red colormap
    norm = colors.Normalize(vmin=min_crime, vmax=max_crime)

    def style_function(feature):
        crime_rate = feature['properties'].get('crime_rate', 0)
        # Ensure crime_rate is numeric; default to 0 if not
        if not isinstance(crime_rate, (int, float)):
            crime_rate = 0
        return {
            'fillColor': colors.rgb2hex(cmap(norm(crime_rate))),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.3
        }

    folium.GeoJson(
        lsoas_gdf_wgs84.to_json(),
        name='LSOA number of burglaries expected',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['LSOA21CD', 'LSOA21NM', 'crime_rate'], aliases=['LSOA Code:', 'LSOA Name:', 'Expected burglaries:'])
    ).add_to(m)

    # Add Patrol Waypoints
    waypoint_group = folium.FeatureGroup(name='Patrol Waypoints').add_to(m)
    for idx, row in waypoints_df.iterrows():

        if not optimal_node_sequence:
            icon = folium.Icon(color='blue', icon='circle', prefix='fa')
        else:
            # Differentiate start point with a different icon/color
            is_start_point = (row['nearest_node_id'] == optimal_node_sequence[0])
            if is_start_point:
                icon = folium.Icon(color='green', icon='play', prefix='fa')
            else:
                icon = folium.Icon(color='blue', icon='circle', prefix='fa')

        folium.Marker(
            location=[row['waypoint_lat'], row['waypoint_lon']],
            popup=f"<b>LSOA:</b> {row['LSOA21NM']}<br><b>Expected burglaries:</b> {row['expected burglaries']:.2f}",
            icon=icon
        ).add_to(waypoint_group)

    # Add Optimized Cycle Route
    full_route_latlons = []

    if route_segments:
        # Concatenate all route segments into a single list of node IDs
        # The segments are already detailed paths from OSMnx, so just flatten them
        # and remove redundant nodes at segment junctions.
        
        # Start with the nodes of the first segment
        if route_segments and route_segments[0]: # Ensure route_segments and its first element exist
            for node_id in route_segments[0]:
                if node_id in G.nodes:
                    full_route_latlons.append([G.nodes[node_id]['y'], G.nodes[node_id]['x']])
                else:
                    print(f"Warning: Node ID {node_id} not found in graph G during final route coordinates extraction (first segment).")

        # For subsequent segments, append all nodes except the first one
        # (as it's a duplicate of the previous segment's last node)
        for i in range(1, len(route_segments)):
            if route_segments[i]:
                for node_id in route_segments[i][1:]: # [1:] to skip the duplicate start node
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

    # Save the map
    m.save(output_file)
    print(f"   Interactive map saved to {output_file}")

def process_location(location_name, ward, only_points=False):
    """
    Main function to process the ward or borough, download area boundary, load LSOA boundaries,
    extract graph network, get crime, and select waypoints, calculate the optimal route,
    and output the interactive map.
    Args:
        location_name (str): The name of the ward to process.
    """
    area_gdf = download_area_boundary(location_name)
    if area_gdf is None or area_gdf.empty:
        print("Failed to get area boundary. Exiting.")
        return

    area_polygon = area_gdf.geometry.iloc[0]

    lsoas_gdf = load_and_filter_lsoa_boundaries(LSOA_SHAPEFILE_PATH, area_gdf, LSOA_MAJORITY_AREA_THRESHOLD)
    
    if lsoas_gdf.empty:
        print("No LSOAs found within the area using the provided file. Exiting.")
        return

    G = extract_cycle_network(area_polygon)
    if G is None:
        print("Failed to extract cycle network. Exiting.")
        return

    patrol_waypoints_df, lsoas_with_crime_gdf = select_waypoints_from_lsoas(
        lsoas_gdf, G
    )

    if patrol_waypoints_df.empty:
        print("No patrol waypoints selected. Exiting.")
        return
    
    if only_points:
        print("Only generating waypoints, not calculating route.")
        
        # Display the waypoints on a map
        print("Visualizing waypoints on an interactive map...")
        OUTPUT_MAP_FILE = os.path.join("waypoints", f"{location_name.replace(' ', '_').lower()}_waypoints.html")
        if not os.path.exists("waypoints"):
            os.makedirs("waypoints")
        visualize_route(
            area_gdf,
            lsoas_with_crime_gdf,
            patrol_waypoints_df,
            G,
            [],
            [],
            OUTPUT_MAP_FILE,
            location_name
        )
        return
    
    optimal_node_sequence, total_route_distance, route_segments = calculate_optimal_route(G, patrol_waypoints_df)
    
    if not optimal_node_sequence:
        print("Failed to calculate optimal route. Exiting.")
        return
    
    #print("\n--- Route Summary ---")
    #print(f"Number of Waypoints: {len(patrol_waypoints_df)}")
    #print(f"Total Route Distance: {total_route_distance / 1000:.2f} km")

    # Check if the "ward route" directory exists, if not create it

    if ward:
        WARD_ROUTE_DIR = "ward_routes"
    else:
        WARD_ROUTE_DIR = "borough_routes"

    OUTPUT_MAP_FILE = os.path.join(WARD_ROUTE_DIR, f"{location_name.replace(' ', '_').lower()}.html")

    if not os.path.exists(WARD_ROUTE_DIR):
        os.makedirs(WARD_ROUTE_DIR)

    
    # Visualize the route
    print(f"Visualizing the route and saving to {OUTPUT_MAP_FILE}...")
    visualize_route(
        area_gdf,
        lsoas_with_crime_gdf,
        patrol_waypoints_df,
        G,
        optimal_node_sequence,
        route_segments,
        OUTPUT_MAP_FILE,
        location_name
    )
