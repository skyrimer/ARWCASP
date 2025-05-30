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
# Suppress Geometry is in a geographic CRS warning for centroid calculation
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS. Results from \'centroid\' are likely incorrect.', UserWarning)


# --- Configuration ---
WARD_NAME = "City of London, UK"
FIXED_CRIME_RATE_THRESHOLD = 0.40 # LSOAs with crime_rate >= this value will be considered high crime
MAX_TOTAL_WAYPOINTS_FOR_TSP = 50 # Maximum total waypoints to feed into the TSP solver
POINTS_PER_LSOA_MIN = 1 # Minimum points to generate for any high-crime LSOA
POINTS_PER_LSOA_MAX_CAP = 5 # Maximum points to generate for a single LSOA, regardless of size
LSOA_AREA_TO_POINT_FACTOR = 50000 # 1 point per this many square meters (e.g., 1 point per 0.05 sq km)
OUTPUT_MAP_FILE = "city_of_london_patrol_route_crime_above_0_40_no_straight_lines_v2.html" # Updated output file name

# --- IMPORTANT: Configure your LSOA Shapefile Path here ---
LSOA_SHAPEFILE_PATH = "../data/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4.gpkg"

# --- 1. Geospatial Data Acquisition and Preprocessing ---

def download_area_boundary(area_name):
    """
    Downloads the boundary of the specified London area using OSMnx.
    Args:
        area_name (str): The name of the area to download.
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the area boundary.
    """
    print(f"1. Downloading boundary for: {area_name}...")
    try:
        area_gdf = ox.geocode_to_gdf(area_name)
        print(f"   Area boundary downloaded. CRS: {area_gdf.crs}")
        
        area_gdf_projected = area_gdf.to_crs(epsg=27700) # British National Grid for accurate area calculation
        print(f"   Approximate Area: {area_gdf_projected.geometry.area.sum() / 1e6:.2f} km² (in projected CRS)")
        
        return area_gdf
    except Exception as e:
        print(f"Error downloading area boundary: {e}")
        return None

def load_and_filter_lsoa_boundaries(lsoa_file_path, area_gdf):
    """
    Loads LSOA boundaries from a geopackage file and filters them to the specified area,
    retaining only LSOAs where a majority of their area is within the target area.
    Assigns random crime rates to the filtered LSOAs.
    Args:
        lsoa_file_path (str): Path to the LSOA geopackage file.
        area_gdf (geopandas.GeoDataFrame): GeoDataFrame of the target area boundary.
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame of LSOAs within the area with random crime rates.
    """
    print(f"2. Loading LSOA boundaries from: {lsoa_file_path} and filtering to {area_gdf.name}...")
    try:
        # Load the full LSOA geopackage
        all_lsoas_gdf = gpd.read_file(lsoa_file_path)
        print(f"   Loaded {len(all_lsoas_gdf)} LSOAs from file. CRS: {all_lsoas_gdf.crs}")

        # Ensure both GeoDataFrames have the same CRS for initial spatial join
        if all_lsoas_gdf.crs != area_gdf.crs:
            all_lsoas_gdf = all_lsoas_gdf.to_crs(area_gdf.crs)
            print(f"   Re-projected LSOAs to {area_gdf.crs} for spatial join.")

        # Initial spatial join to get LSOAs that intersect with the target area
        lsoas_within_area_gdf_initial = gpd.sjoin(all_lsoas_gdf, area_gdf, how="inner", predicate="intersects")
        
        # Drop duplicate columns from the join (e.g., 'index_right', 'name_right')
        lsoas_within_area_gdf_initial = lsoas_within_area_gdf_initial.drop(columns=[col for col in lsoas_within_area_gdf_initial.columns if '_right' in col or 'index_' in col], errors='ignore')

        print(f"   Initial filter resulted in {len(lsoas_within_area_gdf_initial)} LSOAs intersecting the area.")

        if lsoas_within_area_gdf_initial.empty:
            print("   No LSOAs found intersecting the specified area. Please check the area name or LSOA file path.")
            return gpd.GeoDataFrame(columns=['LSOA21CD', 'LSOA21NM', 'geometry', 'crime_rate'], crs="EPSG:4326")

        # Further filter LSOAs where a majority (e.g., >50%) of their area is within the target area
        print("   Filtering LSOAs where majority of area is not in the ward...")
        
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
                if lsoa_original_area > 1e-9 and (intersection_area / lsoa_original_area) > 0.5: # Threshold 50%
                    filtered_lsoas_list.append(row)

        if not filtered_lsoas_list:
            print("   After majority-area filtering, no LSOAs remain. This might indicate an issue with the area or LSOA data.")
            return gpd.GeoDataFrame(columns=['LSOA21CD', 'LSOA21NM', 'geometry', 'crime_rate'], crs="EPSG:4326")

        lsoas_within_area_gdf = gpd.GeoDataFrame(filtered_lsoas_list, crs=lsoas_within_area_gdf_projected.crs).to_crs(epsg=4326) # Convert back to WGS84
        print(f"   Filtered to {len(lsoas_within_area_gdf)} LSOAs after majority-area check.")

        # Assign random crime rates to the actual LSOAs
        lsoas_within_area_gdf['crime_rate'] = np.random.uniform(0.1, 1.0, len(lsoas_within_area_gdf))
        print(f"   Assigned random crime rates to {len(lsoas_within_area_gdf)} LSOAs.")
        return lsoas_within_area_gdf
    except FileNotFoundError:
        print(f"Error: LSOA file not found at {lsoa_file_path}. Please check the path.")
        return gpd.GeoDataFrame(columns=['LSOA21CD', 'LSOA21NM', 'geometry', 'crime_rate'], crs="EPSG:4326")
    except Exception as e:
        print(f"Error loading or filtering LSOA boundaries: {e}")
        return gpd.GeoDataFrame(columns=['LSOA21CD', 'LSOA21NM', 'geometry', 'crime_rate'], crs="EPSG:4326")

def extract_cycle_network(area_polygon):
    """
    Extracts the cycle network within the area boundary using OSMnx.
    Args:
        area_polygon (shapely.geometry.Polygon): The polygon of the area.
    Returns:
        networkx.MultiDiGraph: The cycle network graph.
    """
    print("3. Extracting cycle network within the area...")
    try:
        G = ox.graph_from_polygon(area_polygon, network_type='bike')
        G = ox.distance.add_edge_lengths(G)

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
        main_roads_to_avoid = ['primary', 'secondary', 'tertiary', 'trunk', 'motorway']
        penalty_factor = 2.0 # Multiply length by this factor for main roads

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
    Assigns random crime rates, identifies high-crime LSOAs (based on fixed threshold),
    and selects representative network nodes as patrol waypoints, generating points
    based on LSOA size for all high-crime LSOAs.
    Args:
        lsoas_gdf (geopandas.GeoDataFrame): GeoDataFrame of LSOAs.
        G (networkx.MultiDiGraph): The cycle network graph.
        max_total_waypoints_for_tsp (int): Maximum total waypoints to select for TSP.
        fixed_crime_rate_threshold (float): LSOAs with crime_rate >= this value will be considered high crime.
    Returns:
        pandas.DataFrame: DataFrame of selected waypoints with their details.
        geopandas.GeoDataFrame: LSOAs GeoDataFrame with crime rates and high-crime flag.
    """
    print("4. Simulating crime hotspots and selecting patrol waypoints...")
    
    # Identify high-crime LSOAs based on a fixed crime rate threshold
    lsoas_gdf['is_high_crime'] = lsoas_gdf['crime_rate'] >= fixed_crime_rate_threshold
    
    high_crime_lsoas = lsoas_gdf[lsoas_gdf['is_high_crime']].sort_values(by='crime_rate', ascending=False)
    
    all_generated_waypoints = []
    
    if high_crime_lsoas.empty:
        print("   No high-crime LSOAs found to select waypoints from based on the fixed threshold.")
        return pd.DataFrame(all_generated_waypoints), lsoas_gdf

    # Iterate through all high-crime LSOAs to generate points based on size
    for idx, lsoa_row in high_crime_lsoas.iterrows():
        # Get LSOA geometry in projected CRS for accurate area calculation
        # Corrected: Pass the entire lsoas_gdf.crs to GeoSeries for CRS
        lsoa_polygon_projected = gpd.GeoSeries([lsoa_row.geometry], crs=lsoas_gdf.crs).to_crs(epsg=27700).iloc[0]
        lsoa_area_sqm = lsoa_polygon_projected.area

        num_points_for_this_lsoa = calculate_points_based_on_size(lsoa_area_sqm)
        
        print(f"   Generating {num_points_for_this_lsoa} points in LSOA: {lsoa_row['LSOA21NM']} (Area: {lsoa_area_sqm/1e6:.2f} km², Crime Rate: {lsoa_row['crime_rate']:.2f})")
        
        # Ensure the LSOA geometry is valid before generating points (use original CRS for random point generation)
        lsoa_polygon_geographic = lsoa_row.geometry
        if not lsoa_polygon_geographic.is_valid:
            lsoa_polygon_geographic = lsoa_polygon_geographic.buffer(0) # Attempt to fix invalid geometry

        if lsoa_polygon_geographic.is_empty:
            print(f"   Warning: LSOA ({lsoa_row['LSOA21NM']}) has an empty or invalid geometry. Skipping point generation.")
            continue

        random_points = generate_random_points_in_polygon(lsoa_polygon_geographic, num_points_for_this_lsoa)
        
        for i, point in enumerate(random_points):
            nearest_node = ox.nearest_nodes(G, point.x, point.y)
            # Ensure the nearest node is actually in the graph G (the largest connected component)
            if nearest_node in G.nodes:
                all_generated_waypoints.append({
                    'LSOA21CD': lsoa_row['LSOA21CD'],
                    'LSOA21NM': lsoa_row['LSOA21NM'] + f' (Pt {i+1})', # Differentiate points within the same LSOA
                    'simulated_crime_rate': lsoa_row['crime_rate'],
                    'waypoint_lat': G.nodes[nearest_node]['y'],
                    'waypoint_lon': G.nodes[nearest_node]['x'],
                    'nearest_node_id': nearest_node
                })
            else:
                print(f"   Warning: Nearest node {nearest_node} for point in LSOA {lsoa_row['LSOA21NM']} is not in the main connected component. Skipping this waypoint.")


    patrol_waypoints_df = pd.DataFrame(all_generated_waypoints)

    # If more waypoints were generated than the maximum allowed for TSP, select the top ones
    if len(patrol_waypoints_df) > max_total_waypoints_for_tsp:
        print(f"   Too many waypoints generated ({len(patrol_waypoints_df)}). Selecting top {max_total_waypoints_for_tsp} by crime rate.")
        # Sort by crime rate (descending) and then take the top N
        patrol_waypoints_df = patrol_waypoints_df.sort_values(by='simulated_crime_rate', ascending=False).head(max_total_waypoints_for_tsp)
    
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
    print("5. Calculating optimal cycle patrol route using TSP...")
    
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
                            # Ensure the edge exists and has a 'length' attribute
                            if 0 in G[u][v] and 'length' in G[u][v][0]:
                                segment_length += G[u][v][0]['length']
                            else:
                                # Fallback if length is missing or edge doesn't exist for some reason
                                # This case should ideally not happen if ox.shortest_path returns a path
                                # but it's a safeguard
                                try:
                                    segment_length += ox.distance.great_circle_vec(
                                        G.nodes[u]['y'], G.nodes[u]['x'], G.nodes[v]['y'], G.nodes[v]['x']
                                    )
                                except KeyError: # If node itself is missing coordinates
                                    segment_length += np.inf
                        distance_matrix[i, j] = segment_length
                    else:
                        # If no path, assign a very large distance. This should be rare with connected component filtering.
                        distance_matrix[i, j] = np.inf 
                except Exception as e:
                    print(f"   Warning: Could not find path between {node_ids[i]} and {node_ids[j]}: {e}")
                    distance_matrix[i, j] = np.inf

    # Handle unreachable waypoints by making their distance very high but not infinite for TSP solver
    if np.isinf(distance_matrix).any():
        print("   Warning: Some waypoints are not reachable from each other. TSP might fail or give suboptimal results.")
        # Replace inf with a large number based on existing max distance
        max_finite_dist = distance_matrix[~np.isinf(distance_matrix)].max()
        # Ensure max_finite_dist is not 0 if all distances are 0 or inf
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

def visualize_route(area_gdf, lsoas_gdf, waypoints_df, G, optimal_node_sequence, route_segments, output_file):
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
        name=f'{WARD_NAME} Boundary',
        style_function=lambda x: {
            'fillColor': '#cccccc',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.1
        }
    ).add_to(m)

    # Add LSOA Choropleth based on simulated crime rates
    lsoas_gdf_wgs84 = lsoas_gdf.to_crs(epsg=4326)

    min_crime = lsoas_gdf_wgs84['crime_rate'].min()
    max_crime = lsoas_gdf_wgs84['crime_rate'].max()
    
    # Use cm.get_cmap for compatibility, with warning filter in place
    cmap = cm.get_cmap('YlOrRd')
    norm = colors.Normalize(vmin=min_crime, vmax=max_crime)

    def style_function(feature):
        crime_rate = feature['properties']['crime_rate']
        return {
            'fillColor': colors.rgb2hex(cmap(norm(crime_rate))),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        }

    folium.GeoJson(
        lsoas_gdf_wgs84.to_json(),
        name='LSOA Simulated Crime Rates',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['LSOA21CD', 'LSOA21NM', 'crime_rate'], aliases=['LSOA Code:', 'LSOA Name:', 'Simulated Crime Rate:'])
    ).add_to(m)

    # Add Patrol Waypoints
    waypoint_group = folium.FeatureGroup(name='Patrol Waypoints').add_to(m)
    for idx, row in waypoints_df.iterrows():
        # Differentiate start point with a different icon/color
        is_start_point = (row['nearest_node_id'] == optimal_node_sequence[0])
        if is_start_point:
            icon = folium.Icon(color='green', icon='play', prefix='fa')
        else:
            icon = folium.Icon(color='blue', icon='circle', prefix='fa')

        folium.Marker(
            location=[row['waypoint_lat'], row['waypoint_lon']],
            popup=f"<b>LSOA:</b> {row['LSOA21NM']}<br><b>Crime Rate:</b> {row['simulated_crime_rate']:.2f}",
            icon=icon
        ).add_to(waypoint_group)

    # Add Optimized Cycle Route - now ensuring it follows the network paths
    full_route_latlons = []
    if route_segments:
        # Concatenate all route segments into a single list of node IDs
        # The segments are already detailed paths from OSMnx, so just flatten them
        # and remove redundant nodes at segment junctions.
        
        # Start with the first segment's nodes
        if route_segments[0]:
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

# --- Main Execution ---
if __name__ == "__main__":
    area_gdf = download_area_boundary(WARD_NAME)
    if area_gdf is None or area_gdf.empty:
        print("Failed to get area boundary. Exiting.")
    else:
        area_polygon = area_gdf.geometry.iloc[0]

        lsoas_gdf = load_and_filter_lsoa_boundaries(LSOA_SHAPEFILE_PATH, area_gdf)
        
        if lsoas_gdf.empty:
            print("No LSOAs found within the area using the provided file. Exiting.")
        else:
            G = extract_cycle_network(area_polygon)
            if G is None:
                print("Failed to extract cycle network. Exiting.")
            else:
                patrol_waypoints_df, lsoas_with_crime_gdf = simulate_crime_and_select_waypoints(
                    lsoas_gdf, G, MAX_TOTAL_WAYPOINTS_FOR_TSP, FIXED_CRIME_RATE_THRESHOLD
                )
                
                if patrol_waypoints_df.empty:
                    print("No patrol waypoints selected. Exiting.")
                else:
                    optimal_node_sequence, total_route_distance, route_segments = calculate_optimal_route(G, patrol_waypoints_df)

                    if not optimal_node_sequence:
                        print("Failed to calculate optimal route. Exiting.")
                    else:
                        print("\n--- Route Summary ---")
                        print(f"Number of Waypoints: {len(patrol_waypoints_df)}")
                        print(f"Total Route Distance: {total_route_distance / 1000:.2f} km")
                        estimated_cycle_time_minutes = (total_route_distance / 4.167) / 60
                        print(f"Estimated Cycle Time: {estimated_cycle_time_minutes:.1f} minutes (at 15 km/h avg speed)")
                        print(f"Waypoint Sequence (TSP Order): {optimal_node_sequence}")

                        visualize_route(
                            area_gdf,
                            lsoas_with_crime_gdf,
                            patrol_waypoints_df,
                            G,
                            optimal_node_sequence,
                            route_segments,
                            OUTPUT_MAP_FILE
                        )
                        print(f"\nPatrol route generation complete. Check '{OUTPUT_MAP_FILE}' for the interactive map.")
