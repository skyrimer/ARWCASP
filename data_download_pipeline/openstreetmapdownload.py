import osmnx as ox
import pandas as pd
import time
import os
from shapely.geometry import Point, LineString, Polygon

def download_busy_areas_london():
    """
    Download locations from OpenStreetMap for Greater London.
    Saves the data as a GeoDataFrame to processed_data/london_busy.parquet
    
    Returns:
        GeoDataFrame: Contains locations with their attributes and geometries.
    """

    print("Downloading locations for Greater London...")

    # Define the Greater London area
    place_name = "Greater London, UK"

    # Split tags into smaller groups to avoid timeout
    tag_groups = [
        {'shop': True},
        {'amenity': ['police', 'hospital', 'fire_station']},
        {'amenity': ['school', 'university', 'library']},
        {'public_transport': ['station']},
        {'railway': ['station']},
        {'amenity': ['theatre', 'cinema', 'museum']},
        {'leisure': ['park']},
        {'amenity': ['restaurant', 'cafe', 'bar', 'pub', 'nightclub', 'fast_food']},
        {'amenity': ['parking']}
    ]

    all_gdfs = []

    # Download POIs (Points of Interest) from OSM in groups
    for i, tags in enumerate(tag_groups):
        try:
            print(f"Downloading group {i+1}/{len(tag_groups)}: {tags}")
            gdf = ox.features_from_place(place_name, tags)
            print(f"  - Downloaded {len(gdf)} locations")

            # Create a column for the group name
            group_name = ', '.join([f"{k}: {v}" for k, v in tags.items() if isinstance(v, list)]) or ', '.join([f"{k}: {v}" for k, v in tags.items()])
            gdf['group'] = group_name

            all_gdfs.append(gdf)

            # Add a short delay between requests to avoid overloading the API
            time.sleep(3)  # Increased delay to reduce server load
        except Exception as e:
            print(f"  - Error downloading group {i+1}: {e}")

    # Combine all dataframes if we have any successful downloads
    if all_gdfs:
        try:
            combined_gdf = pd.concat(all_gdfs, ignore_index=True)
            print(f"Total locations downloaded: {len(combined_gdf)}")

            # Clean up the GeoDataFrame
            combined_gdf = combined_gdf.drop_duplicates(subset=['geometry'])
            combined_gdf = combined_gdf.reset_index(drop=True)
            combined_gdf = combined_gdf.dropna(subset=['geometry'])

            # Get centroid coordinates safely for any geometry type
            def get_lat_lon(geom):
                try:
                    if isinstance(geom, Point):
                        return (geom.y, geom.x)
                    centroid = geom.centroid
                    return (centroid.y, centroid.x)
                except Exception:
                    return (None, None)

            # Apply function to extract coordinates
            coords = combined_gdf.geometry.apply(get_lat_lon)
            combined_gdf['latitude'] = coords.apply(lambda x: x[0])
            combined_gdf['longitude'] = coords.apply(lambda x: x[1])

            # Save the data
            os.makedirs(os.path.join('..', 'processed_data'), exist_ok=True)
            output_path = os.path.join('..', 'processed_data', 'london_busy.parquet')

            # Print absolute path to find the file later
            abs_path = os.path.abspath(output_path)
            print(f"Saving to absolute path: {abs_path}")

            combined_gdf.to_parquet(output_path)
            print(f"Saved {len(combined_gdf)} locations to {output_path}")

            return combined_gdf
        except Exception as e:
            print(f"Error combining or saving data: {e}")
            import traceback
            traceback.print_exc()  # Print full error details
            return None
    else:
        print("No data was successfully downloaded")
        return None

if __name__ == "__main__":
    download_busy_areas_london()