import patrol
import patrol_ward
import os
import geopandas as gpd

def get_boroughs():
    """
    Returns a list of boroughs in London, using the shapefile if possible.
    """
    shapefile_path = "../data/London_Boroughs.gpkg"
    try:
        gdf = gpd.read_file(shapefile_path)
        # Try to find the column with borough names
        name_col = None
        for col in gdf.columns:
            if col.lower() in ["name", "borough", "borough_name"]:
                name_col = col
                break
        if not name_col:
            # Fallback: use the first string column
            for col in gdf.columns:
                if gdf[col].dtype == object:
                    name_col = col
                    break
        if name_col:
            boroughs = list(gdf[name_col].unique())
            boroughs = [str(b) for b in boroughs if isinstance(b, str)]
            return boroughs
    except Exception as e:
        print(f"Could not load boroughs from shapefile: {e}")
    # Fallback to hardcoded list
    return [
        "Barking and Dagenham",
        "Barnet",
        "Bexley",
        "Brent",
        "Bromley",
        "Camden",
        "Croydon",
        "Ealing",
        "Enfield",
        "Greenwich",
        "Hackney",
        "Hammersmith and Fulham",
        "Haringey",
        "Harrow",
        "Havering",
        "Hillingdon",
        "Hounslow",
        "Islington",
        "Royal Borough of Kensington and Chelsea",
        "Kingston upon Thames",
        "Lambeth",
        "Lewisham",
        "Merton",
        "Newham",
        "Redbridge",
        "Richmond upon Thames",
        "Southwark",
        "Sutton",
        "Tower Hamlets",
        "Waltham Forest",
        "Wandsworth",
        "City of Westminster",
        "City of London"
    ]

def route_exists(location, ward):
    """
    Checks if a patrol route already exists for the given location.
    """
    
    if ward:
        path = os.path.join("ward_routes", f"{location.replace(' ', '_').lower()}.html")
    else:
        path = os.path.join("borough_routes", f"{location.replace(' ', '_').lower()}.html")

    return os.path.exists(path)

def generate_boroughs(override=False):
    """
    Generates the patrol routes for all boroughs.
    """

    boroughs = get_boroughs()
    
    for borough in boroughs:
        if not override and route_exists(borough, False):
            print(f"Patrol route for {borough} already exists. Skipping...")
            continue

        print(f"Generating patrol route for {borough}...")
        patrol.process_location(borough, False)

def get_wards():
    """
    Returns a list of all ward names from the shapefile.
    """
    WARD_SHAPEFILE_PATH = r"..\data\London_Ward.shp"
    WARD_COLUMN_NAME_IN_SHAPEFILE = 'NAME'
    try:
        wards_gdf = gpd.read_file(WARD_SHAPEFILE_PATH)
        return list(wards_gdf[WARD_COLUMN_NAME_IN_SHAPEFILE].unique())
    except Exception as e:
        print(f"Could not load ward shapefile: {e}")
        return []

def generate_wards(override=False):
    """
    Generates the patrol routes for all wards.
    """
    wards = get_wards()
    for ward in wards:
        if not override and route_exists(ward, True):
            print(f"Patrol route for {ward} already exists. Skipping...")
            continue

        print(f"Generating patrol route for {ward}...")
        patrol_ward.process_location(ward)

def check_boroughs():
    """
    Checks if all borough patrol routes exist.
    """
    boroughs = get_boroughs()

    allexist = True
    for borough in boroughs:
        if not route_exists(borough, False):
            print(f"Patrol route for {borough} does not exist.")
            allexist = False
    
    if allexist:
        print("All borough patrol routes exist.")

def check_wards():
    """
    Checks if all ward patrol routes exist.
    """
    wards = get_wards()

    percentage = 0

    allexist = True

    print(f"Checking {len(wards)} wards for patrol routes...")

    for ward in wards:
        if not route_exists(ward, True):
            percentage += 1
            print(f"Patrol route for {ward} does not exist.")
            allexist = False
    
    if allexist:
        print("All ward patrol routes exist.")
    
    else:
        total_wards = len(wards)
        percentage = (total_wards - percentage) / total_wards * 100
        print(f"{percentage:.2f}% of ward patrol routes exist.")

def create_point_image(location):
    """
    Creates an image of all of the points the patrol route uses.
    """

    patrol.process_location(location, False, True)
    

if __name__ == "__main__":
    #generate_boroughs(True)
    #print("All borough patrol routes generated successfully.")

    generate_wards()
    #print("All ward patrol routes generated successfully.")

    # Example usage
    #patrol.process_location("City of London")

    #check_boroughs()
    check_wards()

    #create_point_image("City of London")

    pass


