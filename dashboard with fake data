import streamlit as st 
import geopandas as gpd
import pandas as pd
import folium
from folium import GeoJson, GeoJsonTooltip, LayerControl
from streamlit_folium import folium_static
import numpy as np
import branca.colormap as cm
import os

st.set_page_config(layout="wide", page_title="London Crime Risk Dashboard")
st.title(" London Crime Risk Map📍")

#sidebar with feature descriptions
st.sidebar.header("ℹ️ Feature Descriptions")
descriptions = {
    "deprivation_index": "Measures local economic and social disadvantage.",
    "seasonality": "Captures predictable seasonal variations in crime.",
    "previous_burglaries": "Historical frequency of burglaries in the area.",
    "bus_stop_density": "Indicates busy areas with high foot traffic.",
    "distance_police": "Proximity to the nearest police station.",
    "ethnic_diversity_index": "Represents demographic diversity in the area.",
    "average_age": "Average age of residents in the area.",
    "avg_people_per_household": "Average number of people living per home.",
    "social_rent_pct": "Proportion of social housing in the area.",
    "flat_pct": "Percentage of flats versus houses.",
    "security_measures_index": "Assumed private security effectiveness.",
    "spillover_effect": "Influence from crime in neighboring areas.",
    "proximity_to_city_center": "Distance from central London.",
    "road_network_complexity": "Influences accessibility and escape routes."
}
for key in descriptions:
    st.sidebar.markdown(f"**{key.replace('_', ' ').title()}**: {descriptions[key]}")

#select geography level
geo_level = st.selectbox("Select Geography Level", ["LSOA", "MSOA", "Borough"])

#select type of map
view_mode = st.radio("Select View Mode", [
    "Prediction (Current Month)", 
    "Actual (One Past Month)", 
    "Prediction vs Actual (One Past Month)",
    "Compare Two Past Months (Actual)"
])

#definition months 
months = ["January", "February", "March", "April", "May"]

selected_month_1 = None
selected_month_2 = None

#month selection depending on view mode chosen
if view_mode in ["Actual (One Past Month)", "Prediction vs Actual (One Past Month)"]:
    selected_month_1 = st.selectbox("Select Past Month", months)

if view_mode == "Compare Two Past Months (Actual)":
    selected_month_1 = st.selectbox("Select First Past Month", months, index=0)
    selected_month_2 = st.selectbox("Select Second Past Month", months, index=1)


#paths for shapefiles
shapefile_paths = {
    "LSOA": r"C:\\Users\\20231096\\Downloads\\LSOA_Boundaries\\LSOA_2011_London_region",
    "MSOA": r"C:\\Users\\20231096\\Downloads\\Filtered_London_MSOAs.shp",
    "Borough": r"C:\\Users\\20231096\\Downloads\\Borough_Boundaries\\statistical-gis-boundaries-london\\MapInfo"
}

#list of features
features = [
    "deprivation_index", "seasonality", "previous_burglaries", "bus_stop_density",
    "distance_police", "ethnic_diversity_index", "average_age", "avg_people_per_household",
    "social_rent_pct", "flat_pct", "security_measures_index", "spillover_effect",
    "proximity_to_city_center", "road_network_complexity"
]

#function to load geospatial data depending on level
def load_geodata(level):
    path = shapefile_paths[level]
    if level == "Borough":
        tab_files = [f for f in os.listdir(path) if f.endswith(".tab")]
        if not tab_files:
            st.error("No .tab file found in Borough directory")
            st.stop()
        filepath = os.path.join(path, tab_files[0])
        gdf = gpd.read_file(filepath)
    else:
        gdf = gpd.read_file(path)
    return gdf.to_crs(epsg=4326)


gdf = load_geodata(geo_level)
gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.0005, preserve_topology=True)

#generate fake data
np.random.seed(42)
for feature in features:
    gdf[feature] = np.random.rand(len(gdf))

#for fake predictions
norm_data = (gdf[features] - gdf[features].min()) / (gdf[features].max() - gdf[features].min())
weights = np.random.rand(len(features))
weights /= weights.sum()
gdf["crime_risk_pred"] = norm_data.dot(weights)

#no se####
month_noise = {
    "January": 0.02,
    "February": -0.03,
    "March": 0.01,
    "April": 0.00,
    "May": -0.01
}
for m in months:
    noise = month_noise.get(m, 0.0)
    col_name = f"crime_risk_actual_{m.lower()}"
    gdf[col_name] = gdf["crime_risk_pred"] + np.random.normal(loc=noise, scale=0.05, size=len(gdf))
    gdf[col_name] = gdf[col_name].clip(0, 1)

#for map
def render_map(gdf, column, caption):
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="cartodbpositron")
    colormap = cm.linear.YlOrRd_09.scale(gdf[column].min(), gdf[column].max())
    colormap.caption = caption

    def style_function(feature):
        risk = feature['properties'][column]
        return {
            'fillColor': colormap(risk),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.7
        }

    name_field = [col for col in gdf.columns if col.lower().endswith("nm") or "name" in col.lower()]
    fields = []
    aliases = []

    if name_field:
        fields.append(name_field[0])
        aliases.append("Name")

    fields.append(column)
    aliases.append(caption)

    for f in features:
        fields.append(f)
        aliases.append(f.replace("_", " ").title())

    GeoJson(
        gdf,
        name=caption,
        style_function=style_function,
        tooltip=GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            localize=True
        )
    ).add_to(m)

    colormap.add_to(m)
    LayerControl().add_to(m)
    return m


#show maps based on view mode
if view_mode == "Prediction (Current Month)":
    st.subheader("🗘️ Predicted Crime Risk (Current Month)")
    folium_static(render_map(gdf, "crime_risk_pred", "Crime Risk (Predicted)"), height=600)

elif view_mode == "Actual (One Past Month)" and selected_month_1:
    col_name = f"crime_risk_actual_{selected_month_1.lower()}"
    st.subheader(f"📊 Actual Crime Risk - {selected_month_1}")
    folium_static(render_map(gdf, col_name, f"Crime Risk ({selected_month_1})"), height=600)

elif view_mode == "Prediction vs Actual (One Past Month)" and selected_month_1:
    col_name = f"crime_risk_actual_{selected_month_1.lower()}"
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Predicted Crime Risk (Current Month)")
        folium_static(render_map(gdf, "crime_risk_pred", "Crime Risk (Predicted)"), height=600)
    with col2:
        st.subheader(f"📈 Actual Crime Risk ({selected_month_1})")
        folium_static(render_map(gdf, col_name, f"Crime Risk ({selected_month_1})"), height=600)

elif view_mode == "Compare Two Past Months (Actual)" and selected_month_1 and selected_month_2:
    col_name_1 = f"crime_risk_actual_{selected_month_1.lower()}"
    col_name_2 = f"crime_risk_actual_{selected_month_2.lower()}"
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"📈 Actual Crime Risk ({selected_month_1})")
        folium_static(render_map(gdf, col_name_1, f"Crime Risk ({selected_month_1})"), height=600)
    with col2:
        st.subheader(f"📈 Actual Crime Risk ({selected_month_2})")
        folium_static(render_map(gdf, col_name_2, f"Crime Risk ({selected_month_2})"), height=600)
