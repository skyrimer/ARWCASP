#change your directory to prototype before running!

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px

# 🔧 Must be the first Streamlit command
st.set_page_config(page_title="London Crime Predictions", layout="wide")

# 💄 Optional: Remove excess white space with CSS
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stPlotlyChart { margin-top: -20px; }
    </style>
""", unsafe_allow_html=True)


# Load geospatial data
try:
    gdf = gpd.read_file("data/London_Boroughs.gpkg")
except FileNotFoundError:
    st.error("❌ Could not find 'London_Boroughs.gpkg'.")
    st.stop()

# Load crime predictions
try:
    crime_df = pd.read_csv("data/crime_predictions.csv")
except FileNotFoundError:
    st.error("❌ Could not find 'crime_predictions.csv'.")
    st.stop()

# Ensure consistency: use uppercase borough names in both datasets
gdf["name"] = gdf["name"].str.strip().str.upper()
crime_df["borough"] = crime_df["borough"].str.strip().str.upper()

# Define the column that represents borough names in the shapefile
borough_column = "name"

st.title("📍 London Borough Crime Predictions")

# Create base map
m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="cartodbpositron")

def style_function(feature):
    return {"fillOpacity": 0.1, "weight": 1, "color": "black"}

# Add interactive layer with tooltips
geojson_layer = folium.GeoJson(
    gdf,
    name="Boroughs",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=[borough_column], aliases=["Borough:"])
)
geojson_layer.add_to(m)

# Display map
st.write("🗺️ Click a borough to see predictions and comparisons.")
map_data = st_folium(m, width=700, height=500)

clicked_borough = None
if map_data and isinstance(map_data, dict):
    last_drawing = map_data.get("last_active_drawing")
    if last_drawing:
        props = last_drawing.get("properties", {})
        clicked_borough = props.get(borough_column)

if clicked_borough:
    st.success(f"✅ Selected Borough: {clicked_borough}")

    # Find the clicked borough row
    selected_row = gdf[gdf[borough_column] == clicked_borough]

    if not selected_row.empty:
        selected_geom = selected_row.geometry.iloc[0]
        adjacent_mask = gdf.geometry.touches(selected_geom)
        adjacent_boroughs = gdf[adjacent_mask][borough_column].astype(str).tolist()
    else:
        adjacent_boroughs = []

    st.write(f"📍 Adjacent boroughs: {', '.join(map(str, adjacent_boroughs))}")

    # Filter prediction data for selected + adjacent boroughs
    relevant_boroughs = [clicked_borough] + adjacent_boroughs
    filtered = crime_df[crime_df["borough"].isin(relevant_boroughs)]

    if filtered.empty:
        st.warning("No prediction data available for selected or adjacent boroughs.")
    else:
        fig = px.bar(
            filtered,
            x="borough",
            y="predicted_count",
            color="borough",
            barmode="group",
            title=f"📊 Predicted Crime Comparison: {clicked_borough} vs. Neighbors",
            labels={"borough": "Borough", "predicted_count": "Predicted Incidents"}
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Click a borough to view prediction comparisons.")
