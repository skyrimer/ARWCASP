# change your directory to prototype before running!

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium

# 🔧 Must be the first Streamlit command
st.set_page_config(page_title="London Crime Predictions", layout="wide")


@st.cache_data
def load_data():
    # Load and preprocess once
    gdf = gpd.read_file("data/London_Boroughs.gpkg").to_crs(epsg=4326)
    crime_df = pd.read_csv("data/crime_predictions.csv")
    # Any heavy merges / transforms here
    merged = gdf[["name", "geometry"]].merge(
        crime_df[["borough", "predicted_count"]],
        left_on="name", right_on="borough", how="left")
    return gdf, crime_df, merged


# Later in your script
gdf, crime_df, merged = load_data()

borough_column = "name"

st.title("London Borough Crime Predictions")

m = folium.Map(location=[51.48, -0.08], zoom_start=10, tiles="cartodbpositron")

folium.Choropleth(
    geo_data=merged,
    name="Crime Choropleth",
    data=merged,
    columns=[borough_column, "predicted_count"],
    key_on=f"feature.properties.{borough_column}",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.3,
    legend_name="Pedicted Crime Count"
).add_to(m)

# Optional: add tooltips on top
folium.GeoJson(
    merged,
    name="Boroughs",
    style_function=lambda _: {"color": "black",
                              "weight": 0.5, "fillOpacity": 0},
    tooltip=folium.GeoJsonTooltip(
        fields=[borough_column, "predicted_count"],
        aliases=["Borough:", "Predicted Count:"]
    )
).add_to(m)

# Display map
clicked_borough = None

# Create two columns: one for the map and one for the graph
col1, col2 = st.columns(2)

with col1:
    # Display the map in the first column
    st.info("🗺️ Click a borough to see predictions and comparisons.")
    map_data = st_folium(m, height=500, use_container_width=True,
                         returned_objects=["last_object_clicked"])

    # extract the clicked feature
    if clicked := map_data["last_object_clicked"]:
        pt = Point(clicked["lng"], clicked["lat"])
        hit = gdf[gdf.geometry.contains(pt)]
        clicked_borough = None if hit.empty else hit[borough_column].iloc[0]
    else:
        clicked_borough = None

with col2:
    # Display the graph in the second column if a borough is clicked
    if clicked_borough:
        # Find the clicked borough row
        selected_row = gdf[gdf[borough_column] == clicked_borough]

        if not selected_row.empty:
            selected_geom = selected_row.geometry.iloc[0]
            adjacent_mask = gdf.geometry.touches(selected_geom)
            adjacent_boroughs = gdf[adjacent_mask][borough_column].astype(
                str).tolist()
        else:
            adjacent_boroughs = []

        st.success(
            f"**{clicked_borough}** has {len(adjacent_boroughs)} neighbors: {', '.join(map(str, adjacent_boroughs))}")

        # Filter prediction data for selected + adjacent boroughs
        relevant_boroughs = [clicked_borough] + adjacent_boroughs
        filtered = crime_df[crime_df["borough"].isin(relevant_boroughs)]

        if filtered.empty:
            st.warning(
                "No prediction data available for selected or adjacent boroughs.")
        else:
            df = (
                filtered
                .set_index("borough")[["predicted_count"]]
            )
            selected_region = f"Selected Region ({clicked_borough})"
            df[selected_region] = df["predicted_count"].where(
                df.index == clicked_borough, 0)
            df["Neighbors"] = df["predicted_count"].where(
                df.index != clicked_borough, 0)
            st.bar_chart(
                df[["Neighbors", selected_region]],
                color=["#FFA15A", "#19D3F3"],
                use_container_width=True,
                x_label="Predicted Incidents",
                y_label="Borough",
                horizontal=True,
                height=500,
            )
    else:
        st.info("👈 Click a borough to view prediction comparisons.")
