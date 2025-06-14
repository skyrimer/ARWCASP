import streamlit as st  # UI
import geopandas as gpd  # Spatial data
import pandas as pd  # Tabular data
import folium  # Interactive mapping
from streamlit_folium import st_folium  # Embed Folium in Streamlit
import streamlit.components.v1 as components
import os  # Filesystem operations
from shapely.geometry import shape
import matplotlib.cm as cm
import matplotlib.colors as colors

def get_colormap_hex_list(cmap_name='YlOrRd', n=6):
    """Return a list of hex colors sampled from a matplotlib colormap."""
    cmap = cm.get_cmap(cmap_name)
    return [colors.rgb2hex(cmap(i/(n-1))) for i in range(n)]

# --- Page setup ---
st.set_page_config(layout="wide", page_title="London Crime & Police Routes Dashboard")
st.title("London Crime Risk & Police Routes📍")

# --- Sidebar: Feature descriptions ---
st.sidebar.header("ℹ️ Feature Descriptions")
descriptions = {
    "deprivation_index": "Local economic and social disadvantage.",
    "seasonality": "Seasonal crime patterns.",
    "previous_burglaries": "Historical burglary frequency.",
    "bus_stop_density": "Foot traffic proxy.",
    "distance_police": "Proximity to nearest station.",
    "ethnic_diversity_index": "Demographic diversity metric.",
    "average_age": "Mean resident age.",
    "avg_people_per_household": "Household occupancy average.",
    "social_rent_pct": "Social housing share.",
    "flat_pct": "Flat vs. house ratio.",
    "security_measures_index": "Private security effectiveness.",
    "spillover_effect": "Neighboring crime influence.",
    "proximity_to_city_center": "Distance to Central London.",
    "road_network_complexity": "Accessibility/escape routes."
}
for key, desc in descriptions.items():
    st.sidebar.markdown(f"**{key.replace('_',' ').title()}**: {desc}")

# --- Sidebar: Upload data files ---
st.sidebar.markdown("📍 To view police route maps, select 'Borough' or 'Ward' and scroll down.")

# Default: automatically detect files
pred_file = "./model/sample_predictions.parquet"
hist_file = "./merged_data.parquet"

if not (os.path.exists(pred_file) and os.path.exists(hist_file)):
    st.sidebar.header("🔄 Upload Data Files")
    st.sidebar.markdown("**Note:** No default files found. Please upload your own data.")

    uploaded_pred_file = st.sidebar.file_uploader("Predictions (.parquet)", type="parquet")
    uploaded_hist_file = st.sidebar.file_uploader("Historical (.parquet)", type="parquet")

    if not uploaded_pred_file or not uploaded_hist_file:
        st.warning("Upload both prediction and historical data to proceed.")
        st.stop()
    else:
        pred_file = uploaded_pred_file
        hist_file = uploaded_hist_file

# Read data
df_pred = pd.read_parquet(pred_file)
df_hist = pd.read_parquet(hist_file)

# --- Prepare historical DataFrame ---
hist_lsoa = next((c for c in df_hist.columns if 'lsoa' in c.lower()), None)
if not hist_lsoa:
    st.error("Historical data lacks LSOA identifier.")
    st.stop()
df_hist = df_hist.rename(columns={hist_lsoa: 'LSOA_code'})
df_hist['date'] = pd.to_datetime(df_hist['date'])  # parse dates

# --- Filter to last 12 months and create year-month labels ---
max_date = df_hist['date'].max()
cutoff = max_date - pd.DateOffset(months=12)
df_recent = df_hist[df_hist['date'] >= cutoff].copy()
df_recent['year_month'] = df_recent['date'].dt.strftime('%Y-%B')
avail_year_months = sorted(
    df_recent['year_month'].unique(),
    key=lambda ym: pd.to_datetime(ym, format='%Y-%B')
)

# --- File paths ---
paths = {
    "Borough": r"./data/London_Boroughs.gpkg",
    "Ward": r"./data/London_Ward.shp"
}
routes_folders = {
    "Borough": r"./routes/borough_routes",
    "Ward": r"./routes/ward_routes"
}

# --- Load geodata ---
@st.cache_data
def load_geodata(level):
    """
    Load geographical data for the specified level.
    """
    folder = paths[level]
    gdf = gpd.read_file(folder)
    return gdf.to_crs(epsg=4326)

# Simplify geometry and load once
borough_gdf_base = load_geodata("Borough")
borough_gdf_base['geometry'] = borough_gdf_base['geometry'].simplify(0.0015, preserve_topology=True)

# Load LSOA to Borough lookup once
lsoa_lookup_path = "./data/LSOA_(2011)_to_LSOA_(2021)_Exact_Fit_Lookup_for_EW_(V3).csv"
lsoa_df = pd.read_csv(lsoa_lookup_path) if os.path.exists(lsoa_lookup_path) else None

# Helper to get a fresh copy of the borough GeoDataFrame
def get_borough_gdf_copy():
    return borough_gdf_base.copy()

# --- Mode switcher ---
st.markdown("### Map Display Mode")
borough_mode = st.radio(
    "Choose map mode:",
    ["Show Predicted", "Compare Predicted vs Actual", "Show Historical"],
    horizontal=True,
    key="borough_mode_radio"
)

# Initialize selected_borough in session state if not present
if "selected_borough" not in st.session_state:
    st.session_state["selected_borough"] = None

# --- Display logic based on mode ---

# --- Show Predicted (main mode with LSOA drill-down) ---
if borough_mode == "Show Predicted":

    # Add a radio button to select borough or ward on the page
    map_type = st.radio(
        "Select Map Type",
        ["Borough", "Ward"],
        horizontal=True,
        key="map_type_radio"
    )
    
    if map_type == "Ward":
        st.subheader("London Wards Map with Predicted Burglaries")
        ward_gdf = load_geodata("Ward")

        # Calculate ward-level predictions if not already present
        if 'Burglaries amount' not in ward_gdf.columns:
            predictions_path = "./model/sample_predictions.parquet"
            lsoa_shp_path = "./data/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4.gpkg"
            if (
                os.path.exists(predictions_path)
                and os.path.exists(lsoa_shp_path)
            ):
                predictions_df = pd.read_parquet(predictions_path)
                predictions_df = predictions_df.reset_index()
                predictions_df.rename(columns={'index': 'LSOA21CD'}, inplace=True)
                if 'median' in predictions_df.columns:
                    predictions_df.rename(columns={'median': 'predictions'}, inplace=True)
                elif 'predictions' not in predictions_df.columns:
                    st.error("Predictions file must have a 'median' or 'predictions' column.")

                # Load LSOA geometries
                lsoa_gdf = gpd.read_file(lsoa_shp_path).to_crs(epsg=4326)
                lsoa_gdf = lsoa_gdf.merge(predictions_df[['LSOA21CD', 'predictions']], on='LSOA21CD', how='left')
                lsoa_gdf = lsoa_gdf[~lsoa_gdf['predictions'].isna()]

                # Area-weighted intersection
                lsoa_gdf['lsoa_area'] = lsoa_gdf.geometry.area
                intersections = gpd.overlay(lsoa_gdf, ward_gdf[['NAME', 'geometry']], how='intersection')
                intersections['intersect_area'] = intersections.geometry.area
                intersections['area_weight'] = intersections['intersect_area'] / intersections['lsoa_area']
                intersections['weighted_burglaries'] = intersections['predictions'] * intersections['area_weight']

                ward_burglaries = intersections.groupby('NAME')['weighted_burglaries'].sum().reset_index()
                ward_burglaries.rename(columns={'weighted_burglaries': 'Burglaries amount'}, inplace=True)
                ward_gdf = ward_gdf.merge(ward_burglaries, on='NAME', how='left')
            else:
                st.error("Cannot compute ward colors: missing predictions or LSOA shapefile.")
                ward_gdf['Burglaries amount'] = 0

        # Color scale using quantiles to reduce outlier effect
        col_data = ward_gdf['Burglaries amount'].dropna()
        if col_data.empty:
            vmin, vmax = 0, 1
        else:
            vmin = col_data.min()
            vmax = col_data.quantile(0.95)
        cmap = cm.get_cmap('YlOrRd')
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        color_stops = get_colormap_hex_list('YlOrRd', 6)

        def style_function_ward(feature):
            crime_rate = feature['properties'].get('Burglaries amount', 0)
            if crime_rate is None:
                crime_rate = 0
            return {
                'fillColor': colors.rgb2hex(cmap(norm(crime_rate))),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.3
            }

        m_ward = folium.Map(location=[51.5074, -0.1278], zoom_start=11, tiles="cartodbpositron")
        folium.GeoJson(
            ward_gdf,
            name="Wards",
            style_function=style_function_ward,
            highlight_function=lambda x: {'weight': 3, 'color': 'blue'},
            tooltip=folium.GeoJsonTooltip(
                fields=['NAME', 'Burglaries amount'],
                aliases=['Ward Name:', 'Predicted Burglaries:']
            )
        ).add_to(m_ward)
        # Add matching color scale
        folium.LinearColormap(color_stops, vmin=vmin, vmax=vmax, caption='Predicted Burglaries').add_to(m_ward)

        # Show the map and capture click events
        ward_map_data = st_folium(
            m_ward,
            height=600,
            width=900,
            returned_objects=["last_active_drawing"],
            key="main_ward_map"
        )

        # --- Ward patrol route display on click ---
        selected_ward = None
        if ward_map_data and ward_map_data.get("last_active_drawing"):
            clicked_geom = shape(ward_map_data["last_active_drawing"]["geometry"])
            for idx, row in ward_gdf.iterrows():
                # Use a small tolerance for exact equality comparison of geometries
                if row['geometry'].equals_exact(clicked_geom, tolerance=1e-6):
                    selected_ward = row['NAME']
                    break
            if not selected_ward:
                pt = clicked_geom.centroid
                for idx, row in ward_gdf.iterrows():
                    if row['geometry'].contains(pt):
                        selected_ward = row['NAME']
                        break

        if selected_ward:
            ward_route_name = selected_ward.lower().replace(" ", "_")
            route_html_path = f"./routes/ward_routes/{ward_route_name}.html"
            st.markdown(f"**Selected Ward:** {selected_ward}")
            if os.path.exists(route_html_path):
                with open(route_html_path, "r", encoding="utf-8") as f:
                    route_html = f.read()
                st.markdown("**Patrol Route Map for this Ward:**")
                components.html(route_html, height=600, scrolling=True)
            else:
                st.warning(f"No patrol route HTML found for {selected_ward} in ./routes/ward_routes/")
        

    else:
        st.subheader("London Boroughs Map with Predicted Burglaries")
        borough_gdf = get_borough_gdf_copy()

        # Calculate borough-level predictions if not already present
        if 'Burglaries amount' not in borough_gdf.columns:
            predictions_path = "./model/sample_predictions.parquet"
            if os.path.exists(predictions_path) and lsoa_df is not None:
                predictions_df = pd.read_parquet(predictions_path)
                predictions_df = predictions_df.reset_index()
                predictions_df.rename(columns={'index': 'LSOA21CD'}, inplace=True)
                if 'median' in predictions_df.columns:
                    predictions_df.rename(columns={'median': 'predictions'}, inplace=True)
                elif 'predictions' not in predictions_df.columns:
                    st.error("Predictions file must have a 'median' or 'predictions' column.")
                merged = pd.merge(predictions_df, lsoa_df, on='LSOA21CD', how='left')
                merged = merged.dropna(subset=['LAD22NM'])
                borough_burglaries = merged.groupby('LAD22NM')['predictions'].sum().reset_index()
                borough_burglaries.rename(columns={'LAD22NM': 'name', 'predictions': 'Burglaries amount'}, inplace=True)
                borough_gdf = borough_gdf.merge(borough_burglaries, on='name', how='left')
            else:
                st.error("Cannot compute borough colors: missing predictions or LSOA lookup.")
                borough_gdf['Burglaries amount'] = 0  # fallback

        # Color scale using quantiles to reduce outlier effect
        col_data = borough_gdf['Burglaries amount'].dropna()
        if col_data.empty:
            vmin, vmax = 0, 1
        else:
            vmin = col_data.min()
            vmax = col_data.quantile(0.95)
        cmap = cm.get_cmap('YlOrRd')
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        color_stops = get_colormap_hex_list('YlOrRd', 6)

        def style_function(feature):
            crime_rate = feature['properties'].get('Burglaries amount', 0)
            if crime_rate is None:
                crime_rate = 0
            return {
                'fillColor': colors.rgb2hex(cmap(norm(crime_rate))),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.3
            }

        m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="cartodbpositron")
        folium.GeoJson(
            borough_gdf,
            name="Boroughs",
            style_function=style_function,
            highlight_function=lambda x: {'weight': 3, 'color': 'blue'},
            tooltip=folium.GeoJsonTooltip(fields=['name', 'Burglaries amount'])
        ).add_to(m)
        folium.LinearColormap(color_stops, vmin=vmin, vmax=vmax, caption='Predicted Burglaries').add_to(m)
        map_data = st_folium(m, height=600, width=900, returned_objects=["last_active_drawing"], key="main_borough_map")

        # Update selected_borough based on map click
        if map_data and map_data.get("last_active_drawing"):
            clicked_geom = shape(map_data["last_active_drawing"]["geometry"])
            found_borough = None
            for idx, row in borough_gdf.iterrows():
                # Use a small tolerance for exact equality comparison of geometries
                if row['geometry'].equals_exact(clicked_geom, tolerance=1e-6):
                    found_borough = row['name']
                    break
            if not found_borough: # Fallback if exact_equals fails, use centroid containment
                pt = clicked_geom.centroid
                for idx, row in borough_gdf.iterrows():
                    if row['geometry'].contains(pt):
                        found_borough = row['name']
                        break
            
            if found_borough and found_borough != st.session_state["selected_borough"]:
                st.session_state["selected_borough"] = found_borough
                # Rerun the app to update the LSOA map section
                st.rerun() # This will ensure the LSOA section below updates immediately

        # --- LSOA Details and Route Viewer (ONLY in "Show Predicted" mode) ---
        selected_borough = st.session_state["selected_borough"]
        if selected_borough:
            st.success(f"Selected Borough: {selected_borough}")
            # --- LSOA Map for Selected Borough ---
            lsoa_shp_path = f"./data/lsoashape/{selected_borough}.shp"
            if os.path.exists(lsoa_shp_path):
                lsoa_gdf = gpd.read_file(lsoa_shp_path).to_crs(epsg=4326)
                
                # Merge predictions into LSOA GeoDataFrame
                pred_col = 'median'
                df_pred_lsoa = df_pred.reset_index().rename(columns={'index': 'LSOA_code', pred_col: 'predictions'})
                key = next((c for c in lsoa_gdf.columns if 'lsoa' in c.lower()), None)
                lsoa_gdf = lsoa_gdf.merge(df_pred_lsoa, how='left', left_on=key, right_on='LSOA_code')

                # Print the predicted burglaries for E01032739 (example specific LSOA)
                if 'predictions' not in lsoa_gdf.columns:
                    st.error("Predictions column not found in LSOA data.")
                    lsoa_gdf['predictions'] = 0
                if 'E01032739' in lsoa_gdf[key].values:
                    e01032739_pred = lsoa_gdf[lsoa_gdf[key] == 'E01032739']['predictions'].values[0]
                    st.write(f"Predicted burglaries for E01032739: {e01032739_pred}")

                # Color scale for predictions
                col_data_lsoa = lsoa_gdf['predictions'].dropna()
                if col_data_lsoa.empty:
                    vmin_lsoa, vmax_lsoa = 0, 1
                else:
                    vmin_lsoa = col_data_lsoa.min()
                    vmax_lsoa = col_data_lsoa.quantile(0.95)
                cmap = cm.get_cmap('YlOrRd')
                norm = colors.Normalize(vmin=vmin_lsoa, vmax=vmax_lsoa)
                color_stops = get_colormap_hex_list('YlOrRd', 6)

                # Checkbox for patrol route, unique key for this specific LSOA map
                checkbox_key = f"show_route_checkbox_{selected_borough}_predicted"
                show_route = st.checkbox(
                    "Show patrol route for this borough",
                    value=st.session_state.get(checkbox_key, False),
                    key=checkbox_key
                )

                if show_route:
                    borough_route_name = selected_borough.lower().replace(" ", "_")
                    route_html_path = f"./routes/borough_routes/{borough_route_name}.html"
                    if os.path.exists(route_html_path):
                        with open(route_html_path, "r", encoding="utf-8") as f:
                            route_html = f.read()
                        st.markdown("**Patrol Route Map for this Borough:**")
                        components.html(route_html, height=600, scrolling=True)
                    else:
                        st.warning(f"No patrol route HTML found for {selected_borough} in ./routes/borough_routes/")
                else:
                    # Map centered on borough
                    centroid = lsoa_gdf.geometry.union_all().centroid
                    m_borough = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="cartodbpositron")

                    def style_function(feature):
                        crime_rate = feature['properties'].get('predictions', 0)
                        if crime_rate is None:
                            crime_rate = 0
                        return {
                            'fillColor': colors.rgb2hex(cmap(norm(crime_rate))),
                            'color': 'black',
                            'weight': 0.5,
                            'fillOpacity': 0.3
                        }

                    folium.GeoJson(
                        lsoa_gdf.to_json(),
                        name='LSOA number of burglaries expected',
                        style_function=style_function,
                        tooltip=folium.GeoJsonTooltip(
                            fields=[key, 'predictions'],
                            aliases=['LSOA Code:', 'Expected burglaries:']
                        )
                    ).add_to(m_borough)
                    folium.LayerControl().add_to(m_borough)
                    folium.LinearColormap(color_stops, vmin=vmin_lsoa, vmax=vmax_lsoa, caption='Predicted Burglaries').add_to(m_borough)
                    # Capture click events on LSOA map
                    lsoa_map_data = st_folium(
                        m_borough,
                        height=600,
                        width=900,
                        returned_objects=["last_active_drawing"],
                        key=f"lsoa_map_{selected_borough}_predicted"
                    )

                    # --- Show LSOA factors from merged_data if clicked ---
                    selected_lsoa_code = None
                    if lsoa_map_data and lsoa_map_data.get("last_active_drawing"):
                        clicked_geom = shape(lsoa_map_data["last_active_drawing"]["geometry"])
                        for idx, row in lsoa_gdf.iterrows():
                            if row['geometry'].equals_exact(clicked_geom, tolerance=1e-6):
                                selected_lsoa_code = row[key]
                                break
                        if not selected_lsoa_code:
                            pt = clicked_geom.centroid
                            for idx, row in lsoa_gdf.iterrows():
                                if row['geometry'].contains(pt):
                                    selected_lsoa_code = row[key]
                                    break

                    if selected_lsoa_code:
                        st.markdown(f"### Factors for LSOA: {selected_lsoa_code}")
                        lsoa_factors = df_hist[df_hist['LSOA_code'] == selected_lsoa_code]
                        if not lsoa_factors.empty:
                            lsoa_factors = lsoa_factors.sort_values("date", ascending=False).iloc[0]
                            exclude_cols = ['geometry', 'date']
                            display_factors = lsoa_factors.drop(labels=[col for col in exclude_cols if col in lsoa_factors.index])

                            import matplotlib.pyplot as plt

                            # --- Pie Chart Data Preparation ---
                            # Dwelling type
                            dwelling_cols = [
                                "Dwelling type|Flat, maisonette or apartment (%)"
                            ]
                            dwelling_vals = [lsoa_factors.get(col, 0) for col in dwelling_cols]
                            dwelling_other = 100 - sum(dwelling_vals)
                            dwelling_labels = ["Flat, maisonette or apartment", "Other"]
                            dwelling_sizes = [dwelling_vals[0], dwelling_other]

                            # Ethnic group
                            ethnic_cols = [
                                "Ethnic Group|Asian/Asian British (%)",
                                "Ethnic Group|Black/African/Caribbean/Black British (%)",
                                "Ethnic Group|Mixed/multiple ethnic groups (%)",
                                "Ethnic Group|Other ethnic group (%)",
                                "Ethnic Group|White (%)"
                            ]
                            ethnic_labels = [
                                "Asian/Asian British",
                                "Black/African/Caribbean/Black British",
                                "Mixed/multiple ethnic groups",
                                "Other ethnic group",
                                "White"
                            ]
                            ethnic_vals = [lsoa_factors.get(col, 0) for col in ethnic_cols]

                            # Household composition
                            hh_cols = [
                                "Household Composition|% Couple household with dependent children",
                                "Household Composition|% Couple household without dependent children",
                                "Household Composition|% Lone parent household",
                                "Household Composition|% One person household",
                                "Household Composition|% Other multi person household"
                            ]
                            hh_labels = [
                                "Couple w/ children",
                                "Couple w/o children",
                                "Lone parent",
                                "One person",
                                "Other multi person"
                            ]
                            hh_vals = [lsoa_factors.get(col, 0) for col in hh_cols]

                            # Mid-year Population Estimates (Aged 0-65+)
                            pop_cols = [
                                "Mid-year Population Estimates|Aged 0-15",
                                "Mid-year Population Estimates|Aged 16-29",
                                "Mid-year Population Estimates|Aged 30-44",
                                "Mid-year Population Estimates|Aged 45-64",
                                "Mid-year Population Estimates|Aged 65+"
                            ]
                            pop_labels = [
                                "0-15", "16-29", "30-44", "45-64", "65+"
                            ]
                            pop_vals = [lsoa_factors.get(col, 0) for col in pop_cols]

                            # Tenure
                            tenure_cols = [
                                "Tenure|Owned outright (%)",
                                "Tenure|Owned with a mortgage or loan (%)",
                                "Tenure|Private rented (%)",
                                "Tenure|Social rented (%)"
                            ]
                            tenure_labels = [
                                "Owned outright",
                                "Owned w/ mortgage/loan",
                                "Private rented",
                                "Social rented"
                            ]
                            tenure_vals = [lsoa_factors.get(col, 0) for col in tenure_cols]

                            # Car or van availability (except cars per household)
                            car_cols = [
                                "Car or van availability|No cars or vans in household (%)",
                                "Car or van availability|1 car or van in household (%)",
                                "Car or van availability|2 cars or vans in household (%)",
                                "Car or van availability|3 cars or vans in household (%)",
                                "Car or van availability|4 or more cars or vans in household (%)"
                            ]
                            car_labels = [
                                "No cars/vans",
                                "1 car/van",
                                "2 cars/vans",
                                "3 cars/vans",
                                "4+ cars/vans"
                            ]
                            car_vals = [lsoa_factors.get(col, 0) for col in car_cols]

                            # Public Transport Accessibility Levels|%
                            ptal_cols = [
                                "Public Transport Accessibility Levels|% 0-1 (poor access)|Level3_65",
                                "Public Transport Accessibility Levels|% 2-3 (average access)|Level3_66",
                                "Public Transport Accessibility Levels|% 4-6 (good access)|Level3_67"
                            ]
                            ptal_labels = [
                                "PTAL 0-1 (poor)",
                                "PTAL 2-3 (average)",
                                "PTAL 4-6 (good)"
                            ]
                            ptal_vals = [lsoa_factors.get(col, 0) for col in ptal_cols]

                            # --- Pie Charts in 3 columns, above the factors table ---
                            col_pie1, col_pie2, col_pie3 = st.columns(3)
                            with col_pie1:
                                fig1, ax1 = plt.subplots()
                                ax1.pie(dwelling_sizes, labels=dwelling_labels, autopct='%1.1f%%', startangle=90)
                                ax1.set_title("Dwelling Type")
                                st.pyplot(fig1)

                            with col_pie2:
                                fig2, ax2 = plt.subplots()
                                ax2.pie(ethnic_vals, labels=ethnic_labels, autopct='%1.1f%%', startangle=90)
                                fig2.suptitle("Ethnic Group")
                                st.pyplot(fig2)

                            with col_pie3:
                                fig3, ax3 = plt.subplots()
                                ax3.pie(hh_vals, labels=hh_labels, autopct='%1.1f%%', startangle=90)
                                ax3.set_title("Household Composition")
                                st.pyplot(fig3)

                            col_pie4, col_pie5, col_pie6 = st.columns(3)
                            with col_pie4:
                                fig4, ax4 = plt.subplots()
                                ax4.pie(pop_vals, labels=pop_labels, autopct='%1.1f%%', startangle=90)
                                ax4.set_title("Population Age Groups")
                                st.pyplot(fig4)

                            with col_pie5:
                                fig5, ax5 = plt.subplots()
                                ax5.pie(tenure_vals, labels=tenure_labels, autopct='%1.1f%%', startangle=90)
                                fig5.suptitle("Tenure")
                                st.pyplot(fig5)

                            with col_pie6:
                                fig6, ax6 = plt.subplots()
                                ax6.pie(car_vals, labels=car_labels, autopct='%1.1f%%', startangle=90)
                                ax6.set_title("Car or Van Availability")
                                st.pyplot(fig6)

                            # Last row for PTAL
                            col_pie7, _, _ = st.columns(3)
                            with col_pie7:
                                fig7, ax7 = plt.subplots()
                                ax7.pie(ptal_vals, labels=ptal_labels, autopct='%1.1f%%', startangle=90)
                                ax7.set_title("Public Transport Accessibility Levels (%)")
                                st.pyplot(fig7)

                            # Show the factors table below the pie charts
                            st.table(display_factors)
                        else:
                            st.info("No factor data found for this LSOA in merged_data.")
# --- Compare Predicted vs Actual ---
elif borough_mode == "Compare Predicted vs Actual":
    st.subheader("Compare Predicted vs Actual Burglaries by Borough")
    st.info("This section shows a side-by-side comparison of predicted (next month) and actual burglaries for the latest available month.")

    borough_gdf_pred_display = get_borough_gdf_copy()
    borough_gdf_actual_display = get_borough_gdf_copy()

    # Calculate predicted borough-level data
    predictions_path = "./model/sample_predictions.parquet"
    if os.path.exists(predictions_path) and lsoa_df is not None:
        predictions_df = pd.read_parquet(predictions_path)
        predictions_df = predictions_df.reset_index()
        predictions_df.rename(columns={'index': 'LSOA21CD'}, inplace=True)
        if 'median' in predictions_df.columns:
            predictions_df.rename(columns={'median': 'predictions'}, inplace=True)
        elif 'predictions' not in predictions_df.columns:
            st.error("Predictions file must have a 'median' or 'predictions' column.")
        merged_pred = pd.merge(predictions_df, lsoa_df, on='LSOA21CD', how='left')
        merged_pred = merged_pred.dropna(subset=['LAD22NM'])
        borough_burglaries_pred = merged_pred.groupby('LAD22NM')['predictions'].sum().reset_index()
        borough_burglaries_pred.rename(columns={'LAD22NM': 'name', 'predictions': 'Predicted Burglaries'}, inplace=True)
        borough_gdf_pred_display = borough_gdf_pred_display.merge(borough_burglaries_pred, on='name', how='left')
    else:
        st.error("Cannot compute predicted borough colors: missing predictions or LSOA lookup.")
        borough_gdf_pred_display['Predicted Burglaries'] = 0  # fallback

    # Calculate actual borough-level data for the latest month
    latest_month = df_recent['year_month'].max()
    df_actual = df_recent[df_recent['year_month'] == latest_month]
    if lsoa_df is not None and 'LAD22NM' in lsoa_df.columns:
        lsoa_actual = df_actual.merge(lsoa_df, left_on='LSOA_code', right_on='LSOA21CD', how='left')
        borough_actual = lsoa_actual.groupby('LAD22NM')['Burglaries amount'].sum().reset_index()
        borough_actual.rename(columns={'LAD22NM': 'name', 'Burglaries amount': 'Actual Burglaries'}, inplace=True)
        borough_gdf_actual_display = borough_gdf_actual_display.merge(borough_actual, on='name', how='left')
    else:
        borough_gdf_actual_display['Actual Burglaries'] = 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Predicted Burglaries**")
        col_data_pred = borough_gdf_pred_display['Predicted Burglaries'].dropna()
        vmin_pred = col_data_pred.min() if not col_data_pred.empty else 0
        vmax_pred = col_data_pred.quantile(0.95) if not col_data_pred.empty else 1
        cmap_pred = cm.get_cmap('YlOrRd')
        norm_pred = colors.Normalize(vmin=vmin_pred, vmax=vmax_pred)
        color_stops_pred = get_colormap_hex_list('YlOrRd', 6)

        m_pred = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="cartodbpositron")
        def style_function_pred(feature):
            val = feature['properties'].get('Predicted Burglaries', 0)
            if val is None:
                val = 0
            return {
                'fillColor': colors.rgb2hex(cmap_pred(norm_pred(val))),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.3
            }
        folium.GeoJson(
            borough_gdf_pred_display,
            name="Boroughs",
            style_function=style_function_pred,
            tooltip=folium.GeoJsonTooltip(fields=['name', 'Predicted Burglaries'])
        ).add_to(m_pred)
        folium.LinearColormap(color_stops_pred, vmin=vmin_pred, vmax=vmax_pred, caption='Predicted Burglaries').add_to(m_pred)
        pred_map_data = st_folium(
            m_pred,
            height=500,
            width=450,
            returned_objects=["last_active_drawing"],
            key="compare_pred_borough_map"
        )

    with col2:
        st.markdown(f"**Actual Burglaries ({latest_month})**")
        col_data_actual = borough_gdf_actual_display['Actual Burglaries'].dropna()
        vmin_actual = col_data_actual.min() if not col_data_actual.empty else 0
        vmax_actual = col_data_actual.quantile(0.95) if not col_data_actual.empty else 1
        cmap_actual = cm.get_cmap('YlOrRd')
        norm_actual = colors.Normalize(vmin=vmin_actual, vmax=vmax_actual)
        color_stops_actual = get_colormap_hex_list('YlOrRd', 6)

        m_actual = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="cartodbpositron")
        def style_function_actual(feature):
            val = feature['properties'].get('Actual Burglaries', 0)
            if val is None:
                val = 0
            return {
                'fillColor': colors.rgb2hex(cmap_actual(norm_actual(val))),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.3
            }
        folium.GeoJson(
            borough_gdf_actual_display,
            name="Boroughs",
            style_function=style_function_actual,
            tooltip=folium.GeoJsonTooltip(fields=['name', 'Actual Burglaries'])
        ).add_to(m_actual)
        folium.LinearColormap(color_stops_actual, vmin=vmin_actual, vmax=vmax_actual, caption='Actual Burglaries').add_to(m_actual)
        actual_map_data = st_folium(
            m_actual,
            height=500,
            width=450,
            returned_objects=["last_active_drawing"],
            key="compare_actual_borough_map"
        )

    # --- Borough selection and LSOA map display ---
    selected_compare_borough = None
    # Prefer click on predicted map, fallback to actual map
    if pred_map_data and pred_map_data.get("last_active_drawing"):
        map_data = pred_map_data
    else:
        map_data = actual_map_data
    if map_data and map_data.get("last_active_drawing"):
        clicked_geom = shape(map_data["last_active_drawing"]["geometry"])
        for idx, row in borough_gdf_pred_display.iterrows():
            if row['geometry'].equals_exact(clicked_geom, tolerance=1e-6):
                selected_compare_borough = row['name']
                break
        if not selected_compare_borough:
            pt = clicked_geom.centroid
            for idx, row in borough_gdf_pred_display.iterrows():
                if row['geometry'].contains(pt):
                    selected_compare_borough = row['name']
                    break

    if selected_compare_borough:
        st.success(f"Selected Borough: {selected_compare_borough}")
        lsoa_shp_path = f"./data/lsoashape/{selected_compare_borough}.shp"
        if os.path.exists(lsoa_shp_path):
            lsoa_gdf = gpd.read_file(lsoa_shp_path).to_crs(epsg=4326)
            # Merge predictions into LSOA GeoDataFrame
            pred_col = 'median'
            df_pred_lsoa = df_pred.reset_index().rename(columns={'index': 'LSOA_code', pred_col: 'predictions'})
            key = next((c for c in lsoa_gdf.columns if 'lsoa' in c.lower()), None)
            lsoa_gdf = lsoa_gdf.merge(df_pred_lsoa, how='left', left_on=key, right_on='LSOA_code')
            # Prepare actuals for the latest month
            lsoa_actuals = df_actual[df_actual['LSOA_code'].isin(lsoa_gdf[key])]
            lsoa_gdf = lsoa_gdf.merge(
                lsoa_actuals[['LSOA_code', 'Burglaries amount']],
                how='left',
                left_on='LSOA_code',
                right_on='LSOA_code',
                suffixes=('', '_actual')
            )
            # Color scales
            col_data_pred = lsoa_gdf['predictions'].dropna()
            col_data_actual = lsoa_gdf['Burglaries amount'].dropna()
            vmin_pred = col_data_pred.min() if not col_data_pred.empty else 0
            vmax_pred = col_data_pred.quantile(0.95) if not col_data_pred.empty else 1
            vmin_actual = col_data_actual.min() if not col_data_actual.empty else 0
            vmax_actual = col_data_actual.quantile(0.95) if not col_data_actual.empty else 1
            cmap_pred = cm.get_cmap('YlOrRd')
            norm_pred = colors.Normalize(vmin=vmin_pred, vmax=vmax_pred)
            cmap_actual = cm.get_cmap('YlOrRd')
            norm_actual = colors.Normalize(vmin=vmin_actual, vmax=vmax_actual)
            color_stops_pred = get_colormap_hex_list('YlOrRd', 6)
            color_stops_actual = get_colormap_hex_list('YlOrRd', 6)
            centroid = lsoa_gdf.geometry.union_all().centroid

            # --- Show both maps side by side ---
            col_lsoa_pred, col_lsoa_actual = st.columns(2)
            with col_lsoa_pred:
                m_borough_pred = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="cartodbpositron")
                def style_function_pred(feature):
                    crime_rate = feature['properties'].get('predictions', 0)
                    if crime_rate is None:
                        crime_rate = 0
                    return {
                        'fillColor': colors.rgb2hex(cmap_pred(norm_pred(crime_rate))),
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.3
                    }
                folium.GeoJson(
                    lsoa_gdf.to_json(),
                    name='LSOA predicted burglaries',
                    style_function=style_function_pred,
                    tooltip=folium.GeoJsonTooltip(
                        fields=[key, 'predictions'],
                        aliases=['LSOA Code:', 'Predicted burglaries:']
                    )
                ).add_to(m_borough_pred)
                folium.LayerControl().add_to(m_borough_pred)
                folium.LinearColormap(color_stops_pred, vmin=vmin_pred, vmax=vmax_pred, caption='Predicted Burglaries').add_to(m_borough_pred)
                st.markdown("**LSOA Predicted Burglaries**")
                st_folium(m_borough_pred, height=600, width=450, key=f"compare_lsoa_map_pred_{selected_compare_borough}")

            with col_lsoa_actual:
                m_borough_actual = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="cartodbpositron")
                def style_function_actual(feature):
                    crime_rate = feature['properties'].get('Burglaries amount', 0)
                    if crime_rate is None:
                        crime_rate = 0
                    return {
                        'fillColor': colors.rgb2hex(cmap_actual(norm_actual(crime_rate))),
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.3
                    }
                folium.GeoJson(
                    lsoa_gdf.to_json(),
                    name='LSOA actual burglaries',
                    style_function=style_function_actual,
                    tooltip=folium.GeoJsonTooltip(
                        fields=[key, 'Burglaries amount'],
                        aliases=['LSOA Code:', 'Actual burglaries:']
                    )
                ).add_to(m_borough_actual)
                folium.LayerControl().add_to(m_borough_actual)
                folium.LinearColormap(color_stops_actual, vmin=vmin_actual, vmax=vmax_actual, caption='Actual Burglaries').add_to(m_borough_actual)
                st.markdown("**LSOA Actual Burglaries**")
                st_folium(m_borough_actual, height=600, width=450, key=f"compare_lsoa_map_actual_{selected_compare_borough}")
# --- Show Historical ---
elif borough_mode == "Show Historical":
    st.subheader("Historical Burglaries by Borough")
    st.info("This section shows actual burglaries by borough for a selected month.")

    borough_gdf_hist_display = get_borough_gdf_copy()

    hist_month = st.selectbox("Select Month", avail_year_months, key="borough_hist_month")
    df_hist_month = df_recent[df_recent['year_month'] == hist_month]
    if lsoa_df is not None and 'LAD22NM' in lsoa_df.columns:
        lsoa_actual_hist = df_hist_month.merge(lsoa_df, left_on='LSOA_code', right_on='LSOA21CD', how='left')
        borough_actual_hist = lsoa_actual_hist.groupby('LAD22NM')['Burglaries amount'].sum().reset_index()
        borough_actual_hist.rename(columns={'LAD22NM': 'name', 'Burglaries amount': 'Actual Burglaries'}, inplace=True)
        borough_gdf_hist_display = borough_gdf_hist_display.merge(borough_actual_hist, on='name', how='left')
    else:
        borough_gdf_hist_display['Actual Burglaries'] = 0

    col_data_hist = borough_gdf_hist_display['Actual Burglaries'].dropna()
    vmin_hist = col_data_hist.min() if not col_data_hist.empty else 0
    vmax_hist = col_data_hist.quantile(0.95) if not col_data_hist.empty else 1
    cmap = cm.get_cmap('YlOrRd')
    norm = colors.Normalize(vmin=vmin_hist, vmax=vmax_hist)
    color_stops = get_colormap_hex_list('YlOrRd', 6)

    m_hist = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="cartodbpositron")
    def style_function_hist(feature):
        val = feature['properties'].get('Actual Burglaries', 0)
        if val is None:
            val = 0
        return {
            'fillColor': colors.rgb2hex(cmap(norm(val))),
            'color': 'black', 'weight': 0.5, 'fillOpacity': 0.3
        }
    folium.GeoJson(
        borough_gdf_hist_display,
        name="Boroughs",
        style_function=style_function_hist,
        tooltip=folium.GeoJsonTooltip(fields=['name', 'Actual Burglaries'])
    ).add_to(m_hist)
    folium.LinearColormap(color_stops, vmin=vmin_hist, vmax=vmax_hist, caption='Actual Burglaries').add_to(m_hist)
    hist_map_data = st_folium(m_hist, height=600, width=900, returned_objects=["last_active_drawing"], key="hist_borough_map")

    # --- Borough selection and LSOA map display for historical ---
    selected_hist_borough = None
    if hist_map_data and hist_map_data.get("last_active_drawing"):
        clicked_geom = shape(hist_map_data["last_active_drawing"]["geometry"])
        for idx, row in borough_gdf_hist_display.iterrows():
            if row['geometry'].equals_exact(clicked_geom, tolerance=1e-6):
                selected_hist_borough = row['name']
                break
        if not selected_hist_borough:
            pt = clicked_geom.centroid
            for idx, row in borough_gdf_hist_display.iterrows():
                if row['geometry'].contains(pt):
                    selected_hist_borough = row['name']
                    break

    if selected_hist_borough:
        st.success(f"Selected Borough: {selected_hist_borough}")
        lsoa_shp_path = f"./data/lsoashape/{selected_hist_borough}.shp"
        if os.path.exists(lsoa_shp_path):
            lsoa_gdf = gpd.read_file(lsoa_shp_path).to_crs(epsg=4326)
            # Merge historical data into LSOA GeoDataFrame
            hist_col = 'Burglaries amount'
            key = next((c for c in lsoa_gdf.columns if 'lsoa' in c.lower()), None)
            # Prepare a DataFrame with LSOA code and actual burglaries for this month
            df_hist_lsoa = df_hist_month[[ 'LSOA_code', hist_col ]].rename(columns={hist_col: 'Actual Burglaries'})
            # Remove duplicate columns if present
            if df_hist_lsoa.columns.duplicated().any():
                df_hist_lsoa = df_hist_lsoa.loc[:, ~df_hist_lsoa.columns.duplicated()]
            if lsoa_gdf.columns.duplicated().any():
                lsoa_gdf = lsoa_gdf.loc[:, ~lsoa_gdf.columns.duplicated()]
            # Ensure the geometry column is named 'geometry' and set as active
            geom_cols = [col for col in lsoa_gdf.columns if str(lsoa_gdf[col].dtype).startswith("geometry")]
            if "geometry" not in lsoa_gdf.columns and geom_cols:
                lsoa_gdf = lsoa_gdf.rename(columns={geom_cols[0]: "geometry"})
            if "geometry" in lsoa_gdf.columns:
                lsoa_gdf = lsoa_gdf.set_geometry("geometry")
            # Merge on LSOA code
            lsoa_gdf = lsoa_gdf.merge(
                df_hist_lsoa,
                how='left',
                left_on=key,
                right_on='LSOA_code'
            )
            # Color scale for historical data
            col_data_lsoa_hist = lsoa_gdf['Actual Burglaries'].dropna()
            if col_data_lsoa_hist.empty:
                vmin_lsoa_hist, vmax_lsoa_hist = 0, 1
            else:
                vmin_lsoa_hist = col_data_lsoa_hist.min()
                vmax_lsoa_hist = col_data_lsoa_hist.quantile(0.95)
            cmap = cm.get_cmap('YlOrRd')
            norm = colors.Normalize(vmin=vmin_lsoa_hist, vmax=vmax_lsoa_hist)
            color_stops = get_colormap_hex_list('YlOrRd', 6)
            centroid = lsoa_gdf.geometry.union_all().centroid

            m_borough_hist = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="cartodbpositron")
            def lsoa_style_fn_hist(feature):
                val = feature['properties'].get('Actual Burglaries', 0)
                if val is None:
                    val = 0
                return {
                    'fillColor': colors.rgb2hex(cmap(norm(val))),
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.3
                }
            folium.GeoJson(
                lsoa_gdf,
                name="LSOAs",
                style_function=lsoa_style_fn_hist,
                tooltip=folium.GeoJsonTooltip(
                    fields=[key, 'Actual Burglaries'],
                    aliases=['LSOA', 'Actual Burglaries']
                )
            ).add_to(m_borough_hist)
            folium.LayerControl().add_to(m_borough_hist)
            folium.LinearColormap(color_stops, vmin=vmin_lsoa_hist, vmax=vmax_lsoa_hist, caption='Actual Burglaries').add_to(m_borough_hist)

            # --- Show LSOA factors from merged_data if clicked ---
            selected_lsoa_code_hist = None
            if lsoa_map_data_hist and lsoa_map_data_hist.get("last_active_drawing"):
                clicked_geom = shape(lsoa_map_data_hist["last_active_drawing"]["geometry"])
                for idx, row in lsoa_gdf.iterrows():
                    if row['geometry'].equals_exact(clicked_geom, tolerance=1e-6):
                        selected_lsoa_code_hist = row[key]
                        break
                if not selected_lsoa_code_hist:
                    pt = clicked_geom.centroid
                    for idx, row in lsoa_gdf.iterrows():
                        if row['geometry'].contains(pt):
                            selected_lsoa_code_hist = row[key]
                            break

            if selected_lsoa_code_hist:
                st.markdown(f"### Factors for LSOA: {selected_lsoa_code_hist}")
                lsoa_factors = df_hist[df_hist['LSOA_code'] == selected_lsoa_code_hist]
                if not lsoa_factors.empty:
                    lsoa_factors = lsoa_factors.sort_values("date", ascending=False).iloc[0]
                    exclude_cols = ['geometry', 'date']
                    display_factors = lsoa_factors.drop(labels=[col for col in exclude_cols if col in lsoa_factors.index])

                    import matplotlib.pyplot as plt

                    # --- Pie Chart Data Preparation ---
                    # Dwelling type
                    dwelling_cols = [
                        "Dwelling type|Flat, maisonette or apartment (%)"
                    ]
                    dwelling_vals = [lsoa_factors.get(col, 0) for col in dwelling_cols]
                    dwelling_other = 100 - sum(dwelling_vals)
                    dwelling_labels = ["Flat, maisonette or apartment", "Other"]
                    dwelling_sizes = [dwelling_vals[0], dwelling_other]

                    # Ethnic group
                    ethnic_cols = [
                        "Ethnic Group|Asian/Asian British (%)",
                        "Ethnic Group|Black/African/Caribbean/Black British (%)",
                        "Ethnic Group|Mixed/multiple ethnic groups (%)",
                        "Ethnic Group|Other ethnic group (%)",
                        "Ethnic Group|White (%)"
                    ]
                    ethnic_labels = [
                        "Asian/Asian British",
                        "Black/African/Caribbean/Black British",
                        "Mixed/multiple ethnic groups",
                        "Other ethnic group",
                        "White"
                    ]
                    ethnic_vals = [lsoa_factors.get(col, 0) for col in ethnic_cols]

                    # Household composition
                    hh_cols = [
                        "Household Composition|% Couple household with dependent children",
                        "Household Composition|% Couple household without dependent children",
                        "Household Composition|% Lone parent household",
                        "Household Composition|% One person household",
                        "Household Composition|% Other multi person household"
                    ]
                    hh_labels = [
                        "Couple w/ children",
                        "Couple w/o children",
                        "Lone parent",
                        "One person",
                        "Other multi person"
                    ]
                    hh_vals = [lsoa_factors.get(col, 0) for col in hh_cols]

                    # Mid-year Population Estimates (Aged 0-65+)
                    pop_cols = [
                        "Mid-year Population Estimates|Aged 0-15",
                        "Mid-year Population Estimates|Aged 16-29",
                        "Mid-year Population Estimates|Aged 30-44",
                        "Mid-year Population Estimates|Aged 45-64",
                        "Mid-year Population Estimates|Aged 65+"
                    ]
                    pop_labels = [
                        "0-15", "16-29", "30-44", "45-64", "65+"
                    ]
                    pop_vals = [lsoa_factors.get(col, 0) for col in pop_cols]

                    # Tenure
                    tenure_cols = [
                        "Tenure|Owned outright (%)",
                        "Tenure|Owned with a mortgage or loan (%)",
                        "Tenure|Private rented (%)",
                        "Tenure|Social rented (%)"
                    ]
                    tenure_labels = [
                        "Owned outright",
                        "Owned w/ mortgage/loan",
                        "Private rented",
                        "Social rented"
                    ]
                    tenure_vals = [lsoa_factors.get(col, 0) for col in tenure_cols]

                    # Car or van availability (except cars per household)
                    car_cols = [
                        "Car or van availability|No cars or vans in household (%)",
                        "Car or van availability|1 car or van in household (%)",
                        "Car or van availability|2 cars or vans in household (%)",
                        "Car or van availability|3 cars or vans in household (%)",
                        "Car or van availability|4 or more cars or vans in household (%)"
                    ]
                    car_labels = [
                        "No cars/vans",
                        "1 car/van",
                        "2 cars/vans",
                        "3 cars/vans",
                        "4+ cars/vans"
                    ]
                    car_vals = [lsoa_factors.get(col, 0) for col in car_cols]

                    # Public Transport Accessibility Levels|%
                    ptal_cols = [
                        "Public Transport Accessibility Levels|% 0-1 (poor access)|Level3_65",
                        "Public Transport Accessibility Levels|% 2-3 (average access)|Level3_66",
                        "Public Transport Accessibility Levels|% 4-6 (good access)|Level3_67"
                    ]
                    ptal_labels = [
                        "PTAL 0-1 (poor)",
                        "PTAL 2-3 (average)",
                        "PTAL 4-6 (good)"
                    ]
                    ptal_vals = [lsoa_factors.get(col, 0) for col in ptal_cols]

                    # --- Pie Charts in 3 columns, above the factors table ---
                    col_pie1, col_pie2, col_pie3 = st.columns(3)
                    with col_pie1:
                        fig1, ax1 = plt.subplots()
                        ax1.pie(dwelling_sizes, labels=dwelling_labels, autopct='%1.1f%%', startangle=90)
                        ax1.set_title("Dwelling Type")
                        st.pyplot(fig1)

                    with col_pie2:
                        fig2, ax2 = plt.subplots()
                        ax2.pie(ethnic_vals, labels=ethnic_labels, autopct='%1.1f%%', startangle=90)
                        fig2.suptitle("Ethnic Group")
                        st.pyplot(fig2)

                    with col_pie3:
                        fig3, ax3 = plt.subplots()
                        ax3.pie(hh_vals, labels=hh_labels, autopct='%1.1f%%', startangle=90)
                        ax3.set_title("Household Composition")
                        st.pyplot(fig3)

                    col_pie4, col_pie5, col_pie6 = st.columns(3)
                    with col_pie4:
                        fig4, ax4 = plt.subplots()
                        ax4.pie(pop_vals, labels=pop_labels, autopct='%1.1f%%', startangle=90)
                        ax4.set_title("Population Age Groups")
                        st.pyplot(fig4)

                    with col_pie5:
                        fig5, ax5 = plt.subplots()
                        ax5.pie(tenure_vals, labels=tenure_labels, autopct='%1.1f%%', startangle=90)
                        fig5.suptitle("Tenure")
                        st.pyplot(fig5)

                    with col_pie6:
                        fig6, ax6 = plt.subplots()
                        ax6.pie(car_vals, labels=car_labels, autopct='%1.1f%%', startangle=90)
                        ax6.set_title("Car or Van Availability")
                        st.pyplot(fig6)

                    # Last row for PTAL
                    col_pie7, _, _ = st.columns(3)
                    with col_pie7:
                        fig7, ax7 = plt.subplots()
                        ax7.pie(ptal_vals, labels=ptal_labels, autopct='%1.1f%%', startangle=90)
                        ax7.set_title("Public Transport Accessibility Levels (%)")
                        st.pyplot(fig7)

                    st.table(display_factors)
                else:
                    st.info("No factor data found for this LSOA in merged_data.")
