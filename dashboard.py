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
    "Ward": r"./data/ward/London_Ward.shp"
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
    # Use matplotlib colormap for fillColor, but use folium.LinearColormap for legend
    cmap = cm.get_cmap('YlOrRd')
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

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
    cmap = cm.get_cmap('YlOrRd')
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    folium_cmap = folium.LinearColormap(['green', 'yellow', 'red'], vmin=vmin, vmax=vmax)
    folium_cmap.caption = 'Predicted Burglaries'
    folium_cmap.add_to(m)
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
                st_folium(m_borough, height=600, width=900, key=f"lsoa_map_{selected_borough}_predicted") # Unique key
        else:
            st.warning(f"No LSOA shapefile found for {selected_borough} in ./data/lsoashape/")
    else:
        st.info("Click a borough on the map above to see its LSOAs and predicted crime.")

# --- Compare Predicted vs Actual ---
elif borough_mode == "Compare Predicted vs Actual":
    st.subheader("Compare Predicted vs Actual Burglaries by Borough")
    st.info("This section shows a side-by-side comparison of predicted and actual burglaries for the latest available month.")

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
        # Use matplotlib colormap for fillColor, but do NOT call add_to for matplotlib colormap
        cmap_pred = cm.get_cmap('YlOrRd')
        norm_pred = colors.Normalize(vmin=vmin_pred, vmax=vmax_pred)

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
        st_folium(m_pred, height=500, width=450, key="compare_pred_borough_map")

    with col2:
        st.markdown(f"**Actual Burglaries ({latest_month})**")
        col_data_actual = borough_gdf_actual_display['Actual Burglaries'].dropna()
        vmin_actual = col_data_actual.min() if not col_data_actual.empty else 0
        vmax_actual = col_data_actual.quantile(0.95) if not col_data_actual.empty else 1
        cmap_actual = cm.get_cmap('YlOrRd')
        norm_actual = colors.Normalize(vmin=vmin_actual, vmax=vmax_actual)

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
        st_folium(m_actual, height=500, width=450, key="compare_actual_borough_map")

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
    # Remove cmap_hist.add_to(m_hist) -- not valid for matplotlib colormap
    st_folium(m_hist, height=600, width=900, key="hist_borough_map")

    # --- LSOA Details and Route Viewer (ONLY in "Show Historical" mode) ---
    selected_borough = st.session_state["selected_borough"]
    if selected_borough:
        st.success(f"Selected Borough: {selected_borough}")
        # --- LSOA Map for Selected Borough ---
        lsoa_shp_path = f"./data/lsoashape/{selected_borough}.shp"
        if os.path.exists(lsoa_shp_path):
            lsoa_gdf = gpd.read_file(lsoa_shp_path).to_crs(epsg=4326)
            
            # Merge historical data into LSOA GeoDataFrame
            hist_col = 'Burglaries amount'
            df_hist_lsoa = df_hist_month.reset_index().rename(columns={'index': 'LSOA_code', hist_col: 'Actual Burglaries'})
            key = next((c for c in lsoa_gdf.columns if 'lsoa' in c.lower()), None)
            lsoa_gdf = lsoa_gdf.merge(df_hist_lsoa, how='left', left_on=key, right_on='LSOA_code')

            # Color scale for historical data
            col_data_lsoa_hist = lsoa_gdf['Actual Burglaries'].dropna()
            if col_data_lsoa_hist.empty:
                vmin_lsoa_hist, vmax_lsoa_hist = 0, 1
            else:
                vmin_lsoa_hist = col_data_lsoa_hist.min()
                vmax_lsoa_hist = col_data_lsoa_hist.quantile(0.95)
            cmap = cm.get_cmap('YlOrRd')
            norm = colors.Normalize(vmin=vmin_lsoa_hist, vmax=vmax_lsoa_hist)

            # Map centered on borough
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
            st_folium(m_borough_hist, height=600, width=900, key=f"lsoa_map_{selected_borough}_historical") # Unique key
        else:
            st.warning(f"No LSOA shapefile found for {selected_borough} in ./data/lsoashape/")
    else:
        st.info("Click a borough on the map above to see its LSOAs and historical crime data.")