import streamlit as st  # UI
import geopandas as gpd  # Spatial data
import pandas as pd  # Tabular data
import folium  # Interactive mapping
from folium import GeoJson, GeoJsonTooltip, LayerControl
from streamlit_folium import folium_static  # Embed Folium in Streamlit
import branca.colormap as cm  # Color scales
import os  # Filesystem operations

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
st.sidebar.header("🔄 Upload Data Files")

# Default: automatically detect files
if os.path.exists("./model/sample_predictions.parquet") and os.path.exists("./merged_data.parquet"):
    pred_file = "./model/sample_predictions.parquet"
    hist_file = "./merged_data.parquet"
else:
    st.sidebar.markdown("**Note:** No default files found. Please upload your own data.")

    pred_file = st.sidebar.file_uploader("Predictions (.parquet)", type="parquet")
    hist_file = st.sidebar.file_uploader("Historical (.parquet)", type="parquet")

    if not pred_file or not hist_file:
        st.warning("Upload both prediction and historical data to proceed.")
        st.stop()

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

# --- Controls ---
geo_level = st.selectbox("Geography Level", ["LSOA", "Borough", "Ward"])
if geo_level == "LSOA":
    view_mode = st.radio("View Mode", [
        "Prediction (Current Month)",
        "Actual (One Past Month)",
        "Compare Two Past Months"
    ])
else:
    view_mode = "Actual (One Past Month)"
sel1 = sel2 = None
if geo_level == "LSOA" and view_mode in ["Actual (One Past Month)"]:
    sel1 = st.selectbox("Select Month", avail_year_months)
if geo_level == "LSOA" and view_mode == "Compare Two Past Months":
    sel1 = st.selectbox("First Month", avail_year_months, index=0)
    sel2 = st.selectbox("Second Month", avail_year_months, index=1)

# --- File paths ---
paths = {
    "LSOA": r"./data/lsoashape/LSOA_2021_EW_BSC_V4.shp",
    "Borough": r"./data/London_Boroughs.gpkg",
    "Ward": r"./data/ward/London_Ward.shp"
}
routes_folders = {
    "Borough": r"./routes/borough_routes",
    "Ward": r"./routes/ward_routes"
}

# --- Load geodata ---
def load_geodata(level):
    """
    Loads shapefiles (.shp for LSOA/Ward, .tab for Borough) and projects to WGS84 (lat/lon) for mapping.
    """
    folder = paths[level]
    if level == "Borough":
        gdf = gpd.read_file(paths["Borough"])
    else:
        
        gdf = gpd.read_file(folder)
    return gdf.to_crs(epsg=4326)

# Simplify geometry
gdf = load_geodata(geo_level)
gdf['geometry'] = gdf['geometry'].simplify(0.0015, preserve_topology=True)

# --- Cache function ---
@st.cache_data
def get_month_df(df, label):
    return df[df['year_month'] == label][['LSOA_code', 'Burglaries amount']].rename(columns={'Burglaries amount': label})

# --- Merge predictions for LSOA ---
def prepare_pred(df_geo):
    if geo_level != 'LSOA':
        return df_geo
    pcol = df_pred.columns[0]
    dfp = df_pred.reset_index().rename(columns={'index':'LSOA_code', pcol:'predictions'})
    key = next((c for c in df_geo.columns if 'lsoa' in c.lower()), None)
    return df_geo.merge(dfp, how='left', left_on=key, right_on='LSOA_code')

# --- Map renderer ---
def render_map(gdfm, col, caption, use_tooltip=True):
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="cartodbpositron")
    # Base boundaries
    GeoJson(
        gdfm,
        name="Boundaries",
        style_function=lambda f: {'fillColor': 'none', 'color': 'gray', 'weight': 1}
    ).add_to(m)

    # Choropleth layer if requested
    if col and col in gdfm.columns:
        col_data = gdfm[col].dropna().astype(float)
        if not col_data.empty:
            # create color map
            cmap = cm.linear.YlOrRd_09.scale(col_data.min(), col_data.max())
            cmap.caption = caption
            def style_fn(feature):
                raw = feature['properties'].get(col)
                val = raw if (raw is not None) else 0
                return {'fillColor': cmap(val), 'color': 'black', 'weight': 0.3, 'fillOpacity': 0.7}
            layer = GeoJson(
                gdfm,
                name=caption,
                style_function=style_fn
            )
            if use_tooltip:
                layer.add_child(GeoJsonTooltip(fields=[col], aliases=[caption], localize=True))
            layer.add_to(m)
            cmap.add_to(m)

    # Routes overlay for Borough/Ward
    if geo_level in routes_folders:
        for fn in os.listdir(routes_folders[geo_level]):
            if fn.lower().endswith('.shp'):
                rg = gpd.read_file(os.path.join(routes_folders[geo_level], fn)).to_crs(epsg=4326)
                GeoJson(
                    rg,
                    name="Police Routes",
                    style_function=lambda _: {'color': 'blue', 'weight': 2}
                ).add_to(m)

    LayerControl().add_to(m)
    return m

# --- Display logic ---
gdfp = prepare_pred(gdf)
if view_mode == "Prediction (Current Month)":
    st.subheader("Predicted Burglaries")
    folium_static(render_map(gdfp, 'predictions', 'Predicted Burglaries'), height=600)
elif view_mode == "Actual (One Past Month)" and sel1:
    if geo_level == 'LSOA':
        dfm = get_month_df(df_recent, sel1)
        key = next((c for c in gdf.columns if 'lsoa' in c.lower()), None)
        g = gdf.merge(dfm, how='left', left_on=key, right_on='LSOA_code')
        st.subheader(f"Actual Burglaries - {sel1}")
        folium_static(render_map(g, sel1, f"Actual {sel1}"), height=600)
    else:
        st.subheader(f"{geo_level} Map and Routes")
        folium_static(render_map(gdf, None, f"{geo_level} Boundaries"), height=600)
elif view_mode == "Compare Two Past Months" and sel1 and sel2:
    if geo_level == 'LSOA':
        # Compare two months side by side using columns
        df1 = get_month_df(df_recent, sel1)
        df2 = get_month_df(df_recent, sel2)
        # merge both months into one DF
        df_comb = pd.merge(df1, df2, on='LSOA_code', how='outer')
        key = next((c for c in gdf.columns if 'lsoa' in c.lower()), None)
        g12 = gdf.merge(df_comb, how='left', left_on=key, right_on='LSOA_code')
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(sel1)
            folium_static(render_map(g12, sel1, sel1, use_tooltip=False), height=600)
        with col2:
            st.subheader(sel2)
            folium_static(render_map(g12, sel2, sel2, use_tooltip=False), height=600)
    else:
        st.subheader(f"{geo_level} Map and Routes")
        folium_static(render_map(gdf, None, f"{geo_level} Boundaries"), height=600)

# --- Route Viewer for HTML maps ---
    st.markdown(f"## 🚓 {geo_level} Police Routes Viewer 🗺️")
    import streamlit.components.v1 as components
    htmls = {f.replace('.html','').title():f for f in os.listdir(routes_folders[geo_level]) if f.endswith('.html')}
    sel = st.selectbox(f"Select {geo_level} Route", sorted(htmls.keys()))
    if sel:
        components.html(
            open(os.path.join(routes_folders[geo_level], htmls[sel]), 'r', encoding='utf-8').read(),
            height=600, scrolling=True
        )
