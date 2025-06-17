# main.py
"""
Streamlit dashboard – London burglary hotspots
Optimised, single-file edition.
"""

# --------------------------------------------------------------------------- #
# Imports                                                                     #
# --------------------------------------------------------------------------- #
import os
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import plotly.express as px
import shapely.geometry as sgeom
import streamlit as st
from streamlit.components.v1 import html as st_html
from streamlit_folium import st_folium

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
DATA_DIR = Path("data")
ROUTE_DIR = Path("routes")
TILESET = "cartodbpositron"
COLORMAP = cm.get_cmap("YlOrRd")
COLOR_STOPS = [mcolors.rgb2hex(COLORMAP(i/5)) for i in range(6)]

FILEPATHS = {
    "predictions": "./model/sample_predictions.parquet",
    "history": "merged_data.parquet",
    "lookup": DATA_DIR / "LSOA_(2011)_to_LSOA_(2021)_Exact_Fit_Lookup_for_EW_(V3).csv",
    "lsoa_gpkg": DATA_DIR / "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4.gpkg",
    "boroughs": DATA_DIR / "London_Boroughs.gpkg",
    "wards": DATA_DIR / "London_Ward.shp",
}

PIE_CHART_LABELS = {
    "Ethnic Group Composition": {
        "cols": [
            "Ethnic Group|Asian/Asian British (%)",
            "Ethnic Group|BAME (%)",
            "Ethnic Group|Black/African/Caribbean/Black British (%)",
            "Ethnic Group|Mixed/multiple ethnic groups (%)",
            "Ethnic Group|Other ethnic group (%)",
            "Ethnic Group|White (%)",
        ],
        "labels": [
            "Asian/Asian British",
            "BAME",
            "Black/African/Caribbean/Black British",
            "Mixed/multiple ethnic groups",
            "Other ethnic group",
            "White",
        ],
    },
    "Household Composition": {
        "cols": [
            "Household Composition|% Couple household with dependent children",
            "Household Composition|% Couple household without dependent children",
            "Household Composition|% Lone parent household",
            "Household Composition|% One person household",
            "Household Composition|% Other multi person household",
        ],
        "labels": [
            "Couple w/ children",
            "Couple w/o children",
            "Lone parent",
            "One person",
            "Other multi person",
        ],
    },
    "Population Age Estimates": {
        "cols": [
            "Mid-year Population Estimates|Aged 0-15",
            "Mid-year Population Estimates|Aged 16-29",
            "Mid-year Population Estimates|Aged 30-44",
            "Mid-year Population Estimates|Aged 45-64",
            "Mid-year Population Estimates|Aged 65+",
        ],
        "labels": ["0-15", "16-29", "30-44", "45-64", "65+"],
    },
    "Tenure (% of households)": {
        "cols": [
            "Tenure|Owned outright (%)",
            "Tenure|Owned with a mortgage or loan (%)",
            "Tenure|Private rented (%)",
            "Tenure|Social rented (%)",
        ],
        "labels": [
            "Owned outright",
            "Owned w/ mortgage/loan",
            "Private rented",
            "Social rented",
        ],
    },
    "Car or Van Availability": {
        "cols": [
            "Car or van availability|1 car or van in household (%)",
            "Car or van availability|2 cars or vans in household (%)",
            "Car or van availability|3 cars or vans in household (%)",
            "Car or van availability|4 or more cars or vans in household (%)",
        ],
        "labels": ["1 car/van", "2 cars/vans", "3 cars/vans", "4+ cars/vans"],
    },
    "PTAL Levels": {
        "cols": [
            "Public Transport Accessibility Levels|Number of people in each PTAL level:|0",
            "Public Transport Accessibility Levels|Number of people in each PTAL level:|1a",
            "Public Transport Accessibility Levels|Number of people in each PTAL level:|1b",
            "Public Transport Accessibility Levels|Number of people in each PTAL level:|2",
            "Public Transport Accessibility Levels|Number of people in each PTAL level:|3",
            "Public Transport Accessibility Levels|Number of people in each PTAL level:|4",
            "Public Transport Accessibility Levels|Number of people in each PTAL level:|5",
            "Public Transport Accessibility Levels|Number of people in each PTAL level:|6a",
            "Public Transport Accessibility Levels|Number of people in each PTAL level:|6b",
        ],
        "labels": ["PTAL 0", "PTAL 1a", "PTAL 1b", "PTAL 2", "PTAL 3", "PTAL 4", "PTAL 5", "PTAL 6a", "PTAL 6b"],
    },
}

FEATURE_DESCRIPTIONS = {
    "public_transport_accessibility": (
        "LSOAs with more stations or stops see slightly higher burglary levels, "
        "as greater mobility expands both offender and victim movements."
    ),
    "race_composition": (
        "Areas with larger Black or minority-ethnic populations tend to have higher "
        "burglary rates, reflecting socio-economic and cohesion factors linked to property crime."
    ),
    "single_person_households": (
        "Neighbourhoods dominated by one-person homes experience substantially more burglaries, "
        "while those with more couples show lower risk due to stronger informal guardianship."
    ),
    "tenure_mix": (
        "Districts with mixed or mostly private-renting stock have higher burglary rates than "
        "predominantly social-rent or owner-occupied areas, suggesting tenant-focused outreach could mitigate risk."
    ),
    "lsoa_size_and_borders": (
        "Units bordering many neighbours have more burglaries, whereas larger, lower-density "
        "LSOAs tend to see fewer incidents."
    ),
    "housing_services_barriers": (
        "Higher barriers to housing and services (poor transport, few amenities) correlate with "
        "greater burglary vulnerability, pointing to social-support interventions."
    ),
    "youth_population_share": (
        "Greater proportions of residents under 30 drive up burglary levels, while family- or retiree-heavy "
        "areas see fewer incidents."
    ),
    "seasonal_pattern": (
        "Burglaries exhibit only modest monthly swings, so timing patrols by month yields limited improvements."
    ),
    "covid_lockdowns": (
        "Strict lockdowns cut burglaries by over 20 % via increased occupancy and mobility limits."
    ),
    "post_lockdown_levels": (
        "After restrictions lifted, burglary rates stayed below pre-pandemic norms, indicating lasting behavioural or economic shifts."
    ),
    "neighboring_crime_spillover": (
        "High burglary counts in adjacent LSOAs predict elevated local risk, underscoring multi-area hotspot mapping."
    ),
    "long_term_trend": (
        "No consistent upward or downward drift over years—resource shifts should follow covariate changes, not gradual trends."
    ),
    "recent_burglaries": (
        "Last month’s burglary count remains the strongest short-term predictor of next month’s incidents, "
        "highlighting the value of near-real-time data."
    ),
}

# --------------------------------------------------------------------------- #
# I/O & caching                                                               #
# --------------------------------------------------------------------------- #


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(FILEPATHS["predictions"]
                         ).reset_index(names="LSOA21CD")
    return df.rename(columns={"median": "predictions"})


@st.cache_data
def load_history() -> pd.DataFrame:
    df = pd.read_parquet(FILEPATHS["history"])
    lsoa_col = next(c for c in df.columns if "lsoa" in c.lower())
    return (df
            .rename(columns={lsoa_col: "LSOA_code"})
            .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce")))


@st.cache_data
def load_lookup() -> pd.DataFrame:
    return pd.read_csv(FILEPATHS["lookup"])[["LSOA21CD", "LAD22NM"]]


@st.cache_data
def load_geodata(level: str) -> gpd.GeoDataFrame:
    path = FILEPATHS["boroughs"] if level == "Borough" else FILEPATHS["wards"]
    return gpd.read_file(path).to_crs(epsg=4326)


@st.cache_data
def load_lsoa_gdf() -> gpd.GeoDataFrame:
    return gpd.read_file(FILEPATHS["lsoa_gpkg"]).to_crs(epsg=4326)

# --------------------------------------------------------------------------- #
# Aggregations                                                                #
# --------------------------------------------------------------------------- #


@st.cache_data
def predictions_to_boroughs() -> gpd.GeoDataFrame:
    df_pred = load_predictions()
    lookup_df = load_lookup()
    borough_gdf = load_geodata("Borough").rename(columns={"name": "NAME"})
    borough_totals = (df_pred
                      .merge(lookup_df, on="LSOA21CD", how="left")
                      .dropna(subset=["LAD22NM"])
                      .groupby("LAD22NM", as_index=False)["predictions"]
                      .sum()
                      .rename(columns={"LAD22NM": "NAME", "predictions": "Burglaries"}))
    return borough_gdf.merge(borough_totals, on="NAME", how="left")


@st.cache_data
def predictions_to_wards() -> gpd.GeoDataFrame:
    df_pred = load_predictions()
    lsoa_gdf = load_lsoa_gdf()
    ward_gdf = load_geodata("Ward")[["NAME", "geometry"]]

    gdf_lsoa = lsoa_gdf.merge(df_pred, on="LSOA21CD")
    gdf_lsoa["lsoa_area"] = gdf_lsoa.area

    intersections = gpd.overlay(gdf_lsoa, ward_gdf)
    intersections["weight"] = intersections.area / intersections["lsoa_area"]

    ward_totals = (intersections
                   .groupby("NAME")["predictions"]
                   .apply(lambda s: (s * intersections.loc[s.index, "weight"]).sum())
                   .round(0)
                   .reset_index()
                   .rename(columns={"predictions": "Burglaries"}))

    return ward_gdf.merge(ward_totals, on="NAME", how="left")

# --------------------------------------------------------------------------- #
# Map helpers                                                                 #
# --------------------------------------------------------------------------- #
# @st.cache_data(show_spinner=True)


def build_choropleth(gdf, column, layer_name):
    vmin, vmax = gdf[column].min(), gdf[column].quantile(0.99)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    m = folium.Map(location=[gdf.geometry.centroid.y.mean(),
                             gdf.geometry.centroid.x.mean()],
                   zoom_start=10, tiles=TILESET)
    folium.GeoJson(
        gdf,
        name=layer_name,
        style_function=lambda feat: {
            "fillColor": mcolors.rgb2hex(COLORMAP(norm(feat["properties"].get(column, 0)))),
            "weight": 0.5,
            "color": "black",
            "fillOpacity": 0.4,
        },
        highlight_function=lambda _: {"weight": 2, "color": "blue"},
        tooltip=folium.GeoJsonTooltip(fields=["NAME", column],
                                      aliases=["Name", column])
    ).add_to(m)
    folium.LinearColormap(COLOR_STOPS, vmin=vmin, vmax=vmax,
                          caption=column).add_to(m)
    return m


def draw_map(gdf, column, key):
    m = build_choropleth(gdf, column, key)
    return st_folium(m, height=600, use_container_width=True,
                     returned_objects=["last_active_drawing"], key=key)

# --------------------------------------------------------------------------- #
# Simple utilities                                                            #
# --------------------------------------------------------------------------- #


def get_selected_occupation(map_data: dict, gdf: gpd.GeoDataFrame,
                            tolerance: float = 1e-6) -> str | None:
    """
    Extract NAME of polygon clicked on the Folium map.
    """
    if not map_data or not map_data.get("last_active_drawing"):
        return None
    geom_json = map_data["last_active_drawing"]["geometry"]
    point = sgeom.shape(geom_json).centroid
    hit = gdf[gdf.geometry.buffer(tolerance).contains(point)]
    return None if hit.empty else hit.iloc[0]["NAME"]


@st.cache_data
def draw_patrol_route(name: str, level: str):
    fname = name.lower().replace(" ", "_") + ".html"
    folder = "borough_routes" if level == "Borough" else "ward_routes"
    path = ROUTE_DIR / folder / fname
    if path.exists():
        st_html(path.read_text(encoding="utf-8"), height=600)
    else:
        st.warning(f"No patrol route found at {path}")

# --------------------------------------------------------------------------- #
# Chart helpers                                                               #
# --------------------------------------------------------------------------- #


@st.cache_data
def timeline_figure(df: pd.DataFrame,
                    date_col="date",
                    value_col="Burglaries amount",
                    title="Burglaries Timeline"):
    df_sorted = df.sort_values(date_col)
    fig = px.line(df_sorted, x=date_col, y=value_col,
                  title=title, markers=True,
                  labels={date_col: "Date", value_col: "Burglaries"})
    fig.update_layout(hovermode="x unified",
                      xaxis=dict(rangeslider=dict(visible=True)))
    return fig


@st.cache_data
def pie_figure(values: list[float], labels: list[str], title: str):
    return px.pie(names=labels, values=values, title=title, hole=0)


# --------------------------------------------------------------------------- #
# Streamlit page                                                              #
# --------------------------------------------------------------------------- #
st.set_page_config("London burglary hotspots", layout="wide")
st.title("London burglary hotspot explorer")

# ---------- Load data ------------------------------------------------------ #
df_pred = load_predictions()
df_hist = load_history()
lookup_df = load_lookup()
borough_gdf = predictions_to_boroughs()
ward_gdf = predictions_to_wards()
lsoa_base_gdf = load_lsoa_gdf()
lsoa_pred_gdf = lsoa_base_gdf.merge(df_pred, on="LSOA21CD")
lsoa_pred_gdf = lsoa_pred_gdf.rename(columns={"LSOA21CD": "NAME",
                                              "predictions": "Burglaries"})

# Create the “year-month” column once
df_hist = df_hist.assign(
    year_month=lambda d: d["date"].dt.to_period("M").dt.strftime("%Y-%B"))
avail_year_months = (df_hist["year_month"]
                     .drop_duplicates()
                     .pipe(pd.Series.sort_values, key=lambda s: pd.to_datetime(s, format="%Y-%B"))
                     .tolist())

# ---------- Tabs ----------------------------------------------------------- #
tab_pred, tab_hist, tab_compare, tab_explain = st.tabs(
    ["Burglary Predictions", "Historical Overview", "Last month comparison", "Model Feature Explanations"])

# ----- Predicted ----------------------------------------------------------- #
with tab_pred:
    sub_b, sub_w, sub_l = st.tabs(["Borough", "Ward", "LSOA"])

    # Borough
    with sub_b:
        st.subheader("Predicted burglaries by borough")
        b_map = draw_map(borough_gdf, "Burglaries", "borough_pred")
        if (sel := get_selected_occupation(b_map, borough_gdf)):
            st.subheader(f"Patrol route for **{sel}**")
            draw_patrol_route(sel, "Borough")

    # Ward
    with sub_w:
        st.subheader("Predicted burglaries by ward")
        w_map = draw_map(ward_gdf, "Burglaries", "ward_pred")
        if (sel := get_selected_occupation(w_map, ward_gdf)):
            st.subheader(f"Patrol route for **{sel}**")
            draw_patrol_route(sel, "Ward")

    # LSOA
    with sub_l:
        st.subheader("Predicted burglaries by LSOA")
        l_map = draw_map(lsoa_pred_gdf, "Burglaries", "lsoa_pred")
        if (sel := get_selected_occupation(l_map, lsoa_pred_gdf)):
            st.success(f"Selected LSOA: {sel}")
            lsoa_hist = df_hist.loc[df_hist["LSOA_code"] == sel]
            st.plotly_chart(timeline_figure(lsoa_hist),
                            use_container_width=True)

            for title, cfg in PIE_CHART_LABELS.items():
                values = lsoa_hist[cfg["cols"]].iloc[-1].tolist()
                st.plotly_chart(pie_figure(values, cfg["labels"], title),
                                use_container_width=True)

            st.write("### Last available row")
            st.dataframe(lsoa_hist.sort_values(
                "date").drop(columns="geometry").tail(1).T)

# ----- Historical ---------------------------------------------------------- #
with tab_hist:
    st.subheader("Historical burglaries by borough")
    month = st.selectbox("Choose month", avail_year_months,
                         index=len(avail_year_months)-1)
    borough_hist = (df_hist.query("year_month == @month")
                    .merge(lookup_df, left_on="LSOA_code", right_on="LSOA21CD", how="left")
                    .groupby("LAD22NM", as_index=False)["Burglaries amount"].sum()
                    .rename(columns={"LAD22NM": "NAME"}))
    borough_hist_gdf = load_geodata("Borough").rename(columns={"name": "NAME"})
    borough_hist_gdf = borough_hist_gdf.merge(
        borough_hist, on="NAME", how="left")

    h_map = draw_map(borough_hist_gdf, "Burglaries amount", "borough_hist")
    if (sel_b := get_selected_occupation(h_map, borough_hist_gdf)):
        st.success(f"Selected Borough: {sel_b}")
        path = DATA_DIR / "lsoashape" / f"{sel_b}.shp"
        if path.exists():
            lsoa_shape = gpd.read_file(path).to_crs(epsg=4326)
            key = next(c for c in lsoa_shape.columns if "lsoa" in c.lower())
            borough_lsoa = (lsoa_shape
                            .merge(df_hist[["LSOA_code", "Burglaries amount"]]
                                   .drop_duplicates("LSOA_code"),
                                   left_on=key, right_on="LSOA_code", how="left")
                            .rename(columns={"Burglaries amount": "Burglaries"}))
            draw_map(borough_lsoa.rename(
                columns={key: "NAME"}), "Burglaries", "lsoa_hist")

# ----- Compare ------------------------------------------------------------- #
with tab_compare:
    st.subheader("Compare predicted vs actual – latest month")
    latest_month = avail_year_months[-1]

    borough_actual = (df_hist.query("year_month == @latest_month")
                      .merge(lookup_df, left_on="LSOA_code", right_on="LSOA21CD", how="left")
                      .groupby("LAD22NM", as_index=False)["Burglaries amount"].sum()
                      .rename(columns={"LAD22NM": "NAME", "Burglaries amount": "Actual"}))

    borough_compare_gdf = load_geodata(
        "Borough").rename(columns={"name": "NAME"})
    borough_compare_gdf = borough_compare_gdf.merge(
        borough_actual, on="NAME", how="left")
    col_pred, col_act = st.columns(2)
    with col_pred:
        st.markdown("**Predicted Burglaries**")
        draw_map(borough_gdf, "Burglaries", "compare_pred")
    with col_act:
        st.markdown(f"**Actual Burglaries ({latest_month})**")
        draw_map(borough_compare_gdf, "Actual", "compare_act")

# ---------- Feature explanations tab -------------------------------------- #
with tab_explain:
    st.header("📑 Model feature explanations")
    st.write(
        "Below are the most important variables in the monthly burglary-risk model "
        "and a short, plain-English note on how each one influences predictions."
    )
    for key, desc in FEATURE_DESCRIPTIONS.items():
        pretty = key.replace("_", " ").title()
        st.markdown(f"**{pretty}**  \n{desc}")
