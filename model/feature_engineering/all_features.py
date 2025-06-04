import geopandas as gpd
import numpy as np
import pandas as pd
from feature_engineering.spatial_features import (
    create_dynamic_spatial_features, create_static_spatial_features)
from utils.utils import detect_new_columns, downcast_numeric


def create_all_features(gdf: gpd.GeoDataFrame, static, dynamic) -> gpd.GeoDataFrame:
    # Split existing columns into static and dynamic

    # Create time-related features
    with detect_new_columns(gdf) as time_features:
        t0 = gdf['period'].min()
        # convert each period to the number of months since t0
        time_idx = (
            (gdf['period'].dt.year - t0.year) * 12
            + (gdf['period'].dt.month - t0.month)
        )
        # center & scale
        gdf['time_s'] = (time_idx - time_idx.mean()) / time_idx.std()

    # Create seasonal features
    with detect_new_columns(gdf) as seasonal_features:
        gdf['month_sin'] = np.sin(
            2*np.pi*gdf['period'].dt.month/12).astype("float32")
        gdf['month_cos'] = np.cos(
            2*np.pi*gdf['period'].dt.month/12).astype("float32")

        lockdown_start = pd.Timestamp("2020-03-23")
        lockdown_cutoff = pd.Timestamp("2020-03-01")

        restrictions_end = pd.Timestamp("2022-02-24")
        post_corona_start = restrictions_end + pd.Timedelta(days=1)

        gdf["during_corona"] = (
            (gdf["period"] >= lockdown_cutoff) &
            (gdf["period"] <= restrictions_end)
        ).astype("int8")
        gdf["post_corona"] = (
            gdf["period"] >= post_corona_start).astype("int8")

    # Create temporal features
    with detect_new_columns(gdf) as temporal_features:
        gdf['lag_1'] = gdf.groupby('occupation_idx', observed=True)[
            'burglaries'].shift(1)
        gdf['lag_2'] = gdf.groupby('occupation_idx', observed=True)[
            'burglaries'].shift(2)
        gdf['lag_3'] = gdf.groupby('occupation_idx', observed=True)[
            'burglaries'].shift(3)
        gdf['lag_4'] = gdf.groupby('occupation_idx', observed=True)[
            'burglaries'].shift(12)

    # Create static spatial features
    with detect_new_columns(gdf) as static_spatial:
        static_cols, neighbor_dict = create_static_spatial_features(gdf)

        gdf["area"] = gdf["occupation_idx"].map(static_cols["area"])
        gdf["n_neighbors"] = gdf["occupation_idx"].map(
            static_cols["n_neighbors"])
        gdf["shared_length"] = gdf["occupation_idx"].map(
            static_cols["shared_length"])

    static.extend(static_spatial)

    # Create dynamic spatial features
    with detect_new_columns(gdf) as dynamic_spatial:
        # !!!
        # Pretty, but runs by assumption that they are indexed correctly
        # Meaning that they are sorted first by occupation_idx, and then time_s
        # !!!
        df = create_dynamic_spatial_features(gdf, neighbor_dict)
        gdf["lag1_sum_neighbors"] = df["lag1_sum_neighbors"].fillna(0)
        gdf["lag1_mean_neighbors"] = df["lag1_mean_neighbors"].fillna(0)
        gdf["lag1_median_neighbors"] = df["lag1_median_neighbors"].fillna(0)

    return (
        gdf.dropna().drop(columns=["geometry", "period"]).pipe(downcast_numeric),
        static, dynamic, seasonal_features, time_features,
        temporal_features, dynamic_spatial
        )
