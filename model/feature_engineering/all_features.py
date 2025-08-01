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
        time_idexs2 = time_idx ** 2
        time_idxlog = np.log1p(time_idx)
        # center & scale
        gdf['time_s'] = (time_idx - time_idx.mean()) / time_idx.std()
        # gdf['time_s2'] = (time_idexs2 - time_idexs2.mean()) / time_idexs2.std()
        gdf['time_log'] = time_idxlog
        # gdf['time_log'] = (time_idxlog - time_idxlog.mean()) / time_idxlog.std()
    time_features.remove("time_s")
    # Create seasonal features
    with detect_new_columns(gdf) as seasonal_features:
        gdf['month_sin'] = np.sin(
            2*np.pi*gdf['period'].dt.month/12).astype("float32")

        lockdown_cutoff = pd.Timestamp("2020-02-01")

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
          # Rolling mean features for longer temporal patterns
        gdf["roll_3_mean"] = (
            gdf.groupby("occupation_idx", observed=True)["burglaries"]
            .apply(lambda s: s.shift(1).rolling(window=3).mean())
            .reset_index(level=0, drop=True)
        )
        gdf["roll_6_mean"] = (
            gdf.groupby("occupation_idx", observed=True)["burglaries"]
            .apply(lambda s: s.shift(1).rolling(window=6).mean())
            .reset_index(level=0, drop=True)
        )
        gdf["roll_12_mean"] = (
            gdf.groupby("occupation_idx", observed=True)["burglaries"]
            .apply(lambda s: s.shift(1).rolling(window=12).mean())
            .reset_index(level=0, drop=True)
        )

    # Create static spatial features
    with detect_new_columns(gdf) as static_spatial:
        static_cols, neighbor_dict = create_static_spatial_features(gdf)

        gdf["area"] = gdf["occupation_idx"].map(static_cols["area"])
        gdf["n_neighbors"] = gdf["occupation_idx"].map(
            static_cols["n_neighbors"])
        # gdf["shared_length"] = gdf["occupation_idx"].map(
        #     static_cols["shared_length"])

    static.extend(static_spatial)

    # Create dynamic spatial features
    with detect_new_columns(gdf) as dynamic_spatial:
        # !!!
        # Pretty, but runs by assumption that they are indexed correctly
        # Meaning that they are sorted first by occupation_idx, and then time_s
        # !!!
        df = create_dynamic_spatial_features(gdf, neighbor_dict)
        gdf["lag1_sum_neighbors"] = df["lag1_sum_neighbors"].fillna(0)
        gdf["lag1_median_neighbors"] = df["lag1_median_neighbors"].fillna(0)

    with detect_new_columns(gdf) as interaction_cols:
        # gdf["lag_1_x_n_neighbors"] = gdf["lag_1"] * gdf["n_neighbors"]
        # gdf["lag1_diff_neighbors"] = gdf["lag_1"] - gdf["lag1_median_neighbors"]
        for column in [*static, *dynamic, *time_features, *temporal_features, *dynamic_spatial]:
            gdf[f"{column}_x_post_corona"] = gdf[column] * gdf["post_corona"]
            # gdf[f"{column}_x_during_corona"] = gdf[column] * gdf["during_corona"]
        

    dynamic.extend(interaction_cols)

    return (
        gdf.dropna().drop(columns=["geometry", "period"]).pipe(downcast_numeric),
        static, dynamic, seasonal_features, time_features,
        temporal_features, dynamic_spatial
        )
