from typing import Tuple

import geopandas as gpd
import pandas as pd


def create_static_spatial_features(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, dict[int, list[int]]]:
    # 1a. Keep only one geometry per occupation_idx (drop duplicate footprints).
    unique_geom = (
        gdf[["occupation_idx", "geometry"]]
        .drop_duplicates(subset=["occupation_idx"])
        .reset_index(drop=True)
    )

    # 1b. Apply buffer(0) to clean any invalid polygons (slivers, self-intersections, etc.).
    unique_geom["geometry"] = unique_geom["geometry"].buffer(0)

    # 1c. Turn it into a proper GeoDataFrame and copy the CRS from gdf_time.
    unique_geom = gpd.GeoDataFrame(
        unique_geom[["occupation_idx", "geometry"]],
        geometry="geometry",
        crs=gdf.crs
    )

    # 1e. Compute each footprint’s area (in the projected CRS units, e.g. m²).
    unique_geom["area"] = unique_geom.geometry.area

    # 1f. Make occupation_idx the index, so that building a spatial index yields
    #     neighbors keyed by occupation_idx directly.
    unique_geom = unique_geom.set_index("occupation_idx")

    # 2a. Build the spatial index (R-tree) on the buffered geometries:
    sindex = unique_geom.sindex

    # 2b. For each occupation_idx, find all occupation_idx neighbors that touch it:
    # will map: occupation_idx → [neighbor_occ1, neighbor_occ2, ...]
    neighbor_dict = {}

    for occ_i, geom_i in unique_geom.geometry.items():
        # 2b.1. Bounding‐box candidates (very fast filter):
        possible_matches = list(sindex.intersection(geom_i.bounds))
        # 2b.2. Keep only those that truly touch (and drop itself):
        actual_neighbors = [
            occ_j
            for occ_j in possible_matches
            if (occ_j != occ_i and geom_i.touches(unique_geom.geometry.loc[occ_j]))
        ]
        neighbor_dict[occ_i] = actual_neighbors

    # Now neighbor_dict[101] might look like [102, 103], etc.

    unique_geom["n_neighbors"] = unique_geom.index.map(
        lambda i: len(neighbor_dict[i]))
    shared_lengths = []

    for occ_i, geom_i in unique_geom.geometry.items():
        total_len = 0.0
        for occ_j in neighbor_dict[occ_i]:
            shared_line = geom_i.intersection(unique_geom.geometry.loc[occ_j])
            total_len += shared_line.length
        shared_lengths.append(total_len)

    unique_geom["shared_length"] = shared_lengths
    # Prepare a small DataFrame of static attributes:
    static_cols = unique_geom[["area", "n_neighbors", "shared_length"]]

    # Merge onto gdf_time by occupation_idx:
    return static_cols.reset_index().set_index("occupation_idx"), neighbor_dict


def create_neighbour_map(neighbor_dict: dict[int, list[int]]) -> pd.DataFrame:
    pairs = []
    for occ_i, neighs in neighbor_dict.items():
        pairs.extend((occ_i, occ_j) for occ_j in neighs)
    return pd.DataFrame(pairs, columns=["occupation_idx", "neighbor_idx"])


def create_dynamic_spatial_features(gdf: gpd.GeoDataFrame,
                                    neighbor_dict: dict[int, list[int]]) -> pd.DataFrame:
    df_neighbors = (
        gdf[["occupation_idx", "time_s", "lag_1"]]
        .rename(columns={
            "occupation_idx": "neighbor_idx",
            "lag_1":          "lag1_neighbor"
        })
    )

    df_base = gdf[["occupation_idx", "time_s"]].drop_duplicates()

    df_occ_neigh = pd.merge(
        df_base,
        create_neighbour_map(neighbor_dict),
        on="occupation_idx",
        how="left"
    )

    df_occ_neigh = pd.merge(
        df_occ_neigh,
        df_neighbors,               # has neighbor_idx | time_s | lag1_neighbor
        on=["neighbor_idx", "time_s"],
        how="left"
    )

    df_occ_neigh = df_occ_neigh.dropna(
        subset=["neighbor_idx", "lag1_neighbor"])

    return df_occ_neigh.groupby(["occupation_idx", "time_s"])["lag1_neighbor"].agg(
        lag1_sum_neighbors="sum",
        lag1_mean_neighbors="mean",
        lag1_median_neighbors="median"
    ).reset_index()
