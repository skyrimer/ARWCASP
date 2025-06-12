from typing import Literal, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np
from feature_engineering.all_features import create_all_features
from utils.utils import downcast_numeric



def get_raw_data(path: str, occupation_type: Literal["lsoa", "msoa", "borough"]) -> gpd.GeoDataFrame:
    if occupation_type not in ["lsoa", "msoa", "borough"]:
        raise ValueError(
            "occupation_type must be one of 'lsoa', 'msoa', or 'borough'")
    if occupation_type == "lsoa":
        return gpd.read_parquet(path).rename(columns={
            'LSOA code (2021)': 'occupation', 'date': 'period',
            'Burglaries amount': 'burglaries'
        })
    elif occupation_type == "msoa":
        pass
    elif occupation_type == "borough":
        pass


def convert_occupation_to_idx(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, dict, dict]:
    """
    Convert occupation names to indices in the GeoDataFrame.
    """
    if "occupation" not in gdf.columns:
        raise ValueError("GeoDataFrame must contain 'occupation' column")

    _, uniques = pd.factorize(gdf["occupation"], sort=False)
    name_to_code = {name: code for code, name in enumerate(uniques)}
    code_to_name = dict(enumerate(uniques))

    return gdf.assign(
        occupation_idx=gdf["occupation"].map(name_to_code)
    ).drop(columns=["occupation"]).pipe(downcast_numeric), name_to_code, code_to_name


def get_static_and_dynamic_cols(gdf: gpd.GeoDataFrame) -> Tuple[list, list]:
    """
    Get static and dynamic columns from the GeoDataFrame.
    """
    other_cols = gdf.columns.difference(
        ["occupation_idx", "period", "burglaries", "geometry"])

    nuniques_per_occ = gdf.groupby("occupation_idx")[other_cols].nunique()
    max_nunique = nuniques_per_occ.max()

    return (max_nunique[max_nunique == 1].index.tolist(),
            max_nunique[max_nunique > 1].index.tolist())

def add_ward_idx(
    gdf: gpd.GeoDataFrame,
    lookup_path: str = "../processed_data/LSOA_to_Ward_LAD_lookup.csv",
) -> tuple[gpd.GeoDataFrame, dict, dict]:
    """Attach ward indices based on an LSOA→Ward lookup."""
    lookup = pd.read_csv(lookup_path)
    lookup = lookup.rename(columns={"LSOA21CD": "occupation", "WD24NM": "ward"})
    merged = gdf.merge(lookup[["occupation", "ward"]], on="occupation", how="left")
    if merged["ward"].isna().any():
        raise ValueError("Missing ward mapping for some LSOAs")
    _, uniques = pd.factorize(merged["ward"], sort=False)
    name_to_code = {name: code for code, name in enumerate(uniques)}
    code_to_name = dict(enumerate(uniques))
    merged = merged.assign(ward_idx=merged["ward"].map(name_to_code))
    merged = merged.drop(columns=["ward"])
    return merged, name_to_code, code_to_name


def prepare_all_data(path: str, occupation_type: Literal["lsoa", "msoa", "borough"]) -> pd.DataFrame:
    """
    Prepare all data by reading, converting occupation to indices,
    and extracting static and dynamic columns.
    """
    gdf = get_raw_data(path, occupation_type).drop(columns="Crime Rank (where 1 is most deprived)")

    gdf, ward_to_code, code_to_ward = add_ward_idx(gdf)
    gdf, name_to_code, code_to_name = convert_occupation_to_idx(gdf)
    static, dynamic = get_static_and_dynamic_cols(gdf)
    if "ward_idx" in static:
        static.remove("ward_idx")

    gdf, *feature_lists = create_all_features(gdf, static, dynamic)

    ward_idx_map = (
        gdf[["occupation_idx", "ward_idx"]]
        .drop_duplicates("occupation_idx")
        .sort_values("occupation_idx")
        .set_index("occupation_idx")["ward_idx"]
        .astype(np.int16)
        .values
    )

    assert ward_idx_map.shape[0] == gdf["occupation_idx"].nunique(), (
        "Ward index map length mismatch after feature creation"
    )

    return (
        (gdf, *feature_lists),
        (name_to_code, code_to_name),
        ward_idx_map,
    )

