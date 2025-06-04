from typing import Literal, Tuple

import geopandas as gpd
import pandas as pd
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


def prepare_all_data(path: str, occupation_type: Literal["lsoa", "msoa", "borough"]) -> pd.DataFrame:
    """
    Prepare all data by reading, converting occupation to indices,
    and extracting static and dynamic columns.
    """
    gdf = get_raw_data(path, occupation_type)

    gdf, name_to_code, code_to_name = convert_occupation_to_idx(gdf)
    static, dynamic = get_static_and_dynamic_cols(gdf)

    return (create_all_features(gdf, static, dynamic), (name_to_code, code_to_name))
