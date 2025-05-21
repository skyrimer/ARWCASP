import os
import zipfile

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm


def save_geodataframe_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").sort_values(by="geometry").to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="gzip"
    )


def read_city_of_london_street_csvs(folder_path, file_pattern):
    """
    Reads all CSV files in the given folder whose filenames contain 'city-of-london-street',
    and concatenates them into a single pandas DataFrame.

    Args:
        folder_path (str): Path to the folder containing the CSV files.

    Returns:
        pd.DataFrame: A single concatenated DataFrame containing all matching CSVs' data.
    """
    # List to collect individual DataFrames
    dataframes = []

    for root, _, files in tqdm(os.walk(folder_path), desc="Reading CSV files"):
        for filename in tqdm(files, desc="Processing files", leave=False):
            if file_pattern in filename and filename.endswith(".csv"):
                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path)
                dataframes.append(df)

    return (
        pd.concat(dataframes, ignore_index=True)
        if dataframes
        else pd.DataFrame()
    )


def extract_filtered_files_from_zips(zip_path: str, output_dir: str,
                                     include_patterns: tuple[str]):
    """
    Extracts only files whose names contain any of the specified patterns
    from zip_path into output_dir, preserving internal structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in tqdm(z.namelist(), desc="Searching files"):
            name = os.path.basename(member)

            if any(pat in name for pat in include_patterns):
                z.extract(member, output_dir)


def download_files(urls: list, folder: str) -> None:
    """
    Download a list of file URLs into the specified folder with progress bars.
    """
    os.makedirs(folder, exist_ok=True)
    for url in tqdm(urls, desc="Downloading files"):
        local_filename = os.path.join(folder, url.split('/')[-1])

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(local_filename, 'wb') as f:
                if total:
                    for chunk in tqdm(r.iter_content(chunk_size=8192),
                                      total=total // 8192,
                                      desc=f"Saving {os.path.basename(local_filename)}",
                                      leave=False):
                        f.write(chunk)
                else:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        extract_filtered_files_from_zips(local_filename,
                                         os.path.join("data", "relevant_data"),
                                         {"city-of-london", "metropolitan"})


def preprocess_files(input_dir: str, output_dir: str) -> None:
    df_streets = read_city_of_london_street_csvs(input_dir, "street")
    df_streets = (
        df_streets.query("`Crime type` == 'Burglary'")
        .assign(
            period=pd.to_datetime(
                df_streets["Month"], format="%Y-%m").dt.to_period("M"),
            Latitude=pd.to_numeric(
                df_streets["Latitude"], errors='coerce', downcast='float'),
            Longitude=pd.to_numeric(
                df_streets["Longitude"], errors='coerce', downcast='float'),
            geometry=lambda df: df.apply(
                lambda row: Point(row['Longitude'], row['Latitude']) if pd.notnull(
                    row['Longitude']) and pd.notnull(row['Latitude']) else Point(),
                axis=1
            )
        )
        .dropna(subset=["LSOA code"])
        .fillna({
            "Last outcome category": "Not stated",
            "Crime ID": "Not stated",
            "LSOA code": "Not stated"
        })
        .astype({
            "Reported by": "category",
            "Falls within": "category",
            "Location": "category",
            "LSOA code": "category",
            "Last outcome category": "category",
            "Crime ID": "string"
        })
        .drop(columns=["Month", "LSOA name", "Context", "Crime type",
                       "Latitude", "Longitude"], errors="ignore")
        .drop_duplicates()
    )
    save_geodataframe_to_parquet(
        df_streets, os.path.join(output_dir, "street.parquet"))

    outcomes = read_city_of_london_street_csvs(
        os.path.join("data", "relevant_data"), "outcomes")
    outcomes = (
        outcomes[outcomes["Crime ID"].isin(df_streets["Crime ID"].unique())]
        .assign(
            period=pd.to_datetime(
                outcomes["Month"], format="%Y-%m").dt.to_period("M"),
            Latitude=pd.to_numeric(
                outcomes["Latitude"], errors='coerce', downcast='float'),
            Longitude=pd.to_numeric(
                outcomes["Longitude"], errors='coerce', downcast='float'),
            geometry=lambda df: df.apply(
                lambda row: Point(row["Longitude"], row["Latitude"]) if pd.notnull(
                    row["Longitude"]) and pd.notnull(row["Latitude"]) else Point(),
                axis=1
            )
        )
        .fillna({
            "LSOA code": "Not stated"
        })
        .astype({
            "Crime ID": "string",
            "Reported by": "category",
            "Falls within": "category",
            "Location": "category",
            "LSOA code": "category",
            "Outcome type": "category"
        })
        .drop(columns=["Month", "LSOA name", "Latitude", "Longitude"], errors="ignore")
        .drop_duplicates()
    )
    save_geodataframe_to_parquet(
        outcomes, os.path.join(output_dir, "outcomes.parquet"))

    search = read_city_of_london_street_csvs(
        os.path.join("data", "relevant_data"), "stop-and-search")
    search = (
        search
        .assign(
            Date=pd.to_datetime(search["Date"], errors="coerce"),
            Latitude=pd.to_numeric(
                search["Latitude"], errors="coerce", downcast="float"),
            Longitude=pd.to_numeric(
                search["Longitude"], errors="coerce", downcast="float"),
            geometry=lambda df: df.apply(
                lambda row: Point(row["Longitude"], row["Latitude"]) if pd.notnull(
                    row["Longitude"]) and pd.notnull(row["Latitude"]) else Point(),
                axis=1
            ),
            Person_search=search["Type"].str.contains("Person", na=False),
            Vehicle_search=search["Type"].str.contains("Vehicle", na=False),
            Part_of_policing_operation=search["Policing operation"].astype(
                "bool", errors="ignore"),
        )
        .fillna({
            "Age range": "Not stated",
            "Object of search": "Not stated",
            "Officer-defined ethnicity": "Not stated",
            "Gender": "Not-stated",
            "Outcome": "Not-stated",
            "Legislation": "Not-stated",
            "Self-defined ethnicity": "Not-stated",
        })
        .astype({
            "Gender": "category",
            "Age range": "category",
            "Self-defined ethnicity": "category",
            "Officer-defined ethnicity": "category",
            "Legislation": "category",
            "Object of search": "category",
            "Outcome": "category",
            "Outcome linked to object of search": "bool",
            "Removal of more than just outer clothing": "bool",
        })
        .drop(columns=["Policing operation", "Type", "Latitude",
                       "Longitude", "Part of a policing operation"], errors="ignore")
        .drop_duplicates()
    )

    save_geodataframe_to_parquet(
        search, os.path.join(output_dir, "search.parquet"))


if __name__ == "__main__":
    # Example usage
    urls = [
        "https://data.police.uk/data/archive/2025-03.zip",
        "https://data.police.uk/data/archive/2022-03.zip",
        "https://data.police.uk/data/archive/2019-03.zip",
        "https://data.police.uk/data/archive/2016-03.zip",
    ]

    print("Download completed.")

    # download_files(urls, os.path.join("data", "raw_download"))
    preprocess_files(os.path.join("data", "relevant_data"),
                     os.path.join("processed_data"))
    
