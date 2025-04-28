import os
import zipfile

import requests
from tqdm import tqdm


def extract_filtered_files_from_zips(zip_path, output_dir, include_patterns):
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


def download_files(urls: list, folder: str):
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


if __name__ == "__main__":
    # Example usage
    urls = [
        "https://data.police.uk/data/archive/2025-02.zip",
        "https://data.police.uk/data/archive/2022-02.zip",
        "https://data.police.uk/data/archive/2019-02.zip",
        "https://data.police.uk/data/archive/2016-02.zip",
    ]
    download_files(urls, os.path.join("data", "raw_download"))
    print("Download completed.")
