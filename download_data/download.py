import os
import requests
from tqdm import tqdm

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



if __name__ == "__main__":
    # Example usage
    urls = [
        "https://data.police.uk/data/archive/2025-02.zip",
        "https://data.police.uk/data/archive/2022-02.zip",
        "https://data.police.uk/data/archive/2019-02.zip",
        "https://data.police.uk/data/archive/2016-02.zip",
        "https://data.police.uk/data/archive/2013-02.zip",
    ]
    download_files(urls, "downloaded_files")
    print("Download completed.")