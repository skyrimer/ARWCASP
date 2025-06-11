import random
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyro
import torch


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns in a DataFrame to the smallest possible numeric type.
    This function processes both integer and float columns, raising an error if none are found.

    Args:
        df: The input pandas DataFrame to be downcasted.

    Returns:
        pd.DataFrame: The DataFrame with downcasted numeric columns.

    Raises:
        ValueError: If no integer or float columns are found in the DataFrame.
    """
    for number_type in ['integer', 'float']:

        if not any(df.select_dtypes(include=[number_type]).columns):
            raise ValueError(f"No {number_type} columns found in DataFrame.")

        number_cols = df.select_dtypes(include=[number_type]).columns
        df[number_cols] = (df[number_cols].apply(
            pd.to_numeric, downcast=number_type))

    return df


@contextmanager
def detect_new_columns(df):
    """
    Context manager that yields a list which will be populated
    with any column names added to `df` inside the `with` block,
    using pandas’ Index.difference rather than Python sets.
    """
    before_idx = df.columns.copy()  # this is a pandas Index
    new_cols = []

    try:
        yield new_cols
    finally:
        new_cols.extend(
            df.columns.difference(before_idx).tolist()
        )


def single_out_last(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and test sets.
    The last month is used for testing.
    """

    return df[df['time_s'] < df['time_s'].max()].copy(), \
        df[df['time_s'] == df['time_s'].max()].copy()


def setup_reproducibility(seed: int = 0) -> torch.device:
    """
    Configure all random seeds and deterministic settings for PyTorch and Pyro,
    then return the chosen device (CPU or CUDA).
    """
    # 1) Seed Python, NumPy, and Torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2) Force deterministic CUDA kernels (if using GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3) Clear Pyro’s parameter store and seed Pyro
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)

    # 4) Choose device and set it as default
    device = get_device()
    torch.set_default_device(device)
    return device


def get_device() -> torch.device:
    """
    Get the current default device for PyTorch.
    This is useful to ensure consistency across different parts of the code.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def last_n_time_splits(df: pd.DataFrame,
                       time_col: str = "time_s",
                       n_splits: int = 12):
    results = []
    results.extend(
        (
            df.query(f"{time_col} < {time_vallue}").copy(),
            df.query(f"{time_col} == {time_vallue}").copy(),
        )
        for time_vallue in df.sort_values(time_col)[time_col]
        .unique()
        .tolist()[-n_splits:]
    )
    return results
