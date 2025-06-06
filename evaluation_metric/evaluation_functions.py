from contextlib import contextmanager

import numpy as np
import pandas as pd



#Function to merge Predicted and Observed data, based on LSOA
#This merged dataframe is necessary for all the other functions related to RMSE

def Full_Data_In_Merged_RMSE(predicted_data, observed_data):    
    predicted_data = predicted_data.copy()
    observed_data = observed_data.copy()
    predicted_data['LSOA'] = predicted_data['LSOA'].astype(str)
    observed_data['LSOA'] = observed_data['LSOA'].astype(str)

    # Merge and fill missing observed counts with 0
    merged = pd.merge(
        predicted_data,
        observed_data,
        on='LSOA',
        how='left',
        suffixes=('_pred', '_obs')
    )
    merged['count_obs'] = merged['count_obs'].fillna(0)

    return merged

#This function is used to convert the 'raw' output data (LSOA's, with 5000 simulation runs on count)
#into a function. needed to calculate CRPS

def fit_gaussian_per_lsoa(original_data: pd.DataFrame, id_col: str = 'LSOA', epsilon: float = 1e-6):
    """
    Fits a normal distribution to the predicted samples for each LSOA.

    Parameters:
    - original_data: DataFrame with one identifier column (e.g. 'LSOA') and the rest are samples
    - id_col: name of the identifier column (e.g., 'LSOA')
    - epsilon: minimum allowed std deviation to avoid degenerate distribution

    Returns:
    - lsoa_pdfs: dict mapping LSOA to scipy.stats.norm distribution
    - stats_df: DataFrame with columns ['LSOA', 'mu', 'sigma']
    """
    from scipy.stats import norm
    import numpy as np
    import pandas as pd

    lsoa_pdfs = {}
    means = []
    stds = []
    index_list = []

    for _, row in original_data.iterrows():
        lsoa_code = row[id_col]
        samples = row.drop(id_col).values.astype(float)

        mu = np.mean(samples)
        sigma = np.std(samples)

        if sigma < epsilon:
            sigma = epsilon

        lsoa_pdfs[lsoa_code] = norm(loc=mu, scale=sigma)
        means.append(mu)
        stds.append(sigma)
        index_list.append(lsoa_code)

    stats_df = pd.DataFrame({
        id_col: index_list,
        'mu': means,
        'sigma': stds
    })

    return stats_df

#Function to merge the stats_df from fit_gaussian_per_lsoa() with the observed data, needed to run the future CRPS function
def Full_Data_In_Merged_CRPS(stats_df, observed_data):    
    merged_crps = pd.merge(stats_df, observed_data, on='LSOA', how='left')
    merged_crps = merged_crps.fillna(0)
    return merged_crps

def rmse_score(merged: pd.DataFrame) -> float:
    """
    Computes average RMSE across LSOAs using the merged DataFrame with 'count_pred' and 'count_obs'.

    Parameters:
    - merged: DataFrame with ['LSOA', 'count_pred', 'count_obs']

    Returns:
    - float: average RMSE across all LSOAs
    """
    def compute_rmse(group):
        return np.sqrt(np.mean((group['count_pred'] - group['count_obs']) ** 2))

    rmse_by_lsoa = merged.groupby('LSOA').apply(compute_rmse)
    return rmse_by_lsoa.mean()



def calculate_crps(mu, sigma, x_obs):
    """
    Computes CRPS for a normal distribution with mean=mu, std=sigma and observation x_obs.
    """
    z = (x_obs - mu) / sigma
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))

def compute_average_crps(stats_df: pd.DataFrame) -> float:
    """
    Computes the average CRPS using mu/sigma from stats_df and the corresponding count_obs.

    Parameters:
    - df: DataFrame with ['LSOA', 'mu', 'sigma', 'count_obs']

    Returns:
    - float: average CRPS
    """
    crps_scores = []

    for _, row in stats_df.iterrows():
        mu = row['mu']
        sigma = row['sigma']
        x_obs = row['count']
        crps = calculate_crps(mu, sigma, x_obs)
        crps_scores.append(crps)

    return np.mean(crps_scores)
