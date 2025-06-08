from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from pyro.infer import Predictive


class PredictionTester:
    def __init__(self, data, model, guide, occupation_mapping: dict):
        """
        data: dictionary containing tensors, including "occupation_idx" (tensor of ints)
        occupation_mapping: dict mapping numeric occupation_idx → string label
        """
        # 1) Store occupation_idx as Tensor and NumPy
        self.occupation_idx = data["occupation_idx"]  # Tensor shape (N,)
        self.ward_idx = data["ward_idx"]
        self.occ_idx_np = self.occupation_idx.cpu().numpy()  # NumPy array shape (N,)
        # Build string labels array in the same order
        self.occ_labels = np.array(
            [occupation_mapping[int(i)] for i in self.occ_idx_np])

        # 2) Store feature tensors
        self.X_static = data["X_static"]
        self.X_dynamic = data["X_dynamic"]
        self.X_seasonal = data["X_seasonal"]
        self.X_time_trend = data["X_time_trend"]
        self.X_temporal = data["X_temporal"]
        self.X_spatial = data["X_spatial"]

        self.model = model
        self.guide = guide
        self.predictions = None  # Will hold Predictive output

    def predict(self, num_samples: int):
        """
        Draw `num_samples` posterior predictive samples for "obs".
        Stores a dict with key "obs" of shape (num_samples, N).
        """
        predictive = Predictive(
            model=self.model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=["obs"]
        )
        self.predictions = predictive(
            self.occupation_idx,
            self.ward_idx,
            self.X_static,
            self.X_dynamic,
            self.X_seasonal,
            self.X_time_trend,
            self.X_temporal,
            self.X_spatial
        )

    def get_all_predictions(self, save_path: str = None) -> pd.DataFrame:
        """
        Return a pandas DataFrame of posterior predictive draws:
        - rows are indexed by the mapped occupation label
        - columns correspond to each sample draw (0, 1, ..., num_samples-1)
        All entries are converted to integers and downcast to the smallest integer dtype.
        If save_path is provided, the DataFrame is written to that CSV file.
        """
        if self.predictions is None:
            raise RuntimeError(
                "No predictions found. Call predict(num_samples) first.")
        obs_tensor = self.predictions["obs"]  # shape: (num_samples, N)
        num_samples, N = obs_tensor.shape

        # Move to numpy and transpose so shape becomes (N, num_samples)
        values = obs_tensor.cpu().numpy().T  # shape: (N, num_samples)

        # Build DataFrame
        index = self.occ_labels  # length N
        columns = list(range(num_samples))
        df = pd.DataFrame(values, index=index, columns=columns)

        # Convert all entries to integer then downcast to smallest integer type
        df = df.astype(int)
        int_cols = df.select_dtypes(include=["int64", "int32"]).columns
        if len(int_cols):
            df[int_cols] = df[int_cols].apply(
                pd.to_numeric, downcast="integer")

        if save_path is not None:
            self._save(df, save_path)

        return df

    def get_mean_predictions(self, save_path: str = None) -> pd.DataFrame:
        """
        Compute the mean over the num_samples draws for each occupation_idx.
        Returns a DataFrame with one column "mean", indexed by the occupation label.
        If save_path is provided, the DataFrame is written to that CSV file.
        """
        if self.predictions is None:
            raise RuntimeError(
                "No predictions found. Call predict(num_samples) first.")
        # Mean across samples → shape (N,)
        mean_vals = self.predictions["obs"].float().mean(dim=0).cpu().numpy()
        df = pd.DataFrame({"mean": mean_vals}, index=self.occ_labels)

        if save_path is not None:
            self._save(df, save_path)

        return df

    def get_median_predictions(self, save_path: str = None) -> pd.DataFrame:
        """
        Compute the median over the num_samples draws for each occupation_idx.
        Returns a DataFrame with one column "median", indexed by the occupation label.
        If save_path is provided, the DataFrame is written to that CSV file.
        """
        if self.predictions is None:
            raise RuntimeError(
                "No predictions found. Call predict(num_samples) first.")
        median_vals = self.predictions["obs"].float().median(
            dim=0).values.cpu().numpy()
        df = pd.DataFrame({"median": median_vals}, index=self.occ_labels)

        if save_path is not None:
            self._save(df, save_path)

        return df

    def get_confidence_intervals(self, alpha: float = 0.05, save_path: str = None) -> pd.DataFrame:
        """
        Compute lower and upper (1 - alpha) credible bounds for each occupation_idx.
        Returns a DataFrame with columns ["lower_bound","upper_bound"], indexed by occupation label.
        If save_path is provided, the DataFrame is written to that CSV file.
        """
        if self.predictions is None:
            raise RuntimeError(
                "No predictions found. Call predict(num_samples) first.")
        # shape: (num_samples, N)
        obs_np = self.predictions["obs"].cpu().numpy()
        lower_bound = np.quantile(obs_np, alpha / 2, axis=0)
        upper_bound = np.quantile(obs_np, 1 - alpha / 2, axis=0)
        df = pd.DataFrame(
            {"lower_bound": lower_bound, "upper_bound": upper_bound},
            index=self.occ_labels
        )

        if save_path is not None:
            self._save(df, save_path)

        return df

    @staticmethod
    def _save(obj, path: str):
        """
        Save a pandas Series or DataFrame to CSV at `path`.
        """
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(path, compression='gzip')
        elif isinstance(obj, pd.Series):
            obj.to_parquet(path, header=True, compression='gzip')
        else:
            raise ValueError("Can only save pandas DataFrame or Series.")

    def __repr__(self):
       return (
            f"Tester(occupation_idx={self.occupation_idx.shape}, ward_idx={self.ward_idx.shape}, "
            f"X_static={self.X_static.shape}, X_dynamic={self.X_dynamic.shape}, "
            f"X_seasonal={self.X_seasonal.shape}, X_time_trend={self.X_time_trend.shape}, "
            f"X_temporal={self.X_temporal.shape}, X_spatial={self.X_spatial.shape})"
        )



import torch
import pandas as pd
from pyro.infer import Predictive
from typing import Dict, List

class StatisticalTester:
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        model,
        guide,
        factors_map: Dict[str, List[str]],
    ):
        """
        data: dict containing tensors, including:
            - "occupation_idx": Tensor of shape (N,)
            - "X_static", "X_dynamic", "X_seasonal", "X_time_trend", "X_temporal", "X_spatial"
        model:     the Pyro model function
        guide:     the trained AutoGuide (or other guide) corresponding to model
        factors_map: dict mapping factor‐names → list of column labels for that factor
        """
        self.occupation_idx = data["occupation_idx"]  # Tensor shape (N,)
        self.ward_idx      = data["ward_idx"]
        self.X_static      = data["X_static"]
        self.X_dynamic     = data["X_dynamic"]
        self.X_seasonal    = data["X_seasonal"]
        self.X_time_trend  = data["X_time_trend"]
        self.X_temporal    = data["X_temporal"]
        self.X_spatial     = data["X_spatial"]

        self.model = model
        self.guide = guide
        self.factors_map = factors_map

        # Will be populated by predict()
        self.posterior_samples = None


    def predict(self, num_samples: int):
        """
        Draw `num_samples` posterior samples for each factor listed in factors_map.
        Stores self.posterior_samples as a dict mapping factor names to Tensors
        of shape (num_samples, 1, n_cols) or (num_samples, n_cols).
        """
        predictive = Predictive(
            model=self.model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=list(self.factors_map.keys())
        )
        self.posterior_samples = predictive(
            self.occupation_idx,
            self.ward_idx,
            self.X_static,
            self.X_dynamic,
            self.X_seasonal,
            self.X_time_trend,
            self.X_temporal,
            self.X_spatial
        )


    def _summarize_factor(self, factor: str) -> pd.DataFrame:
        """
        Compute posterior mean, 95% CI, two‐sided p‐value, and 'significant_CI' flag
        for each coefficient in `factor`. Returns a DataFrame with columns:
          ["col", "mean", "ci_lower", "ci_upper", "p_val", "significant_CI"]
        sorted by p_val ascending.
        """
        if self.posterior_samples is None:
            raise RuntimeError("No posterior_samples found. Call predict(num_samples) first.")

        samples = self.posterior_samples[factor]
        # If shape is (n_samples, 1, n_cols), squeeze to (n_samples, n_cols)
        if samples.ndim == 3 and samples.size(1) == 1:
            samples = samples.squeeze(1)
        
        #If shape is empty or has no columns, return empty DataFrame
        if samples.numel() == 0 or samples.size(-1) == 0:
            return pd.DataFrame(
                columns=["col", "mean", "ci_lower", "ci_upper", "p_val", "significant_CI"]
            )

        # 1) Posterior mean, 95% CI
        mean_vals  = samples.mean(dim=0)   # shape: (n_cols,)
        lower_vals = torch.quantile(samples, 0.025, dim=0)
        upper_vals = torch.quantile(samples, 0.975, dim=0)

        # 2) Two‐sided “p‐value” = 2 * min(P(β>0), P(β<0))
        prop_pos = (samples > 0.0).float().mean(dim=0)
        prop_neg = (samples < 0.0).float().mean(dim=0)
        p_vals = 2.0 * torch.minimum(prop_pos, prop_neg)

        # 3) Build a DataFrame row for each coefficient
        rows = []
        col_labels = self.factors_map[factor]
        for j, col_name in enumerate(col_labels):
            m_j   = mean_vals[j].item()
            lo_j  = lower_vals[j].item()
            hi_j  = upper_vals[j].item()
            p_j   = p_vals[j].item()
            sig_j = (lo_j > 0.0) or (hi_j < 0.0)
            rows.append({
                "col":            col_name,
                "mean":           m_j,
                "ci_lower":       lo_j,
                "ci_upper":       hi_j,
                "p_val":          p_j,
                "significant_CI": "Yes" if sig_j else "No"
            })

        df = pd.DataFrame(rows)
        return df.sort_values("p_val").reset_index(drop=True)


    def evaluate_all(self, save_dir: str = None) -> Dict[str, pd.DataFrame]:
        """
        For each factor in factors_map, run _summarize_factor and collect results in a dict.
        If save_dir is provided, save each factor's DataFrame to "{save_dir}/{factor}_summary.csv".
        Returns: dict mapping factor_name → summary DataFrame.
        """
        if self.posterior_samples is None:
            raise RuntimeError("No posterior_samples found. Call predict(num_samples) first.")

        results = {}
        for factor in self.factors_map.keys():
            df_factor = self._summarize_factor(factor)
            results[factor] = df_factor
            if save_dir is not None:
                path = f"{save_dir}/{factor}_summary.csv"
                df_factor.to_csv(path, index=False)
        return results


    def __repr__(self):
        return (
            f"StatisticalTester(occupation_idx={self.occupation_idx.shape}, ward_idx={self.ward_idx.shape}, "
            f"X_static={self.X_static.shape}, X_dynamic={self.X_dynamic.shape}, "
            f"X_seasonal={self.X_seasonal.shape}, X_time_trend={self.X_time_trend.shape}, "
            f"X_temporal={self.X_temporal.shape}, X_spatial={self.X_spatial.shape})"
        )
