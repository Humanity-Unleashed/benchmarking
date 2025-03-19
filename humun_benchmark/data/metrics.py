import ast
import os
from io import StringIO
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def read_results(paths: Union[str, List[str]]) -> Dict[str, Dict]:
    if isinstance(paths, str):
        paths = [paths]

    benchmarks = {os.path.basename(file): pd.read_parquet(file).iloc[0] for file in paths}

    for model in benchmarks.keys():
        benchmarks[model] = benchmarks[model].to_dict()

        for key in benchmarks[model].keys():
            benchmarks[model][key] = ast.literal_eval(benchmarks[model][key])

        for series_id, json_str in benchmarks[model]["results"].items():
            df = pd.read_json(StringIO(json_str))
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            benchmarks[model]["results"][series_id] = df

    return benchmarks


def compute_all_metrics(
    results_dict: Dict[str, Dict],
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Given a dictionary with model keys (from read_results), where for each model,
    value['results'] is a dictionary of DataFrames keyed by series_id, compute:

    - Per-dataset metrics: a DataFrame with one row per series_id.
    - Overall (cross-dataset) metrics: a one-row DataFrame with aggregated metrics.

    Also, adds a root-level key "benchmark" which is a DataFrame combining each model's
    overall metrics (one row per model).

    Returns a dictionary of the form:
    {
      'model1': {
          'metrics': pd.DataFrame,       # Overall metrics (one row)
          'per_dataset': pd.DataFrame    # Per-dataset metrics (one row per series_id)
      },
      'model2': { ... },
      ...,
      'benchmark': pd.DataFrame         # Combined overall metrics for each model
    }
    """
    all_model_metrics = {}
    overall_list = []

    for model, data in results_dict.items():
        results = data["results"]

        per_dataset_metrics = {}
        for series_id, df in results.items():
            per_dataset_metrics[series_id] = compute_dataset_metrics(df)
        per_dataset_df = pd.DataFrame.from_dict(per_dataset_metrics, orient="index")

        # Compute overall (cross-dataset) metrics using all the DataFrames
        list_of_dfs = list(results.values())
        overall_metrics = compute_forecast_metrics(list_of_dfs)
        overall_metrics_df = pd.DataFrame([overall_metrics])

        overall_metrics_df.insert(0, "model", model.split("_")[0])  # removes date
        overall_list.append(overall_metrics_df)

        all_model_metrics[model] = {
            "metrics": overall_metrics_df,
            "per_dataset": per_dataset_df,
        }

    benchmark_df = pd.concat(overall_list, ignore_index=True)
    all_model_metrics["benchmark"] = benchmark_df

    return all_model_metrics


def crps_closed_form(obs, forecasts):
    """
    Computes CRPS using the closed-form expression for an empirical forecast distribution.
    """
    forecasts = np.array(forecasts)
    # Mean absolute difference between forecasts and observation.
    term1 = np.mean(np.abs(forecasts - obs))
    # 0.5 * mean pairwise absolute difference between forecasts.
    term2 = 0.5 * np.mean(np.abs(forecasts[:, None] - forecasts))
    return term1 - term2


def crps(
    target: np.array,
    samples: np.array,
) -> np.array:
    """
    Compute the CRPS using the probability weighted moment form.
    See Eq ePWM from "Estimation of the Continuous Ranked Probability Score with
    Limited Information and Applications to Ensemble Weather Forecasts"
    https://link.springer.com/article/10.1007/s11004-017-9709-7

    Parameters:
    -----------
    target: np.ndarray
        The target value(s) (for scalar target, shape should be ()).
    samples: np.ndarray
        The forecast values. For a single time point, this should have shape (n_samples,).

    Returns:
    --------
    crps: np.ndarray
        The CRPS for the given target and forecast samples.
    """
    # Ensure target shape matches the "variable" dimensions of samples.
    # For a scalar target and 1D samples, target.shape is () which equals samples.shape[1:]
    assert target.shape == samples.shape[1:], (
        f"shapes mismatch between: {target.shape} and {samples.shape}"
    )

    num_samples = samples.shape[0]
    num_dims = samples.ndim
    # Sort the forecast samples.
    sorted_samples = np.sort(samples, axis=0)
    # Compute the first term: average absolute difference between sorted samples and target.
    abs_diff = np.abs(np.expand_dims(target, axis=0) - sorted_samples).sum(axis=0) / num_samples
    # Compute beta0: the average of the sorted samples.
    beta0 = sorted_samples.sum(axis=0) / num_samples
    # Create an array [0, 1, ..., num_samples-1] and expand dims to match forecast dimensions.
    i_array = np.expand_dims(np.arange(num_samples), axis=tuple(range(1, num_dims)))
    beta1 = (i_array * sorted_samples).sum(axis=0) / (num_samples * (num_samples - 1))
    return abs_diff + beta0 - 2 * beta1


def compute_dataset_metrics(df: pd.DataFrame) -> Dict:
    """
    Computes per-dataset error metrics that can be meaningfully averaged.
    """
    forecast_cols = [col for col in df.columns if col.startswith("forecast_")]
    forecast_matrix = df[forecast_cols].values  # shape: (n_time_points, n_samples)
    actuals = df["value"].values  # shape: (n_time_points,)
    forecast_errors = forecast_matrix - actuals[:, None]

    # Compute metrics: MAE, RMSE, and MAPE (handling zero actuals).
    mae = np.mean(np.abs(forecast_errors))
    rmse = np.sqrt(np.mean(forecast_errors**2))

    non_zero_mask = actuals != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs(forecast_errors[non_zero_mask] / actuals[non_zero_mask, None])) * 100
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "n_samples": len(actuals),
        "forecasts": forecast_matrix.tolist(),
        "actuals": actuals.tolist(),
    }


def compute_cross_dataset_metrics(forecasts: List[np.ndarray], actuals: List[np.ndarray]) -> Dict:
    """
    Computes metrics that need all datasets together.

    Parameters:
        forecasts: List of forecast arrays (each of shape: (n_time_points, n_samples))
        actuals: List of actual arrays (each of shape: (n_time_points,))
    """
    # Combine all forecasts and actuals across datasets.
    all_forecasts = np.vstack(forecasts)  # shape: (total_time_points, n_samples)
    all_actuals = np.concatenate(actuals)  # shape: (total_time_points,)
    all_errors = all_forecasts - all_actuals[:, None]

    # Compute rank-based metrics (ranking errors for each time point).
    ranks = np.apply_along_axis(rankdata, 1, np.abs(all_errors))
    avg_rank = np.mean(ranks)

    # Compute CRPS for each time point using the new CRPS function.
    # Here, each row in all_forecasts corresponds to the n_samples for one time point.
    crps_values = [crps(np.array(obs), fc) for obs, fc in zip(all_actuals, all_forecasts)]
    avg_crps = np.mean(crps_values)

    # Compute error distribution percentiles.
    error_percentiles = np.percentile(np.abs(all_errors), [5, 50, 95])

    return {
        "CRPS": float(avg_crps),
        "Average Rank": float(avg_rank),
        "Error P50": float(error_percentiles[1]),
        "Error P95": float(error_percentiles[2]),
    }


def compute_forecast_metrics(dfs: Union[pd.DataFrame, List[pd.DataFrame]]) -> Dict:
    """
    Computes all metrics, handling both per-dataset and cross-dataset metrics properly.
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    # Compute per-dataset metrics
    dataset_metrics = [compute_dataset_metrics(df) for df in dfs]

    # Weight averageable metrics by dataset size
    total_samples = sum(m["n_samples"] for m in dataset_metrics)
    weighted_metrics = {
        "MAE": float(sum(m["MAE"] * m["n_samples"] for m in dataset_metrics) / total_samples),
        "RMSE": float(
            np.sqrt(sum((m["RMSE"] ** 2 * m["n_samples"]) for m in dataset_metrics) / total_samples)
        ),
        "MAPE": float(sum(m["MAPE"] * m["n_samples"] for m in dataset_metrics) / total_samples),
    }

    # Compute cross-dataset metrics (convert back to np for metrics)
    forecasts = [np.array(m["forecasts"]) for m in dataset_metrics]
    actuals = [np.array(m["actuals"]) for m in dataset_metrics]
    cross_metrics = compute_cross_dataset_metrics(forecasts, actuals)

    # Combine all metrics
    return {**weighted_metrics, **cross_metrics}
