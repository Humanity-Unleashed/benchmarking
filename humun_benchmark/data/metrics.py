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

        benchmarks[model]["results"] = ast.literal_eval(benchmarks[model]["results"])

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

    # mean of the absolute differences
    term1 = np.mean(np.abs(forecasts - obs))

    # average absolute difference between forecasts which is equivalent to
    # the double sum divided by n^2, multiplied by 0.5
    term2 = 0.5 * np.mean(np.abs(forecasts[:, None] - forecasts))
    return term1 - term2


def compute_dataset_metrics(df: pd.DataFrame) -> Dict:
    """
    Computes per-dataset error metrics that can be meaningfully averaged.
    """
    forecast_cols = [col for col in df.columns if col.startswith("forecast_")]
    forecast_matrix = df[forecast_cols].values
    actuals = df["value"].values
    forecast_errors = forecast_matrix - actuals[:, None]

    # Only compute metrics that can be meaningfully averaged
    mae = np.mean(np.abs(forecast_errors))
    rmse = np.sqrt(np.mean(forecast_errors**2))

    # Handle zero values in MAPE calculation
    non_zero_mask = actuals != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs(forecast_errors[non_zero_mask] / actuals[non_zero_mask, None])) * 100
    else:
        mape = np.nan  # or some other fallback

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "n_samples": len(actuals),
        # convert to list for JSON serialisation
        "forecasts": forecast_matrix.tolist(),
        "actuals": actuals.tolist(),
    }


def compute_cross_dataset_metrics(forecasts: List[np.ndarray], actuals: List[np.ndarray]) -> Dict:
    """
    Computes metrics that need all datasets together.
    """
    # Combine all forecasts and actuals
    all_forecasts = np.vstack(forecasts)
    all_actuals = np.concatenate(actuals)
    all_errors = all_forecasts - all_actuals[:, None]

    # Rank metrics across all datasets
    ranks = np.apply_along_axis(rankdata, 1, np.abs(all_errors))
    avg_rank = np.mean(ranks)

    # CRPS across all datasets
    crps_values = [crps_closed_form(obs, fc) for obs, fc in zip(all_actuals, all_forecasts)]
    avg_crps = np.mean(crps_values)

    # Distribution metrics
    error_percentiles = np.percentile(np.abs(all_errors), [25, 50, 75, 90])

    return {
        "CRPS": float(avg_crps),
        "Average Rank": float(avg_rank),
        "Error P25": float(error_percentiles[0]),
        "Error P50": float(error_percentiles[1]),
        "Error P75": float(error_percentiles[2]),
        "Error P90": float(error_percentiles[3]),
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
