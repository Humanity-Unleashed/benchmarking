import ast
import os
from io import StringIO
from typing import Dict, List, Union

import numpy as np
import pandas as pd


def read_results(paths: Union[str, List[str]]) -> Dict[str, Dict]:
    if isinstance(paths, str):
        paths = [paths]

    # strip '_<datetime>.parquet' from path
    benchmarks = {
        os.path.splitext(file)[0][:-16].split("/")[-1]: pd.read_parquet(file).iloc[0] for file in paths
    }

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


### METRICS REFACTOR ###
# * modularise each metric into its own closed function
# * clearly show how things are aggregated / calculated
# * compute_all_metrics should simply be a for loop over each model and return a concatenated dataframe


def crps_closed_form(actual, forecasts):
    """
    Computes CRPS using the closed-form expression for a single time-step's
    forecast distribution.
    """
    forecasts = np.array(forecasts)
    # Mean absolute difference between forecasts and observation.
    term1 = np.mean(np.abs(forecasts - actual))
    # 0.5 * mean pairwise absolute difference between forecasts.
    term2 = 0.5 * np.mean(np.abs(forecasts[:, None] - forecasts))
    return term1 - term2


def normalized_crps(actual, forecasts, overall_range):
    """
    Computes a normalized CRPS by scaling the raw CRPS using a task-dependent factor α,
    where α is 1 divided by the overall range of the actual values.
    This normalization makes the CRPS scale-independent.

    Parameters:
      actual: scalar actual value for a time step.
      forecasts: array-like forecast samples for that time step.
      overall_range: the range (max - min) of the entire actuals dataset.

    Returns:
      Normalized CRPS.
    """
    raw_crps = crps_closed_form(actual, forecasts)
    if overall_range == 0:
        return raw_crps
    alpha = 1 / overall_range
    return raw_crps * alpha


def compute_dataset_metrics(df: pd.DataFrame, title: str = None) -> Dict:
    """
    Computes error metrics for a single dataset.

    Parameters:
        df (pd.DataFrame): A DataFrame containing a 'value' column for actuals and
                           one or more forecast columns starting with "forecast_".
        title (str): dataset label

    Returns:
        pd.DataFrame: A DataFrame with one row per time step and columns for each metric.
    """
    forecast_cols = [col for col in df.columns if col.startswith("forecast_")]
    forecast_matrix = df[forecast_cols].values  # shape: (n_time_points, n_samples)
    actuals = df["value"].values  # shape: (n_time_points,)
    forecast_errors = forecast_matrix - actuals[:, None]

    ### Overall (dataset-level) metrics ###
    mae = np.mean(np.abs(forecast_errors))
    rmse = np.sqrt(np.mean(forecast_errors**2))

    # use forecast mean for wape
    forecast_mean = np.mean(forecast_matrix, axis=1)
    wmape = (
        np.sum(np.abs(actuals - forecast_mean)) / np.sum(np.abs(actuals)) * 100
        if np.sum(np.abs(actuals)) != 0
        else np.nan
    )

    # Compute overall range from the entire dataset's actuals and normalise crps
    overall_range = np.max(actuals) - np.min(actuals)
    norm_crps_list = [
        normalized_crps(actuals[i], forecast_matrix[i, :], overall_range) for i in range(len(actuals))
    ]
    avg_norm_crps = np.mean(norm_crps_list)

    dataset_metrics = pd.DataFrame(
        {
            "Dataset": [title],
            "WMAPE": [float(round(wmape, 2))],
            "CRPS": [float(round(avg_norm_crps, 2))],
            "MAE": [float(round(mae, 2))],
            "RMSE": [float(round(rmse, 2))],
            "n_samples": [len(actuals)],
        }
    )

    ### Per-timestep metrics ###
    mae_list = np.mean(np.abs(forecast_matrix - actuals[:, None]), axis=1)
    rmse_list = np.sqrt(np.mean((forecast_matrix - actuals[:, None]) ** 2, axis=1))
    wmape_list = [
        np.mean(np.abs(forecast_matrix[i] - actuals[i])) / abs(actuals[i]) * 100
        if actuals[i] != 0
        else np.nan
        for i in range(len(actuals))
    ]

    # Construct a per-timestep metrics DataFrame indexed by (Title, Metric)
    metrics_order = ["MAE", "RMSE", "WMAPE"]
    metric_values = {"MAE": mae_list, "RMSE": rmse_list, "WMAPE": wmape_list}

    n_timesteps = len(actuals)
    rows = []
    for metric in metrics_order:
        row = {"Dataset": title, "Metric": metric}
        for i in range(n_timesteps):
            row[f"t_{i + 1}"] = round(metric_values[metric][i], 2)
        rows.append(row)

    timestep_metrics = pd.DataFrame(rows)
    timestep_metrics = timestep_metrics.set_index(["Dataset", "Metric"])

    return dataset_metrics, timestep_metrics


def compute_model_metrics(model_results: Dict[str, pd.DataFrame], model_name: str = None):
    """
    Computes metrics across mutliple datasets for a model.

    Params:
        - model_results (Dict[str, pd.DataFrame]): a dictionary containing a pd.DataFrame
            of forecasts for each dataset.
        - model_name (str, Optional): label for the model to be put into dataframe entries.

    Returns:
        - A dictionary containing:
            * model_metrics (pd.DataFrame): model's overall metrics.
            * dataset_metrics (pd.DataFrame): performance per dataset.
            * timestep_metrics (pd.DataFrame): the model's performance per timestep,
                aggregated across datasets.
            * aggregated_timesteps (pd.DataFrame): model's overall performance over
                timesteps.
    """

    dset_metrics_list = []
    timestep_metrics_list = []

    for dset_name, result_df in model_results.items():
        ds_met, ts_met = compute_dataset_metrics(result_df, title=dset_name)
        dset_metrics_list.append(ds_met)
        timestep_metrics_list.append(ts_met)

    dset_metrics = pd.concat(dset_metrics_list, ignore_index=True)

    timestep_metrics = pd.concat(timestep_metrics_list)

    # aggregate over datasets
    agg_wmape = dset_metrics["WMAPE"].sum() / len(dset_metrics["WMAPE"])
    agg_crps = dset_metrics["CRPS"].sum() / len(dset_metrics["CRPS"])
    agg_mae = dset_metrics["MAE"].sum() / len(dset_metrics["MAE"])
    agg_rmse = dset_metrics["RMSE"].sum() / len(dset_metrics["RMSE"])

    model_metrics = pd.DataFrame(
        {
            "Model": [model_name],
            "WMAPE": [round(agg_wmape, 4)],
            "CRPS": [round(agg_crps, 2)],
            "MAE": [round(agg_mae, 2)],
            "RMSE": [round(agg_rmse, 2)],
        }
    )

    # pivot dataset metrics to be concatenated with other model's
    dset_metrics = dset_metrics.melt(
        id_vars=["Dataset"],
        value_vars=["WMAPE", "CRPS", "MAE", "RMSE"],
        var_name="Metric",
        value_name=model_name,
    )
    dset_metrics = dset_metrics.set_index(["Dataset", "Metric"])

    # aggregate timestep metrics
    agg_ts = timestep_metrics.groupby(level="Metric").mean()
    agg_ts["Model"] = model_name
    agg_ts = agg_ts.reset_index().set_index(["Model", "Metric"])

    return {
        "model_metrics": model_metrics,
        "dataset_metrics": dset_metrics,
        "timestep_metrics": timestep_metrics,
        "agg_timesteps": agg_ts,
    }


def compute_benchmark_metrics(results: Dict[str, Dict]):
    """
    Computes metrics for all models in a benchmarking results object, as read in
    by `read_results()`.

    Params:
        - results (Dict[str, Dict]): a dictionary containing each model run's
            forecasting results.

    Returns:
        - benchmark_metrics: a dictionary containing overall benchmark results,
            as well as each model's individual results per dataset, and time-step
            based metrics.
    """

    model_metrics = {}

    for model_name, model_output in results.items():
        model_metrics[model_name] = compute_model_metrics(model_output["results"], model_name)

    # join together each model's overall metrics
    overall_metrics = pd.concat([x["model_metrics"] for x in model_metrics.values()], ignore_index=True)
    overall_metrics = overall_metrics.set_index("Model").sort_index()

    overall_datasets = (
        pd.concat([x["dataset_metrics"] for x in model_metrics.values()], axis=1)
        .sort_index()
        .sort_index(axis=1)
    )
    overall_timesteps = pd.concat([x["agg_timesteps"] for x in model_metrics.values()]).sort_index()

    return {
        "overall_metrics": overall_metrics,
        "overall_datasets": overall_datasets,
        "overall_timesteps": overall_timesteps,
    }
