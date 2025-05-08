import pandas as pd
from typing import List, Dict, Tuple
import os
import logging
import ast

from humun_benchmark.config import MD_VINTAGE_IDS_MONTHLY


log = logging.getLogger(__name__)


def truncate_dataset(
    timeseries_df: pd.DataFrame, train_ratio: int = 3, n_steps: int = 12
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Truncate a timeseries DataFrame to the last (train_ratio+1)*n_steps rows.

    If the dataset is too short, a ValueError is raised.

    Args:
        timeseries_df (pd.DataFrame): Sorted timeseries data.
        train_ratio (int): Multiplier for training periods (default: 3).
        n_steps (int): Number of forecast steps (default: 12).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (history, forecast)
    """
    total_required = (train_ratio + 1) * n_steps
    if len(timeseries_df) < total_required:
        raise ValueError(
            f"Dataset length ({len(timeseries_df)}) is less than the required {total_required} "
            f"points for train_ratio={train_ratio} and n_steps={n_steps}."
        )

    truncated_df = timeseries_df.iloc[-total_required:]

    history = truncated_df.iloc[:-n_steps]
    forecast = truncated_df.iloc[-n_steps:]

    return history, forecast


def load_from_parquet(
    series_ids: List[str],
    datasets_path: str = os.getenv("DATASETS_PATH"),
    n_datasets: int = None,
    forecast_steps: int = 6,
    train_ratio: int = 3,
    cutoff_year: int = None,
) -> Dict[str, Dict]:
    """
    Get timeseries data for specific series IDs, split into history and forecast.

    Args:
        series_ids: List of series IDs to retrieve.
        datasets_path: Path to parquet file with time series data.
        n_datasets: Limits the number of series to use.
        forecast_steps: Number of timesteps in forecast.
        train_ratio: Multiplier for training data (i.e. ratio of 3 * 6 = 18 training timesteps)
        cutoff_year: If provided, only include data from this year onwards (e.g., 2024)

    Returns:
        Dictionary of format:
            { "id1": { "history": pd.DataFrame, "forecast": pd.DataFrame, "title": str, "notes" : str },
              "id2": ... }
    """

    if n_datasets is not None:
        if n_datasets > len(series_ids):
            raise ValueError("n_datasets is greater than the number of series_ids.")
        series_ids = series_ids[:n_datasets]

    datasets = pd.read_parquet(datasets_path, filters=[("series_id", "in", series_ids)]).convert_dtypes()

    result = {}

    for sid in series_ids:
        try:
            if datasets[datasets["series_id"] == sid].empty:
                log.warning(f"Series ID {sid} has no data.")
                continue

            series_dict = datasets[datasets["series_id"] == sid].iloc[0].to_dict()

            # convert string to dictionary (not a JSON string)
            series_data = ast.literal_eval(series_dict["data"])

            # check for missing values.
            if any(v == "." for v in series_data.values()):
                log.warning(f"Series {sid} contains missing values ('.'). Skipping.")
                continue

            timeseries_df = pd.DataFrame(series_data.items(), columns=["date", "value"])

            timeseries_df["date"] = pd.to_datetime(timeseries_df["date"])
            timeseries_df["value"] = timeseries_df["value"].astype(float)
            
            # Apply year cutoff filter if specified
            if cutoff_year is not None:
                timeseries_df = timeseries_df[timeseries_df["date"].dt.year >= cutoff_year]
                
                # Check if we still have data after filtering
                if len(timeseries_df) == 0:
                    log.warning(f"Series {sid} has no data after applying year cutoff of {cutoff_year}. Skipping.")
                    continue
                    
                # Check if we have enough data for truncation after filtering
                total_required = (train_ratio + 1) * forecast_steps
                if len(timeseries_df) < total_required:
                    log.warning(
                        f"Series {sid} has insufficient data after year cutoff ({len(timeseries_df)} rows, "
                        f"need {total_required}). Skipping."
                    )
                    continue

            # truncate data
            history, forecast = truncate_dataset(timeseries_df, train_ratio, forecast_steps)

            title, notes = series_dict["title"], series_dict["notes"]

            result[sid] = {"history": history, "forecast": forecast, "title": title, "notes": notes}

            # info string for logging
            result[sid]["dataset_info"] = dataset_info(sid, title, timeseries_df, history, forecast)

        except (KeyError, IndexError) as e:
            log.warning(f"Error processing series {sid}: {str(e)}")
            continue

    return result


def dataset_info(series_id, title, original, history, forecast) -> str:
    o_len = len(original["value"])
    n_len = len(history["value"]) + len(forecast["value"])
    freq = pd.infer_freq(original["date"].sort_values()) or "Unknown"
    
    # Add date range info to make year filtering more transparent
    date_range = f"{original['date'].min().strftime('%Y-%m-%d')} to {original['date'].max().strftime('%Y-%m-%d')}"

    return (
        f"ID: {series_id}\n"
        f"Title: {title}\n"
        f"Frequency: {freq}\n"
        f"Date Range: {date_range}\n"
        f"Values: {o_len} -> {n_len}"
    )


if __name__ == "__main__":
    import pickle as pkl

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result = load_from_parquet(
        series_ids=MD_VINTAGE_IDS_MONTHLY,
        datasets_path="/workspace/datasets/fred/fred.parquet",
        cutoff_year=2022,
    )

    outputfile = "result.pkl"
    with open(outputfile, "wb") as f:
        pkl.dump(result, f)

    log.info(f"written to {outputfile}")