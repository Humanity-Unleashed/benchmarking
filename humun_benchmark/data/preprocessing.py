import pandas as pd
from typing import List, Dict
import os
import logging
import ast

from humun_benchmark.config.common import SERIES_IDS


log = logging.getLogger(__name__)


def truncate_dataset(
    timeseries_df: pd.DataFrame, train_ratio: int = 3, n_steps: int = 12
) -> pd.DataFrame:
    """
    Truncate a timeseries DataFrame to the last (train_ratio+1)*n_steps rows.

    If the dataset is too short, a ValueError is raised.

    Args:
        timeseries_df (pd.DataFrame): Sorted timeseries data.
        train_ratio (int): Multiplier for training periods (default: 3).
        n_steps (int): Number of forecast steps (default: 12).

    Returns:
        pd.DataFrame: The truncated DataFrame.
    """
    total_required = (train_ratio + 1) * n_steps
    if len(timeseries_df) < total_required:
        raise ValueError(
            f"Dataset length ({len(timeseries_df)}) is less than the required {total_required} "
            f"points for train_ratio={train_ratio} and n_steps={n_steps}."
        )

    truncated_df = timeseries_df.iloc[-total_required:]
    return truncated_df


# TODO : need metadata.csv? fred.parquet contains some metadata
def load_from_parquet(
    series_ids: List[str],
    datasets_path: str = os.getenv("DATASETS_PATH"),
    n_datasets: int = None,
    forecast_steps: int = 6,
    train_ratio: int = 3,
    require_metadata: str = None,
) -> Dict[str, Dict]:
    """
    Get timeseries data for specific series IDs, split into history and forecast based on
    params provided.

    Args:
        series_ids: List of series IDs to retrieve.
        datasets_path: Path to parquet file with time series data.
        n_datasets: Limits the number of series to use.
        forecast_steps: Number of timesteps in forecast.
        train_ratio: Multiplier for training data (i.e. ratio of 3 * 6 = 18 training timesteps)
        require_metadata: Filepath of metadata - checks if series ID is contained in metadata,
            excluding from returned dictionary if not.

    Returns:
        Dictionary of format:
            { "id1": { "history": pd.DataFrame, "forecast": pd.DataFrame },
              "id2": ... }
    """

    # if required, check if metadata exists
    if require_metadata:
        missing = set(series_ids) - set(pd.read_csv(require_metadata)["id"])
        if missing:
            log.warning(f"Skipping series IDs without metadata: {list(missing)}.")
            series_ids = [sid for sid in series_ids if sid not in missing]
        if not series_ids:
            raise ValueError("No valid series IDs remain after checking metadata.")

    if n_datasets is not None:
        if n_datasets > len(series_ids):
            raise ValueError("n_datasets is greater than the number of series_ids.")
        series_ids = series_ids[:n_datasets]

    datasets = pd.read_parquet(datasets_path, filters=[("series_id", "in", series_ids)]).convert_dtypes()

    result = {}

    for sid in series_ids:
        try:
            series_dict = datasets[datasets["series_id"] == "AAA"].iloc[0].to_dict()

            # convert string to dictionary (not a JSON string)
            series_data = ast.literal_eval(series_dict["data"])

            # Check for missing values.
            if any(v == "." for v in series_data.values()):
                log.warning(f"Series {sid} contains missing values ('.'). Skipping.")
                continue

            timeseries_df = pd.DataFrame(series_data.items(), columns=["date", "value"])
            timeseries_df["date"] = pd.to_datetime(timeseries_df["date"])
            timeseries_df["value"] = timeseries_df["value"].astype(float)

            # truncate data
            history, forecast = truncate_dataset(timeseries_df, train_ratio, forecast_steps)

            result[sid] = {"history": history, "forecast": forecast}

        except (KeyError, IndexError) as e:
            log.warning(f"Error processing series {sid}: {str(e)}")
            continue

    return result


# def get_dataset_info(fred_data: dict):
#     """
#     Create formatted strings for each series with their details.
#
#     Args:
#         fred_data (Dict[str, Dict]) : a dictionary containing, for each series ID, it's
#         metadata and timeseries data. Note this is a processed subset of the fred data-source.
#
#     Returns:
#         A formatted string with dataset info for logging.
#     """
#     series_details = []
#     for series_id, data in fred_data.items():
#         length = len(data["timeseries"])
#         frequency = data["metadata"].get("frequency", "Unknown")
#         title = data["metadata"].get("title", "Unknown")
#
#         detail_str = f"Series ID: {series_id}\n  Length: {length} points\n  Frequency: {frequency}\n  Title: {title}\n"
#         series_details.append(detail_str)
#
#     # Join all details with a separator line.
#     separator = "-" * 50 + "\n"
#     return separator.join(series_details)


if __name__ == "__main__":
    import pickle as pkl

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result = load_from_parquet(
        series_ids=SERIES_IDS,
        datasets_path="/workspace/datasets/fred/fred.parquet",
        require_metadata="/workspace/datasets/fred/all_fred_metadata.csv",
    )

    outputfile = "result.pkl"
    with open(outputfile, "wb") as f:
        pkl.dump(result, f)

    log.info(f"written to {outputfile}")
