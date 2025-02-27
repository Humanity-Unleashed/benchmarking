import logging
import re

import pandas as pd

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


def format_timeseries_input(df: pd.DataFrame, n_timesteps: int) -> str:
    """
    Formats and returns the timeseries history used by the model to forecast
    unseen timestamp values.

    https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/direct_prompt.py#L99C5-L112C69

    Example:

    <history>
    (t1, v1)
    (t2, v2)
    (t3, v3)
    </history>
    <forecast>
    (t4, v4)
    (t5, v5)
    </forecast>
    """

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'value' columns.")
    if not isinstance(n_timesteps, int) or n_timesteps < 1:
        raise ValueError("n_timesteps must be a positive integer")
    if n_timesteps >= len(df):
        raise ValueError("n_timesteps cannot be larger than length of timeseries")

    # Ensure sorted by date
    df = df.sort_values(by="date", ascending=True)

    # Ensure the 'date' column is in daily 'YYYY-MM-DD' format
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.strftime(
            "%Y-%m-%d"
        )
    except Exception as e:
        raise ValueError(f"Date column contains invalid formats: {e}")

    # Split data for forecasting
    # TODO: add optional history shortener (e.g. give 5 yrs instead of 40+)
    history = df.iloc[:-n_timesteps]
    forecast = df.iloc[n_timesteps:]

    history_formatted = "\n".join(
        f"({row['date']}, {row['value']})" for _, row in history.iterrows()
    )
    history_section = f"<history>\n{history_formatted}\n</history>"

    forecast_formatted = "\n".join(
        f"({row['date']}, x)" for _, row in forecast.iterrows()
    )
    forecast_section = f"<forecast>\n{forecast_formatted}\n</forecast>"

    return f"{history_section}\n{forecast_section}"


def format_output_regex(timestamps):
    """
    Regex-enforced model output using a list of timestamps.
    """
    timestamp_regex = "".join(
        [
            r"\(\s*{}\s*,\s*[-+]?\d+(\.\d+)?\)\n".format(re.escape(ts))
            for ts in timestamps
        ]
    )
    return r"<forecast>\n{}<\/forecast>".format(timestamp_regex)
