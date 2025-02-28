import logging
import re

import pandas as pd

log = logging.getLogger(__name__)


def format_timeseries_input(history: pd.DataFrame, forecast: pd.DataFrame) -> str:
    """
    Formats and returns the a string of the timeseries history used by the transformer to forecast
    unseen timestamp values.

    Example:

    <history>
    (t1, v1)
    (t2, v2)
    (t3, v3)
    </history>
    <forecast>
    (t4, x)
    (t5, x)
    </forecast>
    """

    for name, df in [("history", history), ("forecast", forecast)]:
        if "date" not in df.columns or "value" not in df.columns:
            raise ValueError(f"{name} DataFrame must contain 'date' and 'value' columns.")

        df.sort_values(by="date", ascending=True, inplace=True)

        # Ensure 'date' column is in daily 'YYYY-MM-DD' format
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"{name} DataFrame contains invalid date formats: {e}")

    history_formatted = "\n".join(f"({row['date']}, {row['value']})" for _, row in history.iterrows())
    history_section = f"<history>\n{history_formatted}\n</history>"

    forecast_formatted = "\n".join(f"({row['date']}, x)" for _, row in forecast.iterrows())
    forecast_section = f"<forecast>\n{forecast_formatted}\n</forecast>"

    return f"{history_section}\n{forecast_section}"


def format_output_regex(timestamps):
    """
    Regex-enforced model output using a list of timestamps.
    """
    timestamp_regex = "".join(
        [r"\(\s*{}\s*,\s*[-+]?\d+(\.\d+)?\)\n".format(re.escape(ts)) for ts in timestamps]
    )
    return r"<forecast>\n{}<\/forecast>".format(timestamp_regex)


def parse_forecast_output(response: str) -> pd.DataFrame:
    """Parse forecast output text into a DataFrame."""
    # Get forecast section
    forecast_pattern = r"<forecast>\n(.*?)</forecast>"
    forecast_match = re.search(forecast_pattern, response, re.DOTALL)

    if not forecast_match:
        log.info(f"Response: {response}")
        raise ValueError("No forecast section found in response")

    forecast_text = forecast_match.group(1)

    # Parse (date, value) pairs
    data_matches = re.findall(r"\(([\d\-\s:]+),\s*(-?[\d.]+)\)", forecast_text)

    if not data_matches:
        raise ValueError("No valid forecast data found in response")

    # Convert to DataFrame
    forecast_data = [(date.strip(), float(value)) for date, value in data_matches]
    df = pd.DataFrame(forecast_data, columns=["date", "value"])

    # Convert dates to datetime
    df["date"] = pd.to_datetime(df["date"])

    return df
