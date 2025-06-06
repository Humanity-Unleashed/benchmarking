from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from .formatting import format_timeseries_input


# Parent class for all prompt methods
class Prompt(BaseModel):
    task: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    context: Optional[str] = None
    prompt_text: Optional[str] = None
    responses: List[str] = []

    class Config:
        arbitrary_types_allowed = True  # Allow pd.DataFrame as a field

    def serialise(self) -> Dict[str, Any]:
        """
        Serialize prompt configuration and data for logging.
        """
        info = (
            {
                "task": self.task,
                "history": self.history,
                "forecast": self.forecast,
                "context": self.context,
                "prompt_text": self.prompt_text,
                "responses": len(self.responses),
            },
        )

        return info


class InstructPrompt(Prompt):
    results: Optional[pd.DataFrame] = None

    def __init__(
        self,
        task: str,
        history: pd.DataFrame,
        forecast: pd.DataFrame,
        context: str = None,
    ):
        # calls pydantic constructor to initialise fields
        super().__init__(
            task=task,
            history=history,
            forecast=forecast,
            context=context,
        )
        self.prompt_text = self._format_input()

    def _format_input(self) -> str:
        """
        Returns formatted input text.
        """
        prompt_text = self.task

        if self.context:
            prompt_text += f"<context>\n{self.context}\n</context>\n"

        prompt_text += format_timeseries_input(self.history, self.forecast)
        return prompt_text

    def merge_forecasts(self, dfs: List[pd.DataFrame]):
        """
        Merge forecast responses together and include the original forecast values for metric calculation.
        """

        # Rename the value columns to forecast_1, forecast_2, ..., forecast_n
        for i, df in enumerate(dfs, start=1):
            df.rename(columns={"value": f"forecast_{i}"}, inplace=True)

        # Merge all forecast dataframes on the date column
        merged_df = dfs[0]
        if len(dfs) > 1:
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on="date", how="outer")

        # Ensure date columns are datetime type
        merged_df["date"] = pd.to_datetime(merged_df["date"])
        self.forecast["date"] = pd.to_datetime(self.forecast["date"])

        # Merge with original forecast values to obtain actual value column for metrics
        return pd.merge(
            merged_df,
            self.forecast[["date", "value"]],
            on="date",
            how="inner",
        )


class MultiModalPrompt(Prompt):
    """
    Prompt for multimodal models.
    """

    pass
