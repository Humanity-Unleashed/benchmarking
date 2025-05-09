from abc import ABC, abstractmethod
import logging
import pandas as pd
from typing import Dict, Any, Union

log = logging.getLogger(__name__)


class ModelError(Exception):
    pass


class InferenceError(ModelError):
    pass


class ModelLoadError(ModelError):
    pass


class Model(ABC):
    def __init__(self, label: str, model_type: str):
        self.label = label
        self.model_type = model_type  # e.g., "llm", "statistical", "ml"

    @abstractmethod
    def _load_model(self, **kwargs):
        """
        Loads the model.
        Specific arguments for loading (e.g., path, hyperparameters)
        can be passed via kwargs.
        """
        log.exception("'_load_model' must be implemented in subclass.")
        raise NotImplementedError("'_load_model' must be implemented in subclass.")

    @abstractmethod
    def predict(self, data: Dict[str, Union[pd.DataFrame, str]], **kwargs) -> pd.DataFrame:
        """
        Generates forecasts.

        Args:
            data (dict): A dictionary containing 'history' and 'forecast' DataFrames,
                along with optional 'dataset_info' as a string.
            **kwargs: Additional model-specific arguments for prediction (e.g., batch_size for LLMs).

        Returns:
            pd.DataFrame: A DataFrame with predictions.
                          Should contain at 'date' and 'forecast_<sample_number>' columns.
                          i.e. forecast_1
        """
        log.exception("`predict` must be implemented in subclass.")
        raise NotImplementedError("`predict` must be implemented in subclass.")

    @abstractmethod
    def serialise(self) -> Dict[str, Any]:
        """
        Serialize model configuration details for logging.
        Returns dict with model name, architecture details, etc.
        """
        log.exception("`serialise` must be implemented in subclass.")
        raise NotImplementedError("`serialise` must be implemented in subclass.")

    def get_model_type(self) -> str:
        return self.model_type
