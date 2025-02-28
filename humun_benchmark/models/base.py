"""
Parent class for model interfaces.

"""

from humun_benchmark.prompts import Prompt

import logging

logger = logging.getLogger(__name__)


class ModelError(Exception):
    pass


class InferenceError(ModelError):
    pass


class ModelLoadError(ModelError):
    pass


class Model:
    def __init__(self, label: str):
        self.label = label
        self._load_model()

    def _load_model(self):
        logger.exception("'_load_model' must be implemented in subclass.")
        raise NotImplementedError("'_load_model' must be implemented in subclass.")

    def inference(self, prompt_instance: Prompt, batch_size: int) -> None:
        """
        in-place method that appends inference responses to Prompt instance
        """
        logger.exception("`inference` must be implemented in subclass.")
        raise NotImplementedError("`inference` must be implemented in subclass.")

    def serialise():
        """
        Write out model configuration for logs.
        """
        logger.exception("`serialise` must be implemented in subclass.")
        raise NotImplementedError("`serialise` must be implemented in subclass.")
