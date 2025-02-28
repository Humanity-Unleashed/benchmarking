import logging
import os
from datetime import datetime
from typing import List

import torch

from humun_benchmark.config import MD_VINTAGE_IDS_MONTHLY, NUMERICAL, setup_logging
from humun_benchmark.data import load_from_parquet
from humun_benchmark.models import HuggingFace
from humun_benchmark.prompts import InstructPrompt

import dotenv

dotenv.load_dotenv()


def benchmark(
    models: list[str] = ["llama-3.1-8b-instruct"],
    output_path: str = os.getenv("RESULTS_STORE"),
    datasets_path: str = os.getenv("DATASETS_PATH"),
    series_ids=List[str],
    n_datasets: int = None,
    batch_size: int = 1,
    train_ratio: int = 3,
    forecast_steps: int = 12,
) -> None:
    """
    Run benchmarks on time series data, selecting data either by filters or series IDs.

    Args:
        models: List of model names to benchmark
        output_path: Where to store results
        datasets_path: Path to time series data
        series_ids: List of series IDs from FRED data
        n_datasets: Number of datasets to retrieve (used with filters)
        batch_size: Number of runs per inference
        train_ratio: Multiplier for training period
        forecast_steps: Number of forecast steps
    """

    # Log the selection method being used
    params = {
        "models": models,
        "output_path": output_path,
        "datasets_path": datasets_path,
        "series_ids": series_ids[:n_datasets] if n_datasets else series_ids,
        "n_datasets": n_datasets,
        "batch_size": batch_size,
        "train_ratio": train_ratio,
        "forecast_steps": forecast_steps,
    }
    params_str = "\n".join(f"\t{k}: {v}" for k, v in params.items())
    log.info(f"Benchmark Parameters: {{\n{params_str}\n}}")

    # Get data based on selector type
    fred_data = load_from_parquet(
        series_ids=series_ids,
        datasets_path=datasets_path,
        n_datasets=n_datasets,
        forecast_steps=forecast_steps,
        train_ratio=train_ratio,
    )

    for series_id, data in fred_data.items():
        log.info(data["dataset_info"])
        prompt = InstructPrompt(task=NUMERICAL, history=data["history"], forecast=data["forecast"])
        log.info(f" ID: {series_id} Prompt text:\n {prompt.prompt_text}")


if __name__ == "__main__":
    setup_logging()
    log = logging.getLogger("humun_benchmark.benchmark")

    log.info("Running benchmark().")

    benchmark(series_ids=MD_VINTAGE_IDS_MONTHLY, n_datasets=3)

    log.info("Benchmark completed.")
