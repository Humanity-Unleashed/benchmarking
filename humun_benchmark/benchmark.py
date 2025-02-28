import logging
import os
from datetime import datetime
from typing import List
import json
import pandas as pd

import torch

from humun_benchmark.config import MD_VINTAGE_IDS_MONTHLY, NUMERICAL, setup_logging
from humun_benchmark.data import load_from_parquet
from humun_benchmark.models import HuggingFace
from humun_benchmark.prompts import InstructPrompt

import dotenv

dotenv.load_dotenv()

# timestamp for output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def benchmark(
    models: list[str] = ["llama-3.1-8b-instruct"],
    output_path: str = os.getenv("RESULTS_STORE"),
    datasets_path: str = os.getenv("DATASETS_PATH"),
    series_ids=List[str],
    n_datasets: int = None,
    batch_size: int = 10,
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

    # write logs to output path
    setup_logging(os.path.join(output_path, f"{timestamp}.log"))
    log = logging.getLogger(__name__)

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

    # get data based on selector type
    fred_data = load_from_parquet(
        series_ids=series_ids,
        datasets_path=datasets_path,
        n_datasets=n_datasets,
        forecast_steps=forecast_steps,
        train_ratio=train_ratio,
    )

    # generate forecasts for each model
    for model in models:
        log.info(f"Loading model: {model}")

        # create model instance and log config
        llm = HuggingFace(model)
        model_info = llm.serialise()
        log.info(f"Model Info:\n{model_info}")

        model_result = {"dataset_info": {}, "results": {}}

        for series_id, data in fred_data.items():
            model_result["dataset_info"][series_id] = data["dataset_info"]

            prompt = InstructPrompt(task=NUMERICAL, history=data["history"], forecast=data["forecast"])

            log.info(
                f"Model: {model}\n"
                f" {data['dataset_info']}\n\n"
                f" Prompt Tokens Length: {len(llm.tokenizer.encode(prompt.prompt_text))}\n"
                f" Batch size: {batch_size}\n"
                f" On device: {llm.model.device}"
            )

            # run inference
            llm.inference(payload=prompt, batch_size=batch_size)

            model_result["results"][series_id] = prompt.results.to_json(
                orient="records", date_format="iso"
            )

            # free up memory
            torch.cuda.empty_cache()

        # write results to .parquet for each model
        output_df = pd.DataFrame(
            [
                {
                    "model_info": model_info,
                    "dataset_info": json.dumps(model_result["dataset_info"]),
                    "results": json.dumps(model_result["results"]),
                }
            ]
        )
        output_file = os.path.join(output_path, f"{model.split('/')[-1]}_{timestamp}.parquet")
        output_df.to_parquet(output_file, index=False)
        log.info(f"Model output written to {output_file}")

    log.info("Benchmark completed.")


if __name__ == "__main__":
    models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        # "meta-llama/Llama-3.2-1B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Ministral-8B-Instruct-2410",
    ]

    benchmark(
        models=models,
        series_ids=MD_VINTAGE_IDS_MONTHLY,
        n_datasets=20,
        batch_size=10,
        train_ratio=3,
        forecast_steps=12,
    )
