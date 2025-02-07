import argparse
import json
import logging
import os
from datetime import datetime
from typing import List, Union, Dict
from pprint import pformat

from humun_benchmark.interfaces.huggingface import HuggingFace
from humun_benchmark.prompt import InstructPrompt
from humun_benchmark.utils.checks import check_env
from humun_benchmark.utils.log_config import setup_logging
from humun_benchmark.utils.tasks import NUMERICAL
from humun_benchmark.utils.get_data import (
    get_data,
    get_series_by_id,
    get_dataset_info,
    convert_array_to_df,
)


# load .env and check needed variables exist
check_env()

# timestamp for output files
time = datetime.now().strftime("%Y%m%d_%H%M%S")


def benchmark(
    datasets_path: str = os.getenv("DATASETS_PATH"),
    metadata_path: str = os.getenv("METADATA_PATH"),
    output_path: str = os.getenv("RESULTS_STORE"),
    selector: Union[Dict, List[str]] = {"frequency": "Monthly"},
    n_datasets: int = 3,
    models: list[str] = ["llama-3.1-8b-instruct"],
    batch_size: int = 1,
) -> None:
    """
    Run benchmarks on time series data, selecting data either by filters or series IDs.

    Args:
        datasets_path: Path to time series data
        metadata_path: Path to metadata
        output_path: Where to store results
        selector: Either dict of filters or list of series IDs
        n_datasets: Number of datasets to retrieve (used with filters)
        models: List of model names to benchmark
        batch_size: Number of runs per inference
    """

    # Validate required paths
    if not all([datasets_path, metadata_path]):
        raise ValueError(
            "datasets_path and metadata_path must be provided either via arguments or environment variables"
        )

    # Setup output directory and logging
    output_path = output_path or os.getenv("RESULTS_STORE")
    output_path = os.path.join(output_path, time)
    os.makedirs(output_path, exist_ok=True)
    setup_logging(f"{output_path}/benchmark.log")
    log = logging.getLogger("humun_benchmark.benchmark")

    # Log the selection method being used
    selection_method = "series_ids" if isinstance(selector, list) else "filters"
    params = {
        "datasets_path": datasets_path,
        "metadata_path": metadata_path,
        "output_path": output_path,
        f"{selection_method}": selector,
        "n_datasets": n_datasets,
        "models": models,
        "batch_size": batch_size,
    }
    log.info(f"Run Parameters:\n{pformat(params)}")

    log.info("Reading in Metadata and Datasets...")
    # Get data based on selector type
    if isinstance(selector, list):
        fred_data = get_series_by_id(
            series_ids=selector,
            datasets_path=datasets_path,
            metadata_path=metadata_path,
        )
    else:
        fred_data = get_data(
            n_datasets=n_datasets,
            datasets_path=datasets_path,
            metadata_path=metadata_path,
            filters=selector,
        )

    # Log data info: series selected
    log.info(get_dataset_info(fred_data))

    # for model in models
    for model in models:
        # create model instance and log config
        llm = HuggingFace(model)
        log.info(f"Model Info:\n{pformat(llm.serialise())}")

        results = {}

        # for timeseries in data selected
        for series_id, data in fred_data.items():
            results[series_id] = {}

            log.info(f"Prompting {model} for Series ID: {series_id}:")
            timeseries_df = convert_array_to_df(fred_data[series_id]["timeseries"])

            # create a prompt
            prompt = InstructPrompt(task=NUMERICAL, timeseries=timeseries_df)
            log.info(
                f"Prompt Tokens Length: {len(llm.tokenizer.encode(prompt.prompt_text))}"
            )

            # run inference
            llm.inference(payload=prompt, n_runs=batch_size)

            # store results  (TODO: currently overrides results_df on each inference)
            results[series_id]["results"] = prompt.results_df.to_json(
                orient="records", date_format="iso"
            )

            # store metadata
            results[series_id]["metadata"] = data["metadata"]

        # save results to <modelname>.json
        if results:
            json_path = f"{output_path}/{llm.label}.json"
            with open(json_path, "w") as f:
                json.dump(results, f)
            log.info(f"Results saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarks for Instruct LLMs on time series data."
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default=os.getenv("DATASETS_PATH"),
        help="Path to parquet file containing time series data",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=os.getenv("METADATA_PATH"),
        help="Path to CSV file containing metadata",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.getenv("RESULTS_STORE"),
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--series_ids",
        type=str,
        nargs="+",
        help="List of series IDs to benchmark",
    )
    parser.add_argument(
        "--filters",
        type=json.loads,
        default='{"frequency": "Monthly"}',
        help='JSON string of filters e.g. \'{"frequency": "Monthly"}\'',
    )
    parser.add_argument(
        "--n_datasets",
        type=int,
        default=3,
        help="Number of datasets to retrieve when using filters",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["llama-3.1-8b-instruct"],
        help="List of models to benchmark",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of inferences per dataset",
    )
    args = parser.parse_args()

    # Determine selector based on provided arguments
    if args.series_ids:
        selector = args.series_ids
        n_datasets = None
    else:
        selector = args.filters
        n_datasets = args.n_datasets

    # Remove selector-related args before passing to benchmark
    vars_dict = vars(args)
    del vars_dict["series_ids"]
    del vars_dict["filters"]
    del vars_dict["n_datasets"]

    benchmark(selector=selector, n_datasets=n_datasets, **vars_dict)


# Note: currently /workspace/ does not have enough space, so all_fred_metadata.csv has been downloaded into personal directory, use humun_benchmark/adhoc/downloadGC.py (get API key from link in file).

# Due to the same issue above, if using models not available on /workspace/huggingface_cache/ , alter .env HF_HOME value to be user-based (or delete it).

# Usage example:
#  python humun_benchmark/benchmark.py --metadata_path 'all_fred_metadata.csv' --output_path . --models llama-3.1-8b-instruct ministral-8b-instruct-2410
