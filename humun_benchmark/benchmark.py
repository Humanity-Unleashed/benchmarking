import concurrent.futures
import json
import logging
import multiprocessing
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import dotenv
import pandas as pd
import torch

from humun_benchmark import MODEL_REGISTRY
from humun_benchmark.config import MD_VINTAGE_IDS_MONTHLY, setup_logging
from humun_benchmark.data import load_from_parquet
from humun_benchmark.models import HuggingFace

dotenv.load_dotenv()


log = logging.getLogger("humun_benchmark.benchmark")


def single_model_run(
    model_name: str,
    category: str,
    gpu_id: int,
    fred_data: Dict[str, Dict[str, Any]],
    output_path: str,
    batch_size: int,
    context: bool,
    level: int = logging.INFO,
) -> None:
    """
    Benchmarks a single model on a specific GPU. Designed to be run in a separate process.

    Args:
        model_name: The name/path of the Hugging Face model.
        gpu_id: The GPU device ID to use for this model.
        fred_data: The preloaded dictionary containing dataset information and time series data.
        output_path: Base directory to store results.
        batch_size: Number of runs per inference batch (within the model's inference method).
        context: Whether to supply context to the prompt.
        level: logging level to set all loggers at. Default to INFO
    """

    # configure logging specific to this worker process
    setup_logging(os.path.join(output_path, "benchmark.log"), level=level)
    try:
        if category == "llm":
            torch.cuda.set_device(gpu_id)
            device_name = torch.cuda.get_device_name(gpu_id)
            worker_log = logging.getLogger(
                f"humun_benchmark.benchmark.{model_name.split('/')[-1]}.GPU{gpu_id}"
            )
            worker_log.debug(f"Process started for model '{model_name}' on GPU {gpu_id} {device_name}")
        else:
            worker_log = logging.getLogger(f"humun_benchmark.{model_name.split('/')[-1]}.CPU")
            worker_log.debug("Running on CPU")

        # load model
        ModelClass = MODEL_REGISTRY.get(model_name, HuggingFace)
        cuda_arg = gpu_id if category == "llm" else None  # For non-LLMs run on CPU
        model = ModelClass(model_name, cuda=cuda_arg)
        model._load_model()

        model_info = model.serialise()
        worker_log.debug(f"Model loaded. Info:\n{model_info}")

        model_results = {"dataset_info": {}, "results": {}}
        # iterate through datasets for this model
        for series_id, data in fred_data.items():
            model_results["dataset_info"][series_id] = data["dataset_info"]
            worker_log.debug(f"Starting inference for {series_id}... Batch size: {batch_size}")
            # run Inference
            try:
                result: pd.DataFrame = model.predict(
                    data=data,
                    context=context,
                    batch_size=batch_size,
                    # extra args
                )
                model_results["results"][series_id] = result.to_json(orient="records", date_format="iso")
            except Exception as e:
                worker_log.error(f"Inference failed for dataset {series_id}: {e}", exc_info=True)
                model_results["results"][series_id] = json.dumps({"error": str(e)})

            torch.cuda.empty_cache()

        # save model results
        output_df = pd.DataFrame(
            [
                {
                    "model_info": model_info,
                    "dataset_info": json.dumps(model_results["dataset_info"]),
                    "results": json.dumps(model_results["results"]),
                }
            ]
        )

        # Use the shared run_timestamp for consistent naming
        output_file = os.path.join(output_path, f"{model_name.split('/')[-1]}.parquet")
        output_df.to_parquet(output_file, index=False)

        # clean up memory
        del model
        torch.cuda.empty_cache()
        worker_log.info(
            f"Finished processing model '{model_name}' on GPU {gpu_id}. Results written to {output_file}"
        )

    except Exception as e:
        # Log any exceptions that occur during the entire process for this model
        worker_log.error(
            f"FATAL error processing model '{model_name}' on GPU {gpu_id}: {e}", exc_info=True
        )
        error_file = os.path.join(output_path, f"{model_name.split('/')[-1]}.ERROR")
        with open(error_file, "w") as f:
            f.write(f"Error processing model {model_name} on GPU {gpu_id}:\n{e}\n")
            import traceback

            traceback.print_exc(file=f)


def benchmark(
    models: Dict[str, List[str]],
    output_path: str = os.getenv("RESULTS_STORE", "results"),  # Default if env var not set
    datasets_path: str = os.getenv("DATASETS_PATH"),
    series_ids: List[str] = MD_VINTAGE_IDS_MONTHLY,
    n_datasets: Optional[int] = None,
    batch_size: int = 10,
    train_ratio: int = 3,
    forecast_steps: int = 12,
    context: bool = False,
    available_gpu_ids: List[int] = [],
    level: int = logging.INFO,
) -> None:
    """
    Run benchmarks concurrently on available GPUs, assigning one model per GPU.

    Args:
        models: Dict of model types with List of model names to benchmark for each.
        output_path: Where to store results.
        datasets_path: Path to time series data.
        series_ids: List of series IDs from FRED data.
        n_datasets: Max number of datasets to load (loads all if None or > len(series_ids)).
        batch_size: Number of runs per inference (passed to worker).
        train_ratio: Multiplier for training period.
        forecast_steps: Number of forecast steps.
        context: Supply title and notes per dataset as context.
    """

    # basic setup / validation
    if not datasets_path:
        log.error("DATASETS_PATH environment variable not set.")
        raise ValueError("datasets_path must be provided.")
    if not output_path:
        log.warning("RESULTS_STORE environment variable not set. Defaulting to './results'")
        output_path = "results"
    os.makedirs(output_path, exist_ok=True)  # Ensure output dir exists

    if not torch.cuda.is_available():
        log.error("CUDA is not available. Cannot run on GPUs.")
        raise RuntimeError("CUDA is required but not available.")

    num_gpus = len(available_gpu_ids) if available_gpu_ids else torch.cuda.device_count()
    if num_gpus == 0:
        log.error("No CUDA GPUs found.")
        raise RuntimeError("No CUDA GPUs detected.")

    log.debug(f"Detected {num_gpus} CUDA GPUs.")

    # determine actual series IDs to use
    if n_datasets is not None and n_datasets < len(series_ids):
        selected_series_ids = series_ids[:n_datasets]
        log.debug(f"Using the first {n_datasets} series IDs.")
    else:
        selected_series_ids = series_ids
        n_datasets = len(selected_series_ids)  # update for logging
        log.debug(f"Using all {n_datasets} provided series IDs.")

    # log params
    params = {
        "models": models,
        "output_path": output_path,
        "datasets_path": datasets_path,
        "selected_series_ids_count": len(selected_series_ids),
        "n_datasets_requested": n_datasets if n_datasets is not None else "all",
        "batch_size": batch_size,
        "train_ratio": train_ratio,
        "forecast_steps": forecast_steps,
        "context": context,
    }
    params_str = "\n".join(f"\t{k}: {v}" for k, v in params.items())
    log.info(f"\nBenchmark Parameters: {{\n{params_str}\n}}")

    # load all necessary data before starting parallel processes
    log.info(f"Loading {len(selected_series_ids)} datasets...")
    try:
        fred_data = load_from_parquet(
            series_ids=selected_series_ids,
            datasets_path=datasets_path,
            forecast_steps=forecast_steps,
            train_ratio=train_ratio,
        )
        if not fred_data:
            log.error("No data loaded. Check series IDs and dataset path.")
            return
        log.info("Dataset loading complete.")
    except Exception as e:
        log.error(f"Failed to load data: {e}", exc_info=True)
        raise

    tasks = []
    for category, names in models.items():
        if not names:
            continue
        for name in names:
            tasks.append((category, name))

    # Execute benchmarks in parallel,
    # if n_models > max_workers, models will be ran after a GPU becomes available
    futures = []
    log.info(f"Starting benchmark tasks using up to {num_gpus} GPUs concurrently.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        for i, (category, model_name) in enumerate(tasks):
            gpu_id = available_gpu_ids[i % num_gpus]  # assign in round-robin fashion
            log.debug(f"Submitting {category} model '{model_name}' → GPU {gpu_id}")
            future = executor.submit(
                single_model_run,
                model_name=model_name,
                category=category,
                gpu_id=gpu_id,
                fred_data=fred_data,
                output_path=output_path,
                batch_size=batch_size,
                context=context,
                level=level,
            )
            futures.append(future)

        log.debug("All benchmark tasks submitted. Waiting for completion...")
        completed_count = 0
        failed_count = 0
        # use as_completed to process results as they finish- good for large runs
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                completed_count += 1
                log.info(
                    f"A model benchmark task completed successfully ({completed_count}/{len(tasks)})."
                )
            except Exception as e:
                failed_count += 1
                # Logging of the specific error is handled within the worker's except block
                log.error(f"A model benchmark task failed. Error: {e}")

        log.info(
            f"\n{'-' * 30}"
            f"\nBenchmark run completed."
            f"\nTotal models processed: {len(tasks)}"
            f"\nSuccessful: {completed_count}"
            f"\nFailed: {failed_count}"
            "\n"
            f"\nOutputs written to {output_path}\n"
            f"{'-' * 30}"
        )


if __name__ == "__main__":
    try:
        # set start method to 'spawn' - Necessary for CUDA on Linux AND macOS
        multiprocessing.set_start_method("spawn", force=True)
        log.debug("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        log.warning(f"Warning: Could not set multiprocessing start method ('{e}'). Using default.")
        pass

    # possible to make a flag? or as input JSON/YAML file
    # Define models and other parameters here
    models_to_test = {
        "llm": [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Ministral-8B-Instruct-2410",
        ],
        "ml": None,  # e.g. xgboost
        "statistical": None,  # e.g. arima
    }

    # make an input_flag --datasets_path, default to os.getenv("RESULTS_STORE")
    datasets_path_env = os.getenv("DATASETS_PATH")

    # make an input flag --output_directory, default to os.getenv("RESULTS_STORE")
    results_store_env = os.getenv("RESULTS_STORE")
    base_results = results_store_env  # e.g. "./results"

    # make an input flag --benchmark_name, default to RUN_TIMESTAMP
    timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250508_123456"
    timestamp_folder = "llm_zeroshot_with_ctx_" + timestamp_folder
    run_dir = os.path.join(base_results, timestamp_folder)
    os.makedirs(run_dir, exist_ok=True)

    # make an input flag --use_gpus
    available_gpu_ids = [5, 6, 7]

    # make an input flag
    logging_level = logging.INFO

    # point logging at run_dir/benchmark.log
    setup_logging(os.path.join(run_dir, "benchmark.log"), level=logging_level)
    log = logging.getLogger("humun_benchmark.benchmark")
    log.info(f"Writing all outputs into {run_dir}")

    if not datasets_path_env:
        log.warning("Warning: DATASETS_PATH environment variable not set. Please set it.")
    if not results_store_env:
        log.warning("Warning: RESULTS_STORE environment variable not set. Using './results'.")
        results_store_env = "results"

    # Call the parallel benchmark function
    benchmark(
        models=models_to_test,
        datasets_path=datasets_path_env,
        output_path=run_dir,
        series_ids=MD_VINTAGE_IDS_MONTHLY,
        n_datasets=10,
        batch_size=15,
        train_ratio=7,
        forecast_steps=6,
        context=True,
        available_gpu_ids=available_gpu_ids,
        level=logging_level,
    )
