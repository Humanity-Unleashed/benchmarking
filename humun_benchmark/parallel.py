import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import pandas as pd
import torch
import concurrent.futures
import multiprocessing

from humun_benchmark.config import MD_VINTAGE_IDS_MONTHLY, NUMERICAL, setup_logging
from humun_benchmark.data import load_from_parquet
from humun_benchmark.models import HuggingFace
from humun_benchmark.prompts import InstructPrompt

import dotenv

dotenv.load_dotenv()

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

setup_logging(os.path.join(os.getenv("RESULTS_STORE", "results"), f"benchmark_run_{RUN_TIMESTAMP}.log"))
log = logging.getLogger(__name__)


def run_single_model_benchmark(
    model_name: str,
    gpu_id: int,
    fred_data: Dict[str, Dict[str, Any]],
    output_path: str,
    batch_size: int,
    context: bool,
    run_timestamp: str,  # Use the shared timestamp
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
        run_timestamp: The shared timestamp for the overall benchmark run.
    """
    # Optional: Configure logging specific to this worker process
    # E.g., add model name and GPU ID to log messages, or log to a model-specific file.
    worker_log = logging.getLogger(f"{__name__}.worker.{model_name.split('/')[-1]}.GPU{gpu_id}")
    worker_log.info(f"Process started for model '{model_name}' on GPU {gpu_id}")

    try:
        # --- Crucial: Set CUDA device for this process ---
        torch.cuda.set_device(gpu_id)
        device_name = torch.cuda.get_device_name(gpu_id)
        worker_log.info(f"Successfully set device to GPU {gpu_id} ({device_name})")

        # --- Load Model ---
        worker_log.info(f"Loading model '{model_name}' onto GPU {gpu_id}")
        # Pass the specific gpu_id to the model constructor
        llm = HuggingFace(model_name, cuda=gpu_id)
        model_info = llm.serialise()
        worker_log.info(f"Model loaded. Info:\n{model_info}")
        worker_log.info(f"Model is on device: {llm.model.device}")  # Verify device placement

        model_result = {"dataset_info": {}, "results": {}}

        # --- Iterate through datasets for this model ---
        for series_id, data in fred_data.items():
            worker_log.info(f"Processing dataset: {series_id}")
            model_result["dataset_info"][series_id] = data["dataset_info"]

            prompt_context = data["dataset_info"] if context else None

            prompt = InstructPrompt(
                task=NUMERICAL,
                history=data["history"],
                forecast=data["forecast"],
                context=prompt_context,
            )

            # Log prompt length before inference
            try:
                prompt_tokens = len(llm.tokenizer.encode(prompt.prompt_text))
                worker_log.info(f"Prompt Tokens Length: {prompt_tokens}")
            except Exception as e:
                worker_log.warning(f"Could not encode prompt text for token length calculation: {e}")

            worker_log.info(
                f"Starting inference for {series_id}..."
                f" Batch size: {batch_size}"
                f" On device: {llm.model.device}"  # Re-confirm device just before inference
            )

            # Run Inference
            try:
                llm.inference(payload=prompt, batch_size=batch_size)
                model_result["results"][series_id] = prompt.results.to_json(
                    orient="records", date_format="iso"
                )
                worker_log.info(f"Inference completed for {series_id}")
            except Exception as e:
                worker_log.error(f"Inference failed for dataset {series_id}: {e}", exc_info=True)
                # Decide how to handle failed inference: skip dataset, mark as failed, etc.
                model_result["results"][series_id] = json.dumps({"error": str(e)})  # Store error info

            torch.cuda.empty_cache()

        # save model results
        output_df = pd.DataFrame(
            [
                {
                    "model_info": model_info,
                    "dataset_info": json.dumps(model_result["dataset_info"]),
                    "results": json.dumps(model_result["results"]),
                }
            ]
        )
        # Use the shared run_timestamp for consistent naming
        output_file = os.path.join(output_path, f"{model_name.split('/')[-1]}_{run_timestamp}.parquet")
        output_df.to_parquet(output_file, index=False)
        worker_log.info(f"Model output written to {output_file}")

        # clean up memory
        del llm
        torch.cuda.empty_cache()
        worker_log.info(f"Finished processing model '{model_name}' on GPU {gpu_id}")

    except Exception as e:
        # Log any exceptions that occur during the entire process for this model
        worker_log.error(
            f"FATAL error processing model '{model_name}' on GPU {gpu_id}: {e}", exc_info=True
        )
        # Optionally, write an error marker file
        error_file = os.path.join(output_path, f"{model_name.split('/')[-1]}_{run_timestamp}.ERROR")
        with open(error_file, "w") as f:
            f.write(f"Error processing model {model_name} on GPU {gpu_id}:\n{e}\n")
            import traceback

            traceback.print_exc(file=f)


def benchmark(
    models: list[str],
    output_path: str = os.getenv("RESULTS_STORE", "results"),  # Default if env var not set
    datasets_path: str = os.getenv("DATASETS_PATH"),
    series_ids: List[str] = MD_VINTAGE_IDS_MONTHLY,
    n_datasets: Optional[int] = None,  # Allow None to use all series_ids
    batch_size: int = 10,
    train_ratio: int = 3,
    forecast_steps: int = 12,
    context: bool = False,
    available_gpu_ids: List[int] = [],
) -> None:
    """
    Run benchmarks concurrently on available GPUs, assigning one model per GPU.

    Args:
        models: List of model names to benchmark.
        output_path: Where to store results.
        datasets_path: Path to time series data.
        series_ids: List of series IDs from FRED data.
        n_datasets: Max number of datasets to load (loads all if None or > len(series_ids)).
        batch_size: Number of runs per inference (passed to worker).
        train_ratio: Multiplier for training period.
        forecast_steps: Number of forecast steps.
        context: Supply title and notes per dataset as context.
    """
    # --- Basic Setup & Validation ---
    if not datasets_path:
        log.error("DATASETS_PATH environment variable not set.")
        raise ValueError("datasets_path must be provided.")
    if not output_path:
        log.warning("RESULTS_STORE environment variable not set. Defaulting to './results'")
        output_path = "results"
    os.makedirs(output_path, exist_ok=True)  # Ensure output dir exists

    if not torch.cuda.is_available():
        log.error("CUDA is not available. Cannot run on GPUs.")
        # Fallback? Or raise error? For now, raise error.
        raise RuntimeError("CUDA is required but not available.")

    num_gpus = len(available_gpu_ids) if available_gpu_ids else torch.cuda.device_count()
    if num_gpus == 0:
        log.error("No CUDA GPUs found.")
        raise RuntimeError("No CUDA GPUs detected.")

    log.info(f"Detected {num_gpus} CUDA GPUs.")

    # Determine actual series IDs to use
    if n_datasets is not None and n_datasets < len(series_ids):
        selected_series_ids = series_ids[:n_datasets]
        log.info(f"Using the first {n_datasets} series IDs.")
    else:
        selected_series_ids = series_ids
        n_datasets = len(selected_series_ids)  # Update n_datasets for logging
        log.info(f"Using all {n_datasets} provided series IDs.")

    # Log parameters
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
        "num_gpus_used": min(len(models), num_gpus),
        "run_timestamp": RUN_TIMESTAMP,
    }
    params_str = "\n".join(f"\t{k}: {v}" for k, v in params.items())
    log.info(f"Benchmark Parameters: {{\n{params_str}\n}}")

    # Load all necessary data before starting parallel processes
    log.info(f"Loading {len(selected_series_ids)} datasets...")
    try:
        fred_data = load_from_parquet(
            series_ids=selected_series_ids,
            datasets_path=datasets_path,
            forecast_steps=forecast_steps,
            train_ratio=train_ratio,
        )
        log.info("Dataset loading complete.")
        if not fred_data:
            log.error("No data loaded. Check series IDs and dataset path.")
            return
    except Exception as e:
        log.error(f"Failed to load data: {e}", exc_info=True)
        raise

    # Execute benchmarks in parallel,
    # if n_models > max_workers, models will be ran after a GPU becomes available
    futures = []
    log.info(f"Starting benchmark tasks using up to {num_gpus} GPUs concurrently.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        for i, model_name in enumerate(models):
            gpu_id = available_gpu_ids[i % num_gpus]  # Assign GPU in round-robin fashion
            log.info(f"Submitting task for model '{model_name}' to run on GPU {gpu_id}")
            future = executor.submit(
                run_single_model_benchmark,
                model_name=model_name,
                gpu_id=gpu_id,
                fred_data=fred_data,  # Pass the preloaded data
                output_path=output_path,
                batch_size=batch_size,
                context=context,
                run_timestamp=RUN_TIMESTAMP,  # Pass the shared timestamp
            )
            futures.append(future)

        # --- Wait for Completion & Handle Results/Errors ---
        log.info("All benchmark tasks submitted. Waiting for completion...")
        completed_count = 0
        failed_count = 0
        # Use as_completed to process results as they finish (optional, good for large runs)
        for future in concurrent.futures.as_completed(futures):
            try:
                # Retrieve result (None in this case, as worker function doesn't return)
                # This call will re-raise any exceptions that occurred in the worker process
                future.result()
                completed_count += 1
                log.info(
                    f"A model benchmark task completed successfully ({completed_count}/{len(models)})."
                )
            except Exception as e:
                failed_count += 1
                # Logging of the specific error is handled within the worker's except block
                log.error(f"A model benchmark task failed. See worker log for details. Error: {e}")

    log.info("-" * 30)
    log.info(f"Benchmark run {RUN_TIMESTAMP} completed.")
    log.info(f"Total models processed: {len(models)}")
    log.info(f"Successful: {completed_count}")
    log.info(f"Failed: {failed_count}")
    log.info("-" * 30)


if __name__ == "__main__":
    try:
        # Set start method to 'spawn' - Necessary for CUDA on Linux AND macOS
        multiprocessing.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Warning: Could not set multiprocessing start method ('{e}'). Using default.")
        pass
    # Define models and other parameters here
    models_to_test = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Ministral-8B-Instruct-2410",
    ]

    # Make sure environment variables are loaded or paths are set correctly
    datasets_path_env = os.getenv("DATASETS_PATH")
    results_store_env = os.getenv("RESULTS_STORE")

    if not datasets_path_env:
        print("Warning: DATASETS_PATH environment variable not set. Please set it.")
        # exit(1) # Or set a default path for testing

    if not results_store_env:
        print("Warning: RESULTS_STORE environment variable not set. Using './results'.")
        results_store_env = "results"

    available_gpu_ids = [5, 6, 7]

    # Call the parallel benchmark function
    benchmark(
        models=models_to_test,
        datasets_path=datasets_path_env,
        output_path=results_store_env,
        series_ids=MD_VINTAGE_IDS_MONTHLY,
        n_datasets=5,
        batch_size=15,
        train_ratio=3,
        forecast_steps=12,
        context=True,
        available_gpu_ids=available_gpu_ids,
    )
