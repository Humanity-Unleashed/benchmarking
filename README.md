<div align="center">

A tool to evaluate LLMs with time-series forecasting and policy NLP tasks, alongside other forecasting models.  

**Humun Org**

[Notion Page](https://humanity-unleashed.notion.site/LLM-Benchmarking-30835e8e64044ecaaddc84d4abcfdec8)
</div>

## Description

Instruct prompt method inspired by CiK forecasting [ [paper](https://arxiv.org/abs/2410.18959) | [github](https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/direct_prompt.py) ].


## Installation
> [!Note]
> Requires `uv`. See instructions [here](https://docs.astral.sh/uv/getting-started/installation/) if not available on your system.

```bash
make install
```

## Data
The dataset being used is economic timeseries data scraped from FRED by the Data Collection team. This data has been mounted to the server currently at -

* `/workspaces/datasets/fred/fred.parquet`

when read via `humun_benchmark.data.load_from_parquet()`, it assumes the format;   

```python 
# Dictionary of format:
    { "id1": { "history": pd.DataFrame, "forecast": pd.DataFrame, "title": str, "notes" : str },
      "id2": ... }
```
>![!Note] This data is truncated and split in the function - see function definition for details.  

## Environment Variables 
> [!Note]
> Alternative values can be provided at runtime via the benchmark.py or benchmark() parameters.

Contained in `.env` and loaded by `pydotenv`. 

| Variable Name | Description | Default Value |
|--------------|-------------|----------------|
| DATASETS_PATH | Path to FRED time series data parquet file | `/workspace/datasets/fred/fred.parquet` |
| RESULTS_STORE | Directory for storing benchmark results | `/workspace/pretraining/benchmarks` |
| HF_HOME | Directory for shared HuggingFace model cache | `/workspace/huggingface_cache` |
| HF_TOKEN_PATH | Path for HuggingFace authentication token | `~/.cache/huggingface/token` |
| HF_STORED_TOKENS_PATH | Path for additional HuggingFace tokens | *Auto-set based on HF_TOKEN_PATH* see [here](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py#L150)|


## Benchmarking Instructions

To run a benchmark, you can simply run the `benchmark.py` file, where a call is made to the function contained in the same file, using a set of config parameters which you can edit (arg parse will be re-added soon for easier config); 

    Required:
    * models: A dictionary containing models to benchmark.
        e.g. models = {
            "llm": [
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
                "Ministral-8B-Instruct-2410",
            ],
            "statistical": ['arima'],
        }

    Optional:
    * output_path: Where to store results
    * datasets_path: Path to time series data
    * series_ids: List of series IDs from FRED data
    * n_datasets: Number of datasets to retrieve (used with filters)
    * batch_size: Number of runs per inference
    * train_ratio: Multiplier for training period  
    * forecast_steps: Number of forecast steps
    * context: Bool for whether to include context or not for LLMs
    * available_gpu_ids: List of available GPU IDs to use. Tries to use all when not provided.
    * level: logging level (default is logging.INFO)


Generating forecasts - 
```bash
> make install
> source .venv/bin/activate
> python humun_benchmark/benchmark.py 
```

Results store. Uses .env + datetime string by default.
```bash
/workspace/pretraining/benchmarks/YYYYMMDD_HHMMSS/
  ├ benchmark.log
  ├ Qwen…parquet
  ├ meta-llama…parquet
  └ …
```

Calculating metrics - 
```python
from humun_benchmark.data.metrics import read_results, compute_all_metrics

paths = glob.glob(f"/workspace/pretraining/benchmarks/<folder_name>/*.parquet")
results = read_results(paths)
metrics = compute_all_metrics(results)
metrics['overall_metrics'] # pd.DataFrame of cross-dataset results for all models selected
```





