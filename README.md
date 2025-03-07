<div align="center">

A tool to evaluate LLMs with time-series forecasting and policy NLP tasks.  

**Humun Org**

[Notion Page](https://humanity-unleashed.notion.site/LLM-Benchmarking-30835e8e64044ecaaddc84d4abcfdec8)
</div>

## Description

Instruct prompt method inspired by CiK forecasting [ [paper](https://arxiv.org/abs/2410.18959) | [github](https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/direct_prompt.py) ].

## TODOs

* Parallelise model looping: Accelerate can be implemented to speed up inference/training, however we are still running through the models and datasets sequentially. Multi-threading the distribution of models to different GPUs could help speed things up, while still running the datasets in a sequential manner (for each model on it's own GPU- sequentially run through the datasets). Due to each timeseries having different timestamps, they each need their own pipeline due to a different prefix function.
* Use hugging-face chat template: as a lot of models using the transformers package are trained using the hugging face chat-template, it may be more advantageous to adopt this approach. Test to see if there are substantial changes in performance when inferencing. 
* refactor metrics to analyse benchmark results in new format. 
* configure logs for Multi-threading / accelerate (queue-based config)

## Installation
> [!Note]
> Requires `uv`. See instructions [here](https://docs.astral.sh/uv/getting-started/installation/) if not available on your system.

```bash
make install
```

## Data
The dataset being used is economic timeseries data scraped from FRED by the Data Collection team. This data has been mounted to the server currently at -

* `/workspaces/datasets/fred/split.parquet`

when read via `data.load_from_parquet()`, it assumes the format;   

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
| DATASETS_PATH | Path to FRED time series data parquet file | `/workspace/datasets/fred/split.parquet` |
| METADATA_PATH | Path to FRED series metadata CSV | `/workspace/datasets/fred/all_fred_metadata.csv` |
| RESULTS_STORE | Directory for storing benchmark results | `/workspace/pretraining/benchmarks` |
| HF_HOME | Directory for shared HuggingFace model cache | `/workspace/huggingface_cache` |
| HF_TOKEN_PATH | Path for HuggingFace authentication token | `~/.cache/huggingface/token` |
| HF_STORED_TOKENS_PATH | Path for additional HuggingFace tokens | *Auto-set based on HF_TOKEN_PATH* see [here](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py#L150)|


## Benchmarking Instructions

To run a benchmark, you can simply run the `benchmark.py` file, where a call is made to the function contained in the same file, using a set of config parameters which you can edit (arg parse will be re-added soon for easier config); 

    * output_path: Where to store results
    * datasets_path: Path to time series data
    * series_ids: List of series IDs from FRED data
    * n_datasets: Number of datasets to retrieve (used with filters)
    * batch_size: Number of runs per inference
    * train_ratio: Multiplier for training period  
    * forecast_steps: Number of forecast steps
    * cuda: Either an int (for a specific GPU), "accelerate" (to run in accelerate mode), or None.

Generating inference - 
```bash
> make install
> source .venv/bin/activate
> python humun_benchmark/benchmark.py 
```

Calculating metrics - 
```python
from humun_benchmark.data.metrics import read_results, compute_all_metrics

paths = glob.glob(f"/workspace/pretraining/benchmarks/*<some_datestamp>.parquet")
results = read_results(paths)
metrics = compute_all_metrics(results)
metrics['benchmark'] # pd.DataFrame of cross-dataset results for all models selected
```



