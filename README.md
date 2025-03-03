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

## Installation
> [!Note]
> Requires `uv` installed. See instructions [here](https://docs.astral.sh/uv/getting-started/installation/) if not available on your system.

```bash
make install
```

## Data
> [!Note]
> If data is no longer available at these paths, consult the Data Collection Notion [here.](https://humanity-unleashed.notion.site/Data-Collection-131d57b83b518183b5ddc38872f6bd6e)

The dataset being used is economic timeseries data scraped from FRED by the Data Collection team. This data has been mounted to the server currently at `/workspace/datasets/fred`.

Data is split into two parts:

1. Time-series data:
    * `/workspaces/datasets/fred/split.parquet`
    * when read into a pd.DataFrame, it assumes the format;   
```python 
Columns: Index(['series_id', 'history', 'forecast'], dtype='object')
```

2. Metadata:
    * `/workspaces/datasets/fred/all_fred_metadata.csv`
    * it assumes the format; 
```python 
Data columns (total 16 columns):
 #   Column                     Non-Null Count   Dtype 
---  ------                     --------------   ----- 
 0   id                         825282 non-null  object # note: 'id' links to 'series_id' above
 1   realtime_start             825282 non-null  object
 2   realtime_end               825282 non-null  object
 3   title                      825282 non-null  object
 4   observation_start          825282 non-null  object
 5   observation_end            825282 non-null  object
 6   frequency                  825282 non-null  object
 7   frequency_short            825247 non-null  object
 8   units                      825282 non-null  object
 9   units_short                825282 non-null  object
 10  seasonal_adjustment        825282 non-null  object
 11  seasonal_adjustment_short  825282 non-null  object
 12  last_updated               825282 non-null  object
 13  popularity                 825282 non-null  int64 
 14  group_popularity           825282 non-null  int64 
 15  notes                      753933 non-null  object
```



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



