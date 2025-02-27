<div align="center">

A tool to systematically test LLMs with time-series forecasting and policy NLP tasks.  

**Humun Org**

[Notion Page](https://humanity-unleashed.notion.site/LLM-Benchmarking-30835e8e64044ecaaddc84d4abcfdec8)
</div>

## Description

Instruct prompt method inspired by CiK forecasting [ [paper](https://arxiv.org/abs/2410.18959) | [github](https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/direct_prompt.py) ].


## Installation
> [!Note]
> Requires `uv` installed. See instructions [here](https://docs.astral.sh/uv/getting-started/installation/) if not available on your system.

```bash
make install
```

## Data
> [!Note]
> If data is no longer available at these paths, consult the Data Collection Notion [here.](https://humanity-unleashed.notion.site/Data-Collection-131d57b83b518183b5ddc38872f6bd6e)

The dataset being used is economic timeseries data scraped from FRED by the Data Collection team. This data has been mounted to the server currently at `/workspace/datasets/fred` and is stored in parquet files. 
Data is split into two parts:

1. Time-series data:
    * contained in a `.parquet` file stored at `/workspaces/datasets/fred/split.parquet`
    * when read into a pd.DataFrame, it assumes the format;   
```python 
Columns: Index(['series_id', 'history', 'forecast'], dtype='object')
```
2. Metadata:
    * ~~contained in a `.csv` file stored at `/workspaces/datasets/fred/all_fred_metadata.csv`~~ (Note: Currently there is no space to store the metadata file on /workspace, it can be downloaded via `humun_benchmark/adhoc/downloadGC.py` after API key has been retrieved from Notion- see file comments for link) 
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

Contained in `.env` and read-in when `humun_benchmark.utils.checks.check_env()` is imported. 

| Variable Name | Description | Default Value |
|--------------|-------------|----------------|
| DATASETS_PATH | Path to FRED time series data parquet file | `/workspace/datasets/fred/split.parquet` |
| METADATA_PATH | Path to FRED series metadata CSV | `/workspace/datasets/fred/all_fred_metadata.csv` |
| RESULTS_STORE | Directory for storing benchmark results | `/workspace/pretraining/benchmarks` |
| HF_HOME | Directory for shared HuggingFace model cache | `/workspace/huggingface_cache` |
| HF_TOKEN_PATH | Path for HuggingFace authentication token | `~/.cache/huggingface/token` |
| HF_STORED_TOKENS_PATH | Path for additional HuggingFace tokens | *Auto-set based on HF_TOKEN_PATH* see [here](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py#L150)|



