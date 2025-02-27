from humun_benchmark import benchmark
from humun_benchmark.config.common import SERIES_IDS
from humun_benchmark.config.logs import setup_logging
import os

import logging

from dotenv import load_dotenv

load_dotenv()

setup_logging("test_benchmark.log")

log = logging.getLogger("tests.test_benchmark")

if __name__ == "__main__":
    benchmark(
        models=["llama-3.1-8b-instruct"],
        output_path=os.getenv("TESTS_STORE"),
        metadata_path=os.getenv("METADATA_PATH"),
        datasets_path=os.getenv("DATASETS_PATH"),
        selector=SERIES_IDS,
        n_datasets=3,
        batch_size=1,
        train_ratio=3,
        n_steps=12,
    )
