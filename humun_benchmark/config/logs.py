import atexit
import logging
import logging.config
import logging.handlers
import time

from transformers.utils import logging as hf_utils_logging


class StdoutFilter(logging.Filter):
    """Allow only INFO and lower logs to stdout"""

    def filter(self, record):
        return record.levelno <= logging.INFO


class StderrFilter(logging.Filter):
    """Allow only WARNING and higher logs to stderr"""

    def filter(self, record):
        return record.levelno > logging.INFO


def setup_logging(
    log_filepath: str = None,
    filter_shards: bool = True,
    level: int = logging.INFO,
):
    """
    Sets up simple plaintext logging for files and console output.

    Args:
      log_filepath: if provided, logs are also written (in append mode) to this file
      filter_shards: if True, only the 100% shard‐loading line is shown
      level: minimum level for root + humun_benchmark + __main__ (e.g. INFO, DEBUG, WARNING)
    """
    # clear any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    logging.Formatter.converter = time.gmtime

    basic_handlers = ["console_stdout", "console_stderr"]
    handlers = {
        "console_stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "detailed",
            "filters": ["stdout_filter"],
        },
        "console_stderr": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "detailed",
            "filters": ["stderr_filter"],
        },
    }

    if log_filepath:
        handlers["output_log"] = {
            "class": "logging.FileHandler",
            "filename": log_filepath,
            "mode": "a",
            "formatter": "detailed",
        }

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "[%(asctime)s] %(levelname)s - %(name)s:\n %(message)s\n",
                "datefmt": "%Y-%m-%d %H:%M:%S UTC",
            }
        },
        "handlers": handlers,
        "filters": {
            "stdout_filter": {"()": StdoutFilter},
            "stderr_filter": {"()": StderrFilter},
        },
        "loggers": {
            "humun_benchmark": {
                "level": level,
                "handlers": basic_handlers + (["output_log"] if log_filepath else []),
                "propagate": False,
            },
            "tests": {
                "level": level,
                "handlers": basic_handlers + (["output_log"] if log_filepath else []),
                "propagate": True,
            },
        },
        "root": {
            "level": level,
            "handlers": basic_handlers + (["output_log"] if log_filepath else []),
        },
    }

    logging.config.dictConfig(log_config)

    if filter_shards:
        hf_utils_logging.disable_progress_bar()

    if log_filepath:
        log = logging.getLogger(__name__)
        log.debug(f"Writing logs to {log_filepath}")

    # ensure logging is shutdown automatically on exit
    atexit.register(logging.shutdown)
