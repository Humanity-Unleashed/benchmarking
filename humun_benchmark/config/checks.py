import logging
import os
from .common import ENV_VARS

log = logging.getLogger(__name__)


def check_env(vars=ENV_VARS):
    """
    Checks if any required environment variables are missing.
    Logs a warning if any are missing.
    """
    missing = [var for var in vars if os.getenv(var) is None]
    if missing:
        log.warning(f"Missing environment variables: {missing}. Check your setup.")
        return False
    return True
