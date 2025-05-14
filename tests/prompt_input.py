import pandas as pd

from humun_benchmark.models.llm import InstructPrompt
from humun_benchmark.config import NUMERICAL

import logging

log = logging.getLogger(__name__)


if __name__ == "__main__":
    test_data = pd.read_csv("tests/test_data.csv")

    n_steps = 12
    n_history = n_steps * 5
    total = len(test_data)

    history = test_data.iloc[total - n_history : total - n_steps].copy()
    forecast = test_data.iloc[total - n_steps : total].copy()

    prompt_context = "hello I am your prompt context"

    raw_payload = InstructPrompt(
        task=NUMERICAL,
        history=history,
        forecast=forecast,
    )

    context_payload = InstructPrompt(
        task=NUMERICAL,
        history=history,
        forecast=forecast,
        context=prompt_context,
    )

    print("\n=== Prompt without context ===\n")
    print(raw_payload.prompt_text)
    print("\n=== Prompt with context ===\n")
    print(context_payload.prompt_text)
