from humun_benchmark.models import HuggingFace, StatisticalModel, MLModel

MODEL_REGISTRY = {
    "llm": HuggingFace,
    "my_arima": StatisticalModel,
    "my_xgboost": MLModel,
    # …
}
