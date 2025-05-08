"""
Instruct prompt method inspired by CiK forecasting:
    - https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/hf_utils/dp_hf_api.py

LMs configured with Flash-Attention-2 for efficiency:
    - https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
"""

import logging
from typing import Dict, Optional, Union

import pandas as pd
import torch
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)
from transformers import (
    pipeline as hf_pipeline,
)

from humun_benchmark.config.common import NUMERICAL

from ..base import InferenceError, Model, ModelLoadError
from .formatting import format_output_regex, parse_forecast_output
from .prompts import InstructPrompt

log = logging.getLogger(__name__)


DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_model_and_tokenizer(llm_identifier, cuda_device_map):
    if "/" not in llm_identifier:
        if "llama" in llm_identifier.lower():
            llm_identifier = "meta-llama/" + llm_identifier
        elif "mistral" in llm_identifier.lower():
            llm_identifier = "mistralai/" + llm_identifier
        elif "qwen" in llm_identifier.lower():
            llm_identifier = "Qwen/" + llm_identifier
    try:
        if "llama-3.1" in llm_identifier:
            tokenizer = AutoTokenizer.from_pretrained(llm_identifier, padding_side="left", legacy=False)
            model = LlamaForCausalLM.from_pretrained(
                llm_identifier,
                device_map=cuda_device_map,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                llm_identifier,
                device_map=cuda_device_map,
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(llm_identifier)

        special_tokens_dict = dict()
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        if special_tokens_dict:
            tokenizer.add_special_tokens(special_tokens_dict)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ensure model pad_token_id is also aligned if it exists as a config attribute
        if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id

        log.info(f"Model '{llm_identifier}' and tokenizer loaded successfully.")
        return model.eval(), tokenizer

    except Exception as e:
        raise ModelLoadError(f"Failed to load model '{llm_identifier}' from Hugging Face: {e}")


class HuggingFace(Model):
    def __init__(self, label: str, cuda: Optional[Union[int, str]] = None, **model_kwargs):
        super().__init__(label, model_type="llm")
        self.cuda_setting = cuda
        self.model_kwargs = model_kwargs
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        if isinstance(cuda, int):
            # Use specific GPU ID provided
            self.device_map_for_loading = {"": f"cuda:{cuda}"}
        elif isinstance(cuda, str) and cuda.startswith("cuda:"):
            self.device_map_for_loading = {"": cuda}
        else:  # "auto", None, or other strings
            self.device_map_for_loading = "auto"

    def _load_model(self, **kwargs):
        """Loads the model/tokenizer and prepares the inference pipeline."""
        try:
            self.model, self.tokenizer = get_model_and_tokenizer(self.label, self.device_map_for_loading)

            pipeline_args = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
            }

            if isinstance(self.cuda_setting, int):
                pipeline_args["device"] = self.cuda_setting  #  Pass the GPU ID directly

            self.pipeline = hf_pipeline(**pipeline_args)
            log.info(f"HuggingFace model '{self.label}' loaded. Pipeline device: {self.pipeline.device}")

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model '{self.label}' with cuda_setting='{self.cuda_setting}': {e}"
            ) from e

    @torch.inference_mode()
    def predict(
        self,
        data: Dict[str, Union[pd.DataFrame, str]],
        context: bool,
        batch_size=1,
        temperature=1.0,
        task_str: str = NUMERICAL,
    ) -> pd.DataFrame:
        """
        Run LLM model inference.

        Params:
        """

        if self.pipeline is None:
            self._load_model()

        prompt_context = data["dataset_info"] if context else None

        payload = InstructPrompt(
            task=task_str,
            history=data["history"],
            forecast=data["forecast"],
            context=prompt_context,
        )

        future_timestamps = payload.forecast["date"]

        # Build a regex parser to constrain output
        parser = RegexParser(format_output_regex(future_timestamps))
        prefix_function = build_transformers_prefix_allowed_tokens_fn(self.pipeline.tokenizer, parser)

        # Log prompt length before inference
        try:
            prompt_tokens = len(self.tokenizer.encode(payload.prompt_text))
            log.info(f"Prompt Tokens Length: {prompt_tokens}")
        except Exception as e:
            log.warning(f"Could not encode prompt text for token length calculation: {e}")

        # Use the pre-created pipeline (self.pipeline) for inference.
        for response in self.pipeline(
            [payload.prompt_text] * batch_size,
            max_new_tokens=6000,  # Limit response length
            temperature=temperature,
            prefix_allowed_tokens_fn=prefix_function,
            batch_size=batch_size,
        ):
            output_start = len(payload.prompt_text)
            forecast_output = response[0]["generated_text"][output_start:]
            payload.responses.append(forecast_output)

        # Turn text responses into a results dataframe
        try:
            dfs = [parse_forecast_output(r) for r in payload.responses]
        except Exception as e:
            raise InferenceError(f"Failed to parse model output: {e}")

        return payload.merge_forecasts(dfs)

    def serialise(self):
        if not self.model or not self.tokenizer:
            return {
                "model_name": self.label,
                "error": "Model not loaded",
            }

        config = self.model.config.to_dict()
        return {
            "model_name": self.label,
            "model_class": self.model.__class__.__name__,
            "model_config_summary": {
                "architectures": config.get("architectures"),
                "attention_implementation": config.get("attn_implementation"),
                "torch_dtype": str(config.get("torch_dtype")),
                "vocab_size": config.get("vocab_size"),
            },
            "tokenizer_info": {
                "tokenizer_class": self.tokenizer.__class__.__name__,
                "vocab_size": len(self.tokenizer),
                "padding_side": self.tokenizer.padding_side,
                "eos_token": self.tokenizer.eos_token,
                "bos_token": self.tokenizer.bos_token,
                "pad_token": self.tokenizer.pad_token,
            },
            "pipeline_device": str(self.pipeline.device) if self.pipeline else "N/A",
            "model_on_device": str(self.model.device) if self.model else "N/A",
        }
