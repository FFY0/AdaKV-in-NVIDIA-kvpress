# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
import os
from fire import Fire
from infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from kvpress.ada_attn import replace_var_flash_attn
from kvpress.ada_cache import DynamicCacheSplitHeadFlatten
from loogle.calculate_metrics import calculate_metrics as loogle_scorer
from ruler.calculate_metrics import calculate_metrics as ruler_scorer
from tqdm import tqdm
from transformers import pipeline
from zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import time

from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    AdaSnapKVPress,
    AdaScorerPress
)

logger = logging.getLogger(__name__)

PRESS_DICT = {
    "expected_attention": ExpectedAttentionPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "random": RandomPress(),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "ada_snapkv": AdaSnapKVPress(),
    "fullkv": None,
}

@torch.inference_mode()
def efficiency_evaluate(
    model,
    tokenizer,
    press,
    context_length: int = 4 * 1024,
    compression_ratio: float = None,
    budget: int = None,
):
    """
    Evaluate a model on a dataset using a press and save the results
    """
    # assert compression_ratio is not None or budget is not None, "Either compression_ratio or budget must be provided"
    compression_ratio = 1 - budget / context_length if budget is not None else compression_ratio
    assert compression_ratio is None or compression_ratio > 0
    
    # fullkv press is None
    if press is not None:
        press.compression_ratio = compression_ratio

    prompt= "The quick brown fox jumps over the lazy dog." * (context_length)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = input_ids[:, :context_length]

    context_length=input_ids.shape[1]

    if isinstance(press, AdaScorerPress):
        cache = DynamicCacheSplitHeadFlatten()
    else:
        cache = DynamicCache()

    position_ids = torch.arange(
        0, context_length, device=model.device
    ).unsqueeze(0)
    #NOTE cache budget = ( 1 - compression_ratio ) * context_length
    torch.cuda.empty_cache()
    # prefill and compress kv cache
    with press(model) if press is not None else nullcontext():
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )
    position_ids = position_ids[:, -1:] + 1
    generated_ids = [outputs.logits[0, -1].argmax()]

    torch.cuda.synchronize()
    t = time.time()
    ave_token_num = 100
    for i in range(100):
        outputs = model(
            input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
            past_key_values=cache,
            position_ids=position_ids + i,
        )
        # new_id = outputs.logits[0, -1].argmax()

    torch.cuda.synchronize()
    t = time.time() - t
    decoding_latency = t / ave_token_num
    max_memory_allocated = torch.cuda.max_memory_allocated("cuda") / (1024 ** 3)
    max_memory_reserved = torch.cuda.max_memory_reserved("cuda") / (1024 ** 3)
    return decoding_latency, max_memory_allocated, max_memory_reserved



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model efficiency with different press methods.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--press_name", type=str, required=True, choices=PRESS_DICT.keys(), help="Name of the press method to use.")
    args = parser.parse_args()

    model_path = args.model_path
    press_name = args.press_name
    device = 'cuda:0'

    if device is None: device = "cuda:0" if torch.cuda.is_available() else "cpu"


    press = PRESS_DICT[press_name]
    if isinstance(press, ObservedAttentionPress):
        model_kwargs = {"attn_implementation": "eager"}
    # Support AdaKV
    elif isinstance(press, AdaScorerPress):
        replace_var_flash_attn(model=model_path)
        model_kwargs = {"attn_implementation": "flash_attention_2"}
    else:
        try:
            import flash_attn  # noqa: F401
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        except ImportError:
            model_kwargs = {}

    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            **model_kwargs
            )
    
    tokenizer=AutoTokenizer.from_pretrained(model_path)    

    print("press_name: ", press_name)
    budget = 1024
    if press_name == "fullkv":
        budget = None
    for context_length in [4*1024, 8*1024, 16*1024, 32*1024]:
        decoding_latency, max_memory_allocated, max_memory_reserved = efficiency_evaluate(model=model,tokenizer=tokenizer,press=press, budget = budget, context_length=context_length)
        print("=====================================")
        print("budget", budget, "context_length:", context_length, "decoding_latency(s)", f"{decoding_latency:.2f}", "max_memory_allocated(GB)", f"{max_memory_allocated:.2f}", "max_memory_reserved(GB)", f"{max_memory_reserved:.2f}")
        print("=====================================")
