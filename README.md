[![PyPI version](https://badge.fury.io/py/kvpress.svg)](https://badge.fury.io/py/kvpress)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Colab example notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP?usp=drive_link)

![kvpress](kvpress.jpg)

Deploying long-context LLMs is costly due to the linear growth of the key-value (KV) cache in transformer models. For example, handling 1M tokens with Llama 3.1-70B in float16 requires up to 330GB of memory. This repository implements multiple KV cache pruning methods and benchmarks using [🤗 transformers](https://huggingface.co/docs/transformers/en/index), aiming to simplify the development of new methods for researchers and developers in this field.

## A custom implementation of AdaKV under NVIDIA/kvpress open-source project!

In this fork, we have implemented AdaKV under KVPress with a custom CUDA kernel, enabling easy customization of head-specific compression. Additionally, the official (NVIDIA/KVPress)[https://github.com/NVIDIA/kvpress] repository provides a simpler way to simulate AdaKV's performance. The key difference lies in whether actual compression is achieved. The official code offers a fast and convenient starting point, and this repository allows you to test the practical compression benefits likes peak memory usage and decoding latency.
Additionally, there are other implementations of AdaKV available. For example, [Cloudflare](https://github.com/IsaacRe/vllm-kvcompress) provides an AdaKV implementation integrated into VLLM, alongside the (original AdaKV code)[https://github.com/FFY0/AdaKV]. We encourage everyone to explore these versions, and we hope they can be helpful to your work.

## Custom Evaluation
![RULER](evaluation/assets/ruler_4096_average%20score.png)


## Install
```bash
pip install kvpress
```

We recommend using [flash attention](https://github.com/Dao-AILab/flash-attention/) if possible:
```bash
pip install flash-attn --no-build-isolation
```

## Usage

This repository provides a set of "presses" that compress the KV cache. A press is only applied during the pre-filling phase and is associated with a `compression_ratio` parameter that measures the compression of the cache. The easiest way to use a press is through our custom `KVPressTextGenerationPipeline` that is automatically registered as a transformers pipeline with the name "kv-press-text-generation" when kvpress is imported. It handles chat templates and tokenization for you:



```python
from kvpress import ExpectedAttentionPress
from transformers import pipeline

device = "cuda:0"
model= "microsoft/Phi-3.5-mini-instruct"
pipe = pipeline("kv-press-text-generation", model=model, device=device, torch_dtype="auto", model_kwargs={"attn_implementation":"flash_attention_2"})

context = "A very long text you want to compress once and for all"
question = "\nA question about the compressed context" # optional
    
press = ExpectedAttentionPress(compression_ratio=0.4)
answer = pipe(context, question=question, press=press)["answer"]
```

In the snippet above, the compression is only applied on the context tokens so that you can evaluate the compression for different questions. Check the [Wikipedia notebook demo](notebooks/wikipedia_demo.ipynb) for a more detailed example.

> [!IMPORTANT]  
> We focus on compression during the pre-filling phase as the KV cache becomes a bottleneck for long-context sequence (100k - 1M tokens) which are essentially long context prompts. This would typically apply to improving prompt caching systems.

> [!NOTE]  
> To use the `ObservedAttentionPress`, use `model_kwargs={"attn_implementation":"eager"}` in order to materialize the attention weights (this method is not compatible with flash attention).
