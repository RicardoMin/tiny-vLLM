<p align="center">
  <img src="assets/logo.png" alt="Tiny-vLLM" width="680">
</p>

# tiny-vLLM

**English** | [简体中文](README.zh-CN.md)

A compact single-GPU inference engine for dense Qwen3 models, built with PyTorch and CUDA.

**Paged KV · Prefix caching · Continuous batching · Chunked prefill · CUDA Graphs · Fused CUDA kernels**

## Installation

Linux / WSL2, Python 3.11+, an NVIDIA GPU and driver, **CUDA Toolkit 12.8** with `nvcc`, and a compatible C++ compiler.

```bash
git clone https://github.com/RicardoMin/tiny-vllm.git
cd tiny-vllm

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[kernels,server]'

# Use your CUDA 12.8 Toolkit path.
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

export MODEL="$HOME/models/Qwen3/Qwen3-0.6B"
hf download Qwen/Qwen3-0.6B --local-dir "$MODEL"
```

The Toolkit must match PyTorch's CUDA version. Custom CUDA extensions compile on first initialization and are cached. The default CUDA backend requires no FlashInfer; run `python -m pip install -e '.[flashinfer]'` and select `--attention-backend flashinfer` to use the optional backend.

## Usage

### Python

```python
from pathlib import Path
from tinyvllm import EngineConfig, LLM, SamplingParams

config = EngineConfig(
    model=Path.home() / "models/Qwen3/Qwen3-0.6B",
    max_model_len=2048,
    kv_cache_memory_bytes=512 * 1024**2,
)

with LLM(config) as llm:
    reply = llm.chat(
        [{"role": "user", "content": "请介绍一下你自己"}],
        SamplingParams(max_new_tokens=128),
    )
    print(reply.text)
```

Use `llm.generate([prompt1, prompt2], sampling)` for batch completions. `AsyncLLM` provides concurrent generation and streaming via `stream()` / `chat_stream()`. See [example.py](example.py) for a minimal completion example.

### Terminal

```bash
# Local multi-turn chat.
python -m tinyvllm --model "$MODEL" --chat --max-new-tokens 256

# Stream a single prompt.
python -m tinyvllm --model "$MODEL" --prompt "The capital of France is" --stream
```

Chat commands: `/help`, `/reset`, `/exit`; Ctrl-C cancels the current generation.

### HTTP

```bash
python -m tinyvllm.server --model "$MODEL" --served-model-name Qwen3-0.6B
```

From another terminal:

```bash
python -m tinyvllm.chat --base-url http://127.0.0.1:8000/v1 --model Qwen3-0.6B

# Or request an SSE chat stream directly.
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"请介绍一下你自己"}],"max_tokens":128,"stream":true}'
```

Endpoints: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`. OpenAI-style text completion and SSE, with one output per request. Run any CLI entry with `--help` for its options.

## Performance

**Qwen3-0.6B · BF16 · RTX 5060 Laptop 8 GiB**

### Inference throughput and latency

![Inference performance across eight workloads](assets/performance.svg)

![Values and relative improvements for the staggered-arrival workload](assets/arrivals-performance.svg)

![Synchronized measured replay of tiny-vLLM and nano-vLLM answering the same prompt](assets/first-answer.gif)

## Architecture

![tiny-vLLM architecture](assets/architecture.svg)

## Acknowledgements

[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) · [vLLM](https://github.com/vllm-project/vllm) · [SGLang](https://github.com/sgl-project/sglang) · [PyTorch](https://pytorch.org/) · [Qwen3](https://github.com/QwenLM/Qwen3) · [Transformers](https://github.com/huggingface/transformers) · [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
