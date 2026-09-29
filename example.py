"""Minimal local Qwen3 text generation example."""

import argparse
from pathlib import Path

from tinyvllm import EngineConfig, LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "models/Qwen3/Qwen3-0.6B",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    config = EngineConfig(
        model=args.model,
        attention_backend="cuda",
        max_model_len=1024,
        kv_cache_memory_bytes=512 * 1024**2,
    )
    sampling = SamplingParams(max_new_tokens=32, temperature=0.0)

    with LLM(config) as llm:
        output = llm.generate(args.prompt, sampling)[0]
        print(output.text)


if __name__ == "__main__":
    main()
